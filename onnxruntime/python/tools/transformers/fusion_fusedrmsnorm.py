# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Dict

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)

"""
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape)-1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(self.weight.dtype)

    return weight * input

"""


class FusionRMSNorm(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "FusedRMSNorm", "Pow")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse RMSNormalization subgraph into one node FusedRMSNorm:

              +---------------------------------------------------------+
              |                                                         |
              |                                                         v
          [Root] --> Pow --> ReduceMean -->  Add  --> Sqrt --> Div --> Mul --> Mul
                             (axis=2 or -1)  

        """
        print("Start fusing RMSNorm...")
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0 or len(children) > 2:
            print("RMSNorm: children nodes not found")
            return

        if children[0].op_type != "ReduceMean": 
            print("RMSNorm: ReduceMean node not found")
            return 

        first_mul_node = None
        for child in children:
            first_mul_node = self.model.find_first_child_by_type(child, "Mul", input_name_to_nodes, recursive=True)
            if first_mul_node is not None:
                break
        if first_mul_node is None:
            return

        
        print("First mul node: ", first_mul_node)
        path_id, parent_nodes, _ = self.model.match_parent_paths(
            first_mul_node,
            [
                (["Div", "Sqrt", "Add", "ReduceMean"], [None, None, None, None]),
            ],
            output_name_to_node,
        )
        if path_id < 0:
            print("RMSNorm: parent path not found")
            return

        for idx, parent_node in enumerate(parent_nodes):
            print("Parent node idx: ", idx, parent_node)

        add_node = parent_nodes[2]
        i, add_weight = self.model.get_constant_input(add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0e-4:
            logger.warning(f"epsilon value is not expected: {add_weight}")
            return

        last_mul_node = input_name_to_nodes[first_mul_node.output[0]][0]
        if last_mul_node.op_type != "Mul":
            print("RMSNorm: Mul node not found")
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend(children)
        subgraph_nodes.extend(parent_nodes[:-1])

        subgraph_nodes.extend([last_mul_node, first_mul_node])
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            last_mul_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            print(f"It is not safe to fuse RMSNormalization node. Skip")
            return

        weight_input = last_mul_node.input[1 - self.model.input_index(first_mul_node.output[0], last_mul_node)]
        print("Weight input: ", weight_input)
        if not self.model.is_constant_with_specified_dimension(weight_input, 1, "layernorm weight"):
            print("RMSNorm: layernorm weight is not constant")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        normalize_node = helper.make_node(
            "FusedRMSNorm",
            inputs=[node.input[0], weight_input],
            outputs=[last_mul_node.output[0]],
            name=self.model.create_node_name("FusedRMSNorm", name_prefix="FusedRMSNorm"),
        )
        print("Normalize node: ", normalize_node)
        normalize_node.attribute.extend([helper.make_attribute("epsilon", float(add_weight))])
        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name

