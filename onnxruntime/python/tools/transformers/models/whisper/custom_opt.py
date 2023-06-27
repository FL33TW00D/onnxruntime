from whisper_helper import WhisperHelper
import argparse


def main():
    encoder_attention_heads = 6
    hidden_size = 384
    use_external_data_format = False
    use_fp16 = False
    auto_mix_precision = False
    use_gpu = False
    provider = "CPUExecutionProvider"
    parser = argparse.ArgumentParser(description="Optimize ONNX model")
    parser.add_argument("--model_path", type=str, help="Path to ONNX model", required=True)
    parser.add_argument("--output_path", type=str, help="Path to output ONNX model", required=True)
    args = parser.parse_args()
    WhisperHelper.optimize_onnx(
        args.model_path,
        args.output_path,
        use_fp16,
        encoder_attention_heads,
        hidden_size,
        use_external_data_format,
        auto_mix_precision,
        use_gpu,
        provider,
    )

if __name__ == "__main__":
    main()
