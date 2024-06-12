
import torch

from stable_audio_tools.interface.gradio import create_ui


def main(args):
    torch.manual_seed(42)

    interface = create_ui(
        model_config_path=args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        tmp_dir=args.tmp_dir
    )
    interface.queue()
    interface.launch(share=True, auth=(args.username, args.password) if args.username is not None else None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config. Ignored if pretrained-name is specified.', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint. Ignored if pretrained-name is specified.', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint.', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    parser.add_argument('--tmp-dir', type=str, default='', help="Temporary directory for saving output files")
    args = parser.parse_args()
    main(args)
