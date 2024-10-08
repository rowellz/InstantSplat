import argparse

from instant_splat.coarse_init_infer import coarse_infer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_size", type=int, default=512, choices=[512, 224], help="image size"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="path to the model weights",
    )
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", action="store_true")

    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument(
        "--img_base_path",
        type=str,
        default="/home/workspace/datasets/instantsplat/Tanks/Barn/24_views",
    )

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    coarse_infer(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        schedule=args.schedule,
        lr=args.lr,
        niter=args.niter,
        n_views=args.n_views,
        img_base_path=args.img_base_path,
        focal_avg=args.focal_avg,
    )
