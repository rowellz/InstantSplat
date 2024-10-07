import gradio as gr
from gradio_rerun import Rerun

import rerun as rr
import rerun.blueprint as rrb

import sys
import mmcv
import numpy as np
from numpy import ndarray
import torch
import PIL
from pathlib import Path
from jaxtyping import UInt8
from typing import Generator

from instant_splat.coarse_init_infer import coarse_infer
import os
import numpy as np
from random import randint
from instant_splat.utils.loss_utils import l1_loss, ssim
from instant_splat.gaussian_renderer import render
from instant_splat.scene import Scene, GaussianModel
from tqdm import tqdm
from instant_splat.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from instant_splat.utils.pose_utils import get_camera_from_tensor
from instant_splat.train_joint import save_pose

from time import perf_counter


try:
    import spaces  # type: ignore # noqa: F401

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

zero = torch.Tensor([0]).cuda()
print(zero.device)  # <-- 'cpu' 🤔


def create_blueprint() -> rrb.Blueprint:
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="image/original"),
            rrb.Spatial2DView(origin="image/blurred"),
        ),
        collapse_panels=True,
    )
    return blueprint


def estimate_pose_fn(img):
    """
    This is required because of
    https://github.com/beartype/beartype/issues/423
    """
    yield from _estimate_pose_fn(img)


@rr.thread_local_stream("example stream")
def _estimate_pose_fn(
    input_files: list[str], progress=gr.Progress()
) -> Generator[bytes, None, None]:
    print(zero.device)
    # beartype causes gradio to break, so type hint after the fact, intead of on function definition
    stream: rr.BinaryStream = rr.binary_stream()

    if input_files is None or len(input_files) < 3:
        raise gr.Error("Must provide 3 or more images.")
    if len(input_files) > 12:
        raise gr.Warning(
            "More than 12 images will result in longer processing time, and potentially gpu time outs!"
        )

    progress(0.05, desc="Estimating Camera parameters with DUSt3R... please wait")
    base_path = f"{Path(input_files[0]).parent}/processed"
    coarse_infer(
        model_path="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device="cuda",
        batch_size=1,
        schedule="linear",
        lr=0.01,
        niter=100,  # 300
        n_views=len(input_files),
        img_base_path=base_path,
        focal_avg=True,
    )
    progress(0.4, desc="Camera parameters estimated! Starting streaming...")

    #####
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000],
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--scene", type=str, default="test")
    parser.add_argument("--n_views", type=int, default=len(input_files))
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true", default=True)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # overwrite some parameters
    args.source_path = base_path

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations

    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)
    gaussians.training_setup(opt)
    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + "pose", exist_ok=True)
    save_pose(scene.model_path + "pose" + "/pose_org.npy", gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    start = perf_counter()
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if args.optim_pose is False:
            gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # # Render
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(
                    scene.model_path + "pose" + f"/pose_{iteration}.npy",
                    gaussians.P,
                    train_cams_init,
                )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

        end = perf_counter()
        train_time = end - start

    # img: UInt8[np.ndarray, "h w 3"] = mmcv.imread(input_files[0], channel_order="rgb")

    # img: UInt8[np.ndarray, "... 3"] = mmcv.imrescale(img, 0.25)

    # blueprint: rrb.Blueprint = create_blueprint()
    # rr.send_blueprint(blueprint)

    # rr.set_time_sequence("iteration", 0)

    # rr.log("image/original", rr.Image(img).compress(jpeg_quality=80))
    # yield stream.read()

    # blur: UInt8[np.ndarray, "... 3"] = img

    # for i in range(100):
    #     rr.set_time_sequence("iteration", i)

    #     time.sleep(0.1)
    #     blur = cv2.GaussianBlur(blur, (3, 3), 0)

    #     rr.log("image/blurred", rr.Image(blur).compress(jpeg_quality=80))
    #     yield stream.read()


if IN_SPACES:
    estimate_pose_fn = spaces.GPU(estimate_pose_fn)


def preview_input(input_files: list[str]) -> list[UInt8[ndarray, "h w 3"]]:
    """
    Processes a list of image file paths, resizes them if necessary, and saves the processed images to a new folder.

    Args:
        input_files (list[str]): List of image file paths to be processed.

    Returns:
        list[UInt8[ndarray, "h w 3"]]: List of processed images as NumPy arrays with shape (height, width, 3).

    Notes:
        - If the input file is in HEIC or HEIF format, it is converted to RGB using PIL.
        - Images are resized such that their maximum dimension does not exceed 720 pixels.
        - Processed images are saved in a new folder named 'processed' within the parent directory of the input files.
    """
    if input_files is None:
        return None
    img_list: list[UInt8[ndarray, "h w 3"]] = []
    for img_file in input_files:
        if img_file.lower().endswith((".heic", ".heif")):
            img = PIL.Image.open(img_file).convert("RGB")
            img: UInt8[ndarray, "h w 3"] = np.array(img)
        else:
            img: UInt8[ndarray, "h w 3"] = mmcv.imread(img_file, channel_order="rgb")

        # reshape image so that its max dimension is 720
        max_dim = max(img.shape[:2])
        if max_dim > 720:
            img = mmcv.imrescale(img=img, scale=(720 / max_dim))
        img_list.append(img)

    # save images to parent folder in a new folder called processed
    input_file_parent: Path = Path(input_files[0]).parent
    processed_folder: Path = input_file_parent / "processed"
    processed_folder.mkdir(exist_ok=True)
    for i, img in enumerate(img_list):
        print(f"Saving processed image {i} to {processed_folder}")
        mmcv.imwrite(
            img=img,
            file_path=f"{processed_folder}/images/img_{i}.jpg",
        )

    return img_list


with gr.Blocks() as multi_img_block:
    with gr.Row():
        input_imgs = gr.File(file_count="multiple")
        gallery_imgs = gr.Gallery(rows=1)
        img = gr.Image(interactive=True, label="Image", visible=False)
    with gr.Row():
        with gr.Column():
            pose_btn = gr.Button("Stream Pose Estimation")
            stop_btn = gr.Button("Stop Streaming")
            test_btn = gr.Button("Test")

    with gr.Row():
        viewer = Rerun(
            streaming=True,
        )

    click_event = pose_btn.click(
        estimate_pose_fn, inputs=[input_imgs], outputs=[viewer]
    )
    stop_btn.click(fn=None, inputs=[], outputs=[], cancels=[click_event])

    input_imgs.change(fn=preview_input, inputs=[input_imgs], outputs=[gallery_imgs])
