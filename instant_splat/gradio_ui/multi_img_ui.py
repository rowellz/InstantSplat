try:
    import spaces  # type: ignore # noqa: F401

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

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
import tempfile
from pathlib import Path
from jaxtyping import UInt8
from typing import Generator

from instant_splat.coarse_init_infer import coarse_infer
import os
from random import randint
from instant_splat.scene.cameras import Camera
from instant_splat.utils.loss_utils import l1_loss, ssim
from instant_splat.gaussian_renderer import render
from instant_splat.utils.sh_utils import SH2RGB
from instant_splat.scene import Scene, GaussianModel
from tqdm import tqdm
from instant_splat.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from instant_splat.arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    GroupParams,
)
from instant_splat.utils.pose_utils import get_camera_from_tensor
from torch import Tensor
from jaxtyping import Float32
from typing import Any

from time import perf_counter


zero = torch.Tensor([0]).cuda()
print(zero.device)  # <-- 'cpu' 🤔


def save_pose(path: str, quat_pose, train_cams, llffhold=2):
    output_poses = []
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    colmap_poses = []
    for i in range(len(index_colmap)):
        ind = index_colmap.index(i + 1)
        bb = output_poses[ind]
        bb = bb  # .inverse()
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def log_3d_splats(parent_log_path: Path, gaussians: GaussianModel) -> None:
    initial_gaussians: Float32[Tensor, "num_gaussians 3"] = gaussians.get_xyz
    colors_rgb: Float32[Tensor, "num_gaussians 3"] = SH2RGB(gaussians.get_features)[
        :, 0, :
    ]
    # get only the first 10 gaussians
    rr.log(
        f"{parent_log_path}/gaussian_points",
        rr.Points3D(
            positions=initial_gaussians.numpy(force=True),
            colors=colors_rgb.numpy(force=True),
        ),
    )


def log_cameras(
    parent_log_path: Path,
    cameras: list[Camera],
    gaussians: GaussianModel,
    pipe: PipelineParams | GroupParams,
    bg: Float32[Tensor, "3"],
) -> None:
    cam: Camera
    for idx, cam in enumerate(cameras):
        quat_t: Float32[Tensor, "7"] = gaussians.get_RT(cam.uid)
        w2c: Float32[Tensor, "4 4"] = get_camera_from_tensor(quat_t)
        cam_T_world: Float32[np.ndarray, "4 4"] = w2c.numpy(force=True)
        cam_log_path: Path = parent_log_path / f"camera_{cam.uid}"
        FoVx = cam.FoVx
        FoVy = cam.FoVy
        # convert to principal point and focal length
        fx = cam.image_width / (2 * np.tan(FoVx / 2))
        fy = cam.image_height / (2 * np.tan(FoVy / 2))
        principal_point = (cam.image_width / 2, cam.image_height / 2)

        img_gt_viz: Float32[Tensor, "3 h w"] = cam.original_image * 255
        img_gt_viz: UInt8[np.ndarray, "h w 3"] = (
            img_gt_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)
        )

        render_pkg: dict[str, Any] = render(
            cam, gaussians, pipe, bg, camera_pose=quat_t
        )
        img_pred_viz: Float32[Tensor, "3 h w"] = render_pkg["render"] * 255
        img_pred_viz: UInt8[np.ndarray, "h w 3"] = (
            img_pred_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)
        )

        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(
                translation=cam_T_world[:3, 3],
                mat3x3=cam_T_world[:3, :3],
                from_parent=True,
                axis_length=0.01,
            ),
        )
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                width=cam.image_width,
                height=cam.image_height,
                focal_length=(fx, fy),
                principal_point=principal_point,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.01,
            ),
        )
        # only log the first three cameras images for efficiency
        if idx > 2:
            continue
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(img_pred_viz).compress(jpeg_quality=90),
        )
        # log outside of camera to avoid cluttering the view
        rr.log(
            f"{parent_log_path}/gt_image_{cam.uid}",
            rr.Image(img_gt_viz).compress(jpeg_quality=90),
        )


def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=f"{parent_log_path}",
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin=f"{parent_log_path}/loss_plot",
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_0/pinhole/",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/gt_image_0",
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_1/pinhole/",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/gt_image_1",
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_2/pinhole/",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/gt_image_2",
                    ),
                ),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def prepare_output_and_logger(args) -> None:
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    return tb_writer


def training_report(
    iteration,
    l1_loss,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(len(scene.getTrainCameras()))
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    if config["name"] == "train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(
                        renderFunc(
                            viewpoint, scene.gaussians, *renderArgs, camera_pose=pose
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}"
                )
        torch.cuda.empty_cache()


def train_splat_fn(input_files, processed_folder, dust3r_conf):
    """
    This is required because of
    https://github.com/beartype/beartype/issues/423
    """
    yield from _train_splat_fn(input_files, processed_folder, dust3r_conf)


@rr.thread_local_stream("train splat stream")
def _train_splat_fn(
    input_files: list[str],
    processed_folder: Path,
    dust3r_conf: int | float,
    progress=gr.Progress(),
) -> Generator[tuple[bytes, Path | None, Path | None], None, None]:
    print(zero.device)
    # beartype causes gradio to break, so type hint after the fact, intead of on function definition
    stream: rr.BinaryStream = rr.binary_stream()

    ##################
    # Estimate Poses #
    ##################

    if input_files is None or len(input_files) < 3:
        raise gr.Error("Must provide 3 or more images.")
    if len(input_files) > 5 and IN_SPACES:
        gr.Warning(
            "More than 5 images will result in longer processing time, and potentially gpu time outs!"
        )

    progress(0.05, desc="Estimating Camera parameters with DUSt3R... please wait")
    if not processed_folder.exists():
        raise gr.Error(
            message="Processed folder not found. Please make sure to upload images"
        )
    base_path: str = f"{processed_folder}"
    coarse_infer(
        model_path="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device="cuda",
        batch_size=1,
        schedule="linear",
        lr=0.01,
        niter=300,
        n_views=len(input_files),
        img_base_path=base_path,
        focal_avg=True,
        confidence=float(dust3r_conf),
    )
    progress(0.4, desc="Camera parameters estimated! Starting streaming...")

    ##################
    # Splat Training #
    ##################

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            300,
            500,
            800,
            1000,
        ],
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    # Values usually set by the user
    args.model_path = f"{base_path}/output/"
    args.source_path = base_path
    args.iterations = 300

    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)

    print("Optimizing " + args.model_path)

    dataset: GroupParams = lp.extract(args)
    opt: GroupParams = op.extract(args)
    pipe: GroupParams = pp.extract(args)
    ###
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)
    gaussians.training_setup(opt)

    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + "pose", exist_ok=True)
    save_pose(scene.model_path + "pose" + "/pose_org.npy", gaussians.P, train_cams_init)
    bg_color: list[int] = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background: Float32[Tensor, "3 "] = torch.tensor(
        bg_color, dtype=torch.float32, device="cuda"
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    start = perf_counter()
    # Setup rerun logging
    parent_log_path = Path("world")
    blueprint: rrb.Blueprint = create_blueprint(parent_log_path)
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        f"{parent_log_path}/loss_plot",
        rr.SeriesLine(color=[255, 0, 0], name="Loss", width=2),
        static=True,
    )

    yield stream.read(), None, None

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        rr.set_time_sequence("iteration", iteration)

        gaussians.update_learning_rate(iteration)

        if args.optim_pose is False:
            gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack: list[Camera] = scene.getTrainCameras().copy()

        viewpoint_cam: Camera = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1)
        )
        pose: Float32[Tensor, "7"] = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        bg: Float32[Tensor, "3"] = (
            torch.rand((3), device="cuda") if opt.random_background else background
        )

        render_pkg: dict[str, Any] = render(
            viewpoint_cam, gaussians, pipe, bg, camera_pose=pose
        )
        image: Float32[Tensor, "c h w"] = render_pkg["render"]
        # Loss
        gt_image: Float32[Tensor, "c h w"] = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0 or iteration == 1:
                log_cameras(
                    parent_log_path, train_cams_init, gaussians, pipe, background
                )
                rr.log(f"{parent_log_path}/loss_plot", rr.Scalar(ema_loss_for_log))
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                iteration,
                l1_loss,
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                save_pose(
                    scene.model_path + "pose" + f"/pose_{iteration}.npy",
                    gaussians.P,
                    train_cams_init,
                )

            if iteration % 100 == 0 or iteration == 1:
                log_3d_splats(parent_log_path, gaussians)

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
        train_time: float = end - start
        final_ply_path = Path(
            f"{scene.model_path}/point_cloud/iteration_{opt.iterations}/point_cloud.ply"
        )
        if final_ply_path.exists():
            yield stream.read(), str(final_ply_path), str(final_ply_path)
        else:
            yield stream.read(), None, None


if IN_SPACES:
    train_splat_fn = spaces.GPU(train_splat_fn, duration=90)


def preview_input(input_files: list[str]) -> tuple[list[UInt8[ndarray, "h w 3"]], Path]:
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

    # create a temporary directory to store the processed images
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    processed_folder = temp_dir_path / "processed"
    processed_folder.mkdir(exist_ok=True)

    for i, img in enumerate(img_list):
        print(f"Saving processed image {i} to {processed_folder}")
        # needs to be in bgr as mmcv use cv2 to write image
        mmcv.imwrite(
            img=mmcv.rgb2bgr(img),
            file_path=f"{processed_folder}/images/img_{i}.jpg",
        )

    return img_list, processed_folder


def change_tab_to_output() -> gr.Tabs:
    return gr.Tabs(selected=1)


with gr.Blocks() as multi_img_block:
    with gr.Row():
        input_imgs = gr.File(file_count="multiple")
        processed_folder = gr.State(value=Path(""))
        with gr.Tabs() as tabs:
            with gr.TabItem(label="Gallery", id=0):
                gallery_imgs = gr.Gallery()
            with gr.TabItem(label="Outputs", id=1):
                splat_3d = gr.Model3D()
                splat_output = gr.File(label="Output Splat")
    with gr.Row():
        splat_btn = gr.Button("Train Splat")
        stop_splat_btn = gr.Button("Stop Training")
    with gr.Row():
        with gr.Accordion(label="Advanced Options", open=False):
            dust3r_conf = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=2.5,
                step=0.5,
                label="DUSt3R Confidence Threshold (Higher means more confident)",
            )

    with gr.Row():
        viewer = Rerun(
            streaming=True,
        )

    splat_event = splat_btn.click(
        fn=change_tab_to_output,
        inputs=None,
        outputs=tabs,
    ).then(
        fn=train_splat_fn,
        inputs=[input_imgs, processed_folder, dust3r_conf],
        outputs=[viewer, splat_output, splat_3d],
    )
    stop_splat_btn.click(fn=None, inputs=[], outputs=[], cancels=[splat_event])

    # events
    input_imgs.change(
        fn=preview_input, inputs=[input_imgs], outputs=[gallery_imgs, processed_folder]
    )

    car_example_path = Path("data/custom/car_landscape/4_views")
    car_image_paths: Generator[Path, None, None] = car_example_path.glob("images/*")
    car_image_paths: list[str] = [str(path) for path in car_image_paths]
    guitars_example_path = Path("data/custom/guitars/5_views")
    guitar_image_paths = guitars_example_path.glob("images/*")
    guitar_image_paths = [str(path) for path in guitar_image_paths]
    headphones_example_path = Path("data/custom/headphones/4_views")
    headphones_image_paths = headphones_example_path.glob("images/*")
    headphones_image_paths = [str(path) for path in headphones_image_paths]
    gr.Examples(
        examples=[[guitar_image_paths], [car_image_paths], [headphones_image_paths]],
        fn=train_splat_fn,
        inputs=[input_imgs, processed_folder, dust3r_conf],
        outputs=[viewer, splat_output],
        cache_examples=False,
    )
