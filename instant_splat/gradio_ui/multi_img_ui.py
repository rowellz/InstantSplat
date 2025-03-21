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

from typing import List, Union, Optional
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
import random
from math import tau

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
    rr.log(
        f"{parent_log_path}/gaussian_points",
        rr.Points3D(
            positions=initial_gaussians.numpy(force=True),
            colors=colors_rgb.numpy(force=True),
        ),
    )


def ray_sphere_intersection(
    ray_origin: ndarray,
    ray_dir: ndarray,
    sphere_center: ndarray,
    sphere_radius: float
) -> Optional[float]:
    oc = ray_origin - sphere_center
    a = float(np.dot(ray_dir, ray_dir))
    b = 2.0 * float(np.dot(oc, ray_dir))
    c = float(np.dot(oc, oc)) - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return None
    
    sqrt_disc = np.sqrt(discriminant)
    t0 = (-b - sqrt_disc) / (2 * a)
    t1 = (-b + sqrt_disc) / (2 * a)
    
    if t0 > 0:
        return t0
    if t1 > 0:
        return t1
    return None

def find_closest_intersection(
    ray_origin: ndarray,
    ray_dir: ndarray,
    gaussian_centers: ndarray,
    gaussian_radii: ndarray
) -> Optional[ndarray]:
    closest_t = float('inf')
    closest_point = None
    
    for i in range(len(gaussian_centers)):
        t = ray_sphere_intersection(
            ray_origin,
            ray_dir,
            gaussian_centers[i],
            float(gaussian_radii[i])
        )
        if t is not None and t < closest_t:
            closest_t = t
            closest_point = ray_origin + t * ray_dir
    
    return closest_point

def log_cameras(
    parent_log_path: Path,  # Fixed the parameter name from Marlparent_log_path
    cameras: list[Camera],
    gaussians: GaussianModel,
    pipe: PipelineParams | GroupParams,
    bg: Float32[Tensor, "3"],
) -> None:
    """
    Logs camera poses, pinhole projections, and rendered images in Rerun.

    Args:
        parent_log_path: Base path for logging in Rerun.
        cameras: List of Camera objects to log.
        gaussians: GaussianModel instance for rendering.
        pipe: Pipeline parameters for rendering.
        bg: Background color tensor (3D float tensor).
    """
    for i, cam in enumerate(cameras):
        # Get camera pose and convert to camera-to-world transform
        quat_t: Float32[Tensor, "7"] = gaussians.get_RT(cam.uid)
        w2c: Float32[Tensor, "4 4"] = get_camera_from_tensor(quat_t)  # World-to-camera
        cam_to_world: ndarray = np.linalg.inv(w2c.numpy(force=True))  # Camera-to-world
        cam_log_path = parent_log_path / f"camera_{cam.uid}"

        # Camera intrinsics
        fx = cam.image_width / (2 * np.tan(cam.FoVx / 2))
        fy = cam.image_height / (2 * np.tan(cam.FoVy / 2))
        principal_point = (cam.image_width / 2, cam.image_height / 2)

        # Log camera transform in world space
        rr.log(
            str(cam_log_path),
            rr.Transform3D(
                translation=cam_to_world[:3, 3],  # Camera position in world space
                mat3x3=cam_to_world[:3, :3],      # Camera orientation
                from_parent=False,                # World is parent frame
            ),
        )

        # Log pinhole camera model
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                resolution=[cam.image_width, cam.image_height],
                focal_length=[fx, fy],
                principal_point=principal_point,
                camera_xyz=rr.ViewCoordinates.RDF,  # Rerun's default: Right, Down, Forward
            ),
        )

        # Render and log predicted image
        render_pkg = render(cam, gaussians, pipe, bg, camera_pose=quat_t)
        img_pred_viz: Float32[Tensor, "3 h w"] = render_pkg["render"] * 255
        img_pred_viz: UInt8[ndarray, "h w 3"] = (
            img_pred_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)
        )
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(img_pred_viz).compress(jpeg_quality=90),
        )

        # Log ground truth image
        img_gt_viz: Float32[Tensor, "3 h w"] = cam.original_image * 255
        img_gt_viz: UInt8[ndarray, "h w 3"] = (
            img_gt_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)
        )
        rr.log(
            f"{parent_log_path}/gt_image_{cam.uid}",
            rr.Image(img_gt_viz).compress(jpeg_quality=90),
        )

def log_camera_lines(
    parent_log_path: Path,
    cameras: list[Camera],
    gaussians: GaussianModel
) -> None:
    """
    Logs 3D lines from camera positions to their intersections with Gaussian splats in Rerun.

    Args:
        parent_log_path: Base path for logging in Rerun.
        cameras: List of Camera objects to log lines for.
        gaussians: GaussianModel instance containing camera poses and Gaussian data.
    """
    # Clear previous lines
    rr.log(f"{parent_log_path}/arrows", rr.Clear(recursive=True))

    # Get Gaussian data for intersection calculations
    gaussian_centers = gaussians.get_xyz.numpy(force=True)  # [N, 3]
    gaussian_radii = gaussians.get_scaling.numpy(force=True)  # [N, 3]
    # Use maximum scaling dimension as sphere radius for simplicity
    gaussian_radii = np.max(gaussian_radii, axis=1)  # [N]

    line_strips = []
    for i, cam in enumerate(cameras):
        # Get camera pose and convert to camera-to-world transform
        quat_t: Float32[Tensor, "7"] = gaussians.get_RT(cam.uid)
        w2c: Float32[Tensor, "4 4"] = get_camera_from_tensor(quat_t)
        cam_to_world: ndarray = np.linalg.inv(w2c.numpy(force=True))

        # Camera intrinsics
        fx = cam.image_width / (2 * np.tan(cam.FoVx / 2))
        fy = cam.image_height / (2 * np.tan(cam.FoVy / 2))
        principal_point = (cam.image_width / 2, cam.image_height / 2)

        # Calculate ray direction towards image center in camera space
        center_2d = np.array([cam.image_width / 2, cam.image_height / 2, 1.0])
        intrinsics = np.array([
            [fx, 0, principal_point[0]],
            [0, fy, principal_point[1]],
            [0, 0, 1]
        ])
        inv_intrinsics = np.linalg.inv(intrinsics)
        center_3d_cam = inv_intrinsics @ center_2d
        center_3d_cam = center_3d_cam / center_3d_cam[2]  # Normalize
        center_3d_world = cam_to_world @ np.append(center_3d_cam, 1.0)
        
        # Compute ray parameters
        cam_origin = cam_to_world[:3, 3]
        ray_dir = center_3d_world[:3] - cam_origin
        ray_dir = ray_dir / np.linalg.norm(ray_dir)  # Normalize direction
        
        # Find intersection with Gaussians
        intersection_point = find_closest_intersection(
            cam_origin,
            ray_dir,
            gaussian_centers,
            gaussian_radii
        )
        
        # Set line endpoint to intersection if found, otherwise use original endpoint
        if intersection_point is not None:
            line_end = intersection_point
        else:
            line_end = center_3d_world[:3]  # Fallback to original endpoint

        # Add line segment
        line_strips.append([cam_origin, line_end])
        line_strips.append([line_end, cam_origin, ])


        # Debug logging
        rr.log(
            f"{parent_log_path}/camera_{cam.uid}/debug",
            rr.TextDocument(
                f"Origin: {cam_origin.tolist()}, End: {line_end.tolist()}",
                media_type=rr.MediaType.TEXT,
            ),
            timeless=False,
        )

    # Log all line segments
    rr.log(
        f"{parent_log_path}/arrows",
        rr.LineStrips3D(line_strips),
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
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_3/pinhole/",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/gt_image_3",
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_4/pinhole/",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/gt_image_4",
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_5/pinhole/",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/gt_image_5",
                    ),
                ),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def prepare_output_and_logger(args) -> None:
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(
    iteration,
    l1_loss,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
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
    yield from _train_splat_fn(input_files, processed_folder, dust3r_conf)


@rr.thread_local_stream("train splat stream")
def _train_splat_fn(
    input_files: list[str],
    processed_folder: Path,
    dust3r_conf: int | float,
    progress=gr.Progress(),
) -> Generator[tuple[bytes, Path | None, Path | None], None, None]:
    print(zero.device)
    stream: rr.BinaryStream = rr.binary_stream()

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

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[300, 500, 800, 1000],
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
    args.model_path = f"{base_path}/output/"
    args.source_path = base_path
    args.iterations = 1

    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)

    print("Optimizing " + args.model_path)

    dataset: GroupParams = lp.extract(args)
    opt: GroupParams = op.extract(args)
    pipe: GroupParams = pp.extract(args)
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

    iteration = 1
    iter_start.record()
    rr.set_time_sequence("iteration", iteration)

    gaussians.update_learning_rate(iteration)

    if args.optim_pose is False:
        gaussians.P.requires_grad_(False)

    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    if not viewpoint_stack:
        viewpoint_stack: list[Camera] = scene.getTrainCameras().copy()

    viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    pose: Float32[Tensor, "7"] = gaussians.get_RT(viewpoint_cam.uid)

    if (iteration - 1) == args.debug_from:
        pipe.debug = True

    bg: Float32[Tensor, "3"] = (
        torch.rand((3), device="cuda") if opt.random_background else background
    )

    render_pkg: dict[str, Any] = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
    image: Float32[Tensor, "c h w"] = render_pkg["render"]
    gt_image: Float32[Tensor, "c h w"] = viewpoint_cam.original_image.cuda()

    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward()

    iter_end.record()

    with torch.no_grad():
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        log_cameras(parent_log_path, train_cams_init, gaussians, pipe, background)
        log_camera_lines(parent_log_path, train_cams_init, gaussians)  # New call to log lines
        rr.log(f"{parent_log_path}/loss_plot", rr.Scalar(ema_loss_for_log))
        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(1)
        progress_bar.close()

        training_report(iteration, l1_loss, testing_iterations, scene, render, (pipe, background))
        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)
            save_pose(scene.model_path + "pose" + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

        log_3d_splats(parent_log_path, gaussians)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration in checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    end = perf_counter()
    train_time: float = end - start
    final_ply_path = Path(f"{scene.model_path}/point_cloud/iteration_{opt.iterations}/point_cloud.ply")
    if final_ply_path.exists():
        yield stream.read(), str(final_ply_path), str(final_ply_path)
    else:
        yield stream.read(), None, None

if IN_SPACES:
    train_splat_fn = spaces.GPU(train_splat_fn, duration=90)


def preview_input(input_files: list[str]) -> tuple[list[UInt8[ndarray, "h w 3"]], Path]:
    if input_files is None:
        return None
    img_list: list[UInt8[ndarray, "h w 3"]] = []
    for img_file in input_files:
        if img_file.lower().endswith((".heic", ".heif")):
            img = PIL.Image.open(img_file).convert("RGB")
            img: UInt8[ndarray, "h w 3"] = np.array(img)
        else:
            img: UInt8[ndarray, "h w 3"] = mmcv.imread(img_file, channel_order="rgb")

        max_dim = max(img.shape[:2])
        if max_dim > 720:
            img = mmcv.imrescale(img=img, scale=(720 / max_dim))
        img_list.append(img)

    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    processed_folder = temp_dir_path / "processed"
    processed_folder.mkdir(exist_ok=True)

    for i, img in enumerate(img_list):
        print(f"Saving processed image {i} to {processed_folder}")
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