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
from typing import Generator, List, Union, Optional
from instant_splat.coarse_init_infer import coarse_infer
import os
import json
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
from time import perf_counter
import trimesh
import open3d as o3d

zero = torch.Tensor([0]).cuda()

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
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)
    dir = Path(path).parent

    # --- Save as readable txt (for debugging, not COLMAP native) ---
    with open(dir / 'output.txt', 'w') as f:
        for i, pose in enumerate(colmap_poses):
            f.write(f"# Pose {i}\n")
            np.savetxt(f, pose)
            f.write("\n")

    # --- Save COLMAP images.txt format ---
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        R = None
    images_txt = dir / 'images.txt'
    with open(images_txt, 'w') as f:
        for i, (pose, cam) in enumerate(zip(colmap_poses, train_cams)):
            # COLMAP expects world-to-camera: quaternion (w, x, y, z), translation (t_x, t_y, t_z)
            # pose: (4,4) or (3,4) matrix
            if pose.shape == (3, 4):
                pose4 = np.eye(4)
                pose4[:3, :4] = pose
                pose = pose4
            R_wc = pose[:3, :3]
            t_wc = pose[:3, 3]
            if R is not None:
                quat = R.from_matrix(R_wc).as_quat()  # (x, y, z, w)
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            else:
                # fallback: identity quaternion
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            # Use cam.colmap_id and cam.image_name if available
            image_id = getattr(cam, 'colmap_id', i+1)
            camera_id = getattr(cam, 'colmap_id', i+1)
            image_name = getattr(cam, 'image_name', f"img_{i}.jpg")
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t_wc[0]} {t_wc[1]} {t_wc[2]} {camera_id} {image_name}\n")

    # --- Save COLMAP cameras.txt format ---
    cameras_txt = dir / 'cameras.txt'
    with open(cameras_txt, 'w') as f:
        for i, cam in enumerate(train_cams):
            camera_id = getattr(cam, 'colmap_id', i+1)
            width = getattr(cam, 'image_width', 0)
            height = getattr(cam, 'image_height', 0)
            fx = getattr(cam, 'fx', width / 2)  # fallback
            fy = getattr(cam, 'fy', height / 2)
            cx = width / 2
            cy = height / 2
            # COLMAP PINHOLE: fx fy cx cy
            f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def log_3d_splats(parent_log_path: Path, gaussians: GaussianModel) -> None:
    initial_gaussians = gaussians.get_xyz
    colors_rgb = SH2RGB(gaussians.get_features)[:, 0, :]
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
    parent_log_path: Path,
    cameras: list[Camera],
    gaussians: GaussianModel,
    pipe: PipelineParams | GroupParams,
    bg: Float32[Tensor, "3"],
    custom_coords: Optional[List[List[tuple[float, float]]]] = None
) -> tuple[None, list[dict]]:
    arrow_origins: list[ndarray] = []
    arrow_vectors: list[ndarray] = []
    arrow_colors: list[ndarray] = []
    gaussian_centers = gaussians.get_xyz.numpy(force=True)
    gaussian_radii = gaussians.get_scaling.numpy(force=True)
    gaussian_radii = np.max(gaussian_radii, axis=1)
    intersection_data = []


    if custom_coords is None:
        custom_coords = [[(cam.image_width / 2, cam.image_height / 2)] for cam in cameras]

    for idx, cam in enumerate(cameras):
        print(f"SCALE {cam.image_height} {cam.image_width}")

        quat_t = gaussians.get_RT(cam.uid)
        w2c = get_camera_from_tensor(quat_t)
        cam_T_world = w2c.numpy(force=True)
        cam_log_path = parent_log_path / f"camera_{idx}"
        cam_to_world = np.linalg.inv(w2c.numpy(force=True))

        FoVx = cam.FoVx
        FoVy = cam.FoVy
        fx = cam.image_width / (2 * np.tan(FoVx / 2))
        fy = cam.image_height / (2 * np.tan(FoVy / 2))
        principal_point = (cam.image_width / 2, cam.image_height / 2)

        img_gt_viz = cam.original_image * 255
        img_gt_viz = img_gt_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)
        render_pkg = render(cam, gaussians, pipe, bg, camera_pose=quat_t)
        img_pred_viz = render_pkg["render"] * 255
        img_pred_viz = img_pred_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)

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

        intrinsics = np.array([
            [fx, 0, principal_point[0]],
            [0, fy, principal_point[1]],
            [0, 0, 1]
        ])
        inv_intrinsics = np.linalg.inv(intrinsics)

        cam_origin = cam_to_world[:3, 3]

        for x, y in custom_coords[idx]:
            coord_2d = np.array([x, y, 1.0])
            coord_3d_cam = inv_intrinsics @ coord_2d
            coord_3d_cam = coord_3d_cam / coord_3d_cam[2]
            coord_3d_world = cam_to_world @ np.append(coord_3d_cam, 1.0)

            ray_dir = coord_3d_world[:3] - cam_origin
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            closest_point = find_closest_intersection(
                cam_origin, ray_dir, gaussian_centers, gaussian_radii
            )

            if closest_point is not None:
                arrow_vector = closest_point - cam_origin
                intersection_data.append({
                    "camera_index": idx,
                    "x": float(x),
                    "y": float(y),
                    "xyz": closest_point.tolist()
                })
            else:
                arrow_vector = ray_dir * 0.1
                intersection_data.append({
                    "camera_index": idx,
                    "x": float(x),
                    "y": float(y),
                    "xyz": None
                })

            arrow_origins.append(cam_origin)
            arrow_vectors.append(arrow_vector)
            arrow_colors.append([0, 0, 255])

    rr.log(
        f"{parent_log_path}/arrows_to_centers",
        rr.Arrows3D(
            origins=np.array(arrow_origins),
            vectors=np.array(arrow_vectors),
            colors=np.array(arrow_colors, dtype=np.uint8),
        ),
    )
    
    return None, intersection_data

def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=f"{parent_log_path}"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin=f"{parent_log_path}/loss_plot"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera_0/pinhole/", contents="$origin/**"),
                    rrb.Spatial2DView(origin=f"{parent_log_path}/gt_image_0"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera_1/pinhole/", contents="$origin/**"),
                    rrb.Spatial2DView(origin=f"{parent_log_path}/gt_image_1"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera_2/pinhole/", contents="$origin/**"),
                    rrb.Spatial2DView(origin=f"{parent_log_path}/gt_image_2"),
                ),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )

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
            {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    pose = scene.gaussians.get_RT(viewpoint.uid) if config["name"] == "train" else scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
        torch.cuda.empty_cache()

def train_splat_fn(input_files, dust3r_conf, coords_df, cameras_txt=None, images_txt=None):
    import tempfile
    temp_dir = tempfile.mkdtemp()
    processed_folder = Path(temp_dir) / "processed"
    processed_folder.mkdir(exist_ok=True)
    images_dir = processed_folder / "images"
    images_dir.mkdir(exist_ok=True)

    if input_files is None or len(input_files) < 3:
        raise gr.Error("Must provide 3 or more images.")

    for i, img_file in enumerate(input_files):
        if img_file.lower().endswith((".heic", ".heif")):
            img = PIL.Image.open(img_file).convert("RGB")
            img = np.array(img)
        else:
            img = mmcv.imread(img_file, channel_order="rgb")
        max_dim = max(img.shape[:2])
        if max_dim > 720:
            img = mmcv.imrescale(img=img, scale=(720 / max_dim))
        mmcv.imwrite(img=mmcv.rgb2bgr(img), file_path=f"{images_dir}/img_{i}.jpg")

    # Save raw text to temp files if provided
    cameras_txt_path = None
    images_txt_path = None
    if cameras_txt and cameras_txt.strip():
        cameras_txt_path = processed_folder / "cameras.txt"
        with open(cameras_txt_path, "w") as f:
            f.write(cameras_txt)
    if images_txt and images_txt.strip():
        images_txt_path = processed_folder / "images.txt"
        with open(images_txt_path, "w") as f:
            f.write(images_txt)

    yield from _train_splat_fn(
        input_files, processed_folder, dust3r_conf, coords_df,
        str(cameras_txt_path) if cameras_txt_path else None,
        str(images_txt_path) if images_txt_path else None
    )

@rr.thread_local_stream("train splat stream")
def _train_splat_fn(
    input_files: list[str],
    processed_folder: Path,
    dust3r_conf: int | float,
    coords_df,
    cameras_txt: Optional[str] = None,
    images_txt: Optional[str] = None,
    progress=gr.Progress(),
) -> Generator[tuple[bytes, Path | None, Path | None, Path | None], None, None]:
    stream = rr.binary_stream()

    if len(input_files) > 5 and IN_SPACES:
        gr.Warning("More than 5 images will result in longer processing time, and potentially gpu time outs!")

    progress(0.05, desc="Estimating Camera parameters with DUSt3R... please wait")
    base_path = str(processed_folder)
    coarse_infer(
        model_path="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device="cuda",
        batch_size=1,
        schedule="linear",
        lr=0.01,
        niter=20,
        n_views=len(input_files),
        img_base_path=base_path,
        focal_avg=True,
        confidence=float(dust3r_conf),
        cameras_txt=cameras_txt,
        images_txt=images_txt,
    )
    progress(0.4, desc="Camera parameters estimated! Starting streaming...")

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20, 500, 800, 1000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose",  action="store_true")
    args = parser.parse_args(sys.argv[1:])

    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    args.model_path = f"{base_path}/output/"
    args.source_path = base_path
    args.iterations = 20
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    print("Optimizing " + args.model_path)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)
    gaussians.training_setup(opt)

    train_cams_init = scene.getTrainCameras().copy()
    train_cams_init.sort(key=lambda cam: int(cam.image_name.split('_')[-1].split('.')[0]))
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
    parent_log_path = Path("world")
    blueprint = create_blueprint(parent_log_path)
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    rr.log(f"{parent_log_path}/loss_plot", rr.SeriesLine(color=[255, 0, 0], name="Loss", width=2), static=True)

    num_images = len(input_files)
    custom_coords = [[] for _ in range(num_images)]
    if coords_df is not None:
        for row in coords_df.values:
            img_idx, x, y = map(float, row)
            if 0 <= img_idx < num_images:
                custom_coords[int(img_idx)].append((x, y))

    yield stream.read(), None, None, None, None

    intersection_data_final = []
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        rr.set_time_sequence("iteration", iteration)
        gaussians.update_learning_rate(iteration)

        if args.optim_pose is False:
            gaussians.P.requires_grad_(False)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        pose = gaussians.get_RT(viewpoint_cam.uid)

        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0 or iteration == 1:
                rr.log(f"{parent_log_path}/loss_plot", rr.Scalar(ema_loss_for_log))
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(iteration, l1_loss, testing_iterations, scene, render, (pipe, background))
            
            if iteration % 10 == 0 or iteration == 1:
                _, intersection_data = log_cameras(parent_log_path, train_cams_init, gaussians, pipe, background, custom_coords=custom_coords)
                if iteration == opt.iterations:
                    intersection_data_final = intersection_data

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                save_pose(scene.model_path + "pose" + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            if iteration % 100 == 0 or iteration == 1:
                log_3d_splats(parent_log_path, gaussians)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        end = perf_counter()
        final_ply_path = Path(f"{scene.model_path}/point_cloud/iteration_{opt.iterations}/point_cloud.ply")
        json_path = None
        npy_path = None
        zip_path = None
        if iteration == opt.iterations and intersection_data_final:
            json_path = Path(f"{scene.model_path}/intersection_data.json")
            with open(json_path, 'w') as f:
                json.dump(intersection_data_final, f, indent=2)
            # Return the .npy pose file as well
            npy_path = Path(scene.model_path + "pose" + f"/pose_{iteration}.npy")
            # Zip the processed/sparse directory
            import zipfile
            sparse_dir = processed_folder / "sparse"
            zip_path = processed_folder / "sparse.zip"
            if processed_folder.exists():
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(processed_folder):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(processed_folder)
                            zipf.write(file_path, arcname=str(arcname))

        if final_ply_path.exists():
            yield stream.read(), str(final_ply_path), str(final_ply_path), str(json_path) if json_path else None, str(zip_path) if zip_path and zip_path.exists() else None
        else:
            yield stream.read(), None, None, None, None

if IN_SPACES:
    train_splat_fn = spaces.GPU(train_splat_fn, duration=90)

def preview_input(input_files: list[str]) -> list[UInt8[ndarray, "h w 3"]]:
    if input_files is None:
        return None
    img_list: list[UInt8[ndarray, "h w 3"]] = []
    for img_file in input_files:
        if img_file.lower().endswith((".heic", ".heif")):
            img = PIL.Image.open(img_file).convert("RGB")
            img = np.array(img)
        else:
            img = mmcv.imread(img_file, channel_order="rgb")
        
        max_dim = max(img.shape[:2])
        if max_dim > 720:
            img = mmcv.imrescale(img=img, scale=(720 / max_dim))
        img_list.append(img)
    return img_list

def change_tab_to_output() -> gr.Tabs:
    return gr.Tabs(selected=1)

def obj_to_pointcloud(obj_path: str) -> o3d.geometry.PointCloud:
    mesh = trimesh.load(obj_path)
    points = mesh.vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def scale_and_position_pcd(
    obj_pcd: o3d.geometry.PointCloud,
    splat_pcd: o3d.geometry.PointCloud,
    scale_factor: float = 1.0,
    offset: float = 2.0
) -> o3d.geometry.PointCloud:
    obj_bbox = obj_pcd.get_axis_aligned_bounding_box()
    splat_bbox = splat_pcd.get_axis_aligned_bounding_box()
    
    obj_size = obj_bbox.get_max_extent()
    splat_size = splat_bbox.get_max_extent()
    if obj_size > 0:
        scale = (splat_size / obj_size) * scale_factor
        obj_pcd.scale(scale, center=obj_bbox.get_center())
    
    splat_center = splat_bbox.get_center()
    obj_center = obj_pcd.get_axis_aligned_bounding_box().get_center()
    
    translation = np.array([
        splat_bbox.get_max_bound()[0] + offset - obj_center[0],
        splat_center[1] - obj_center[1],
        splat_center[2] - obj_center[2]
    ])
    
    obj_pcd.translate(translation)
    return obj_pcd

def combine_point_clouds(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud
) -> o3d.geometry.PointCloud:
    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
    )
    return combined

with gr.Blocks() as multi_img_block:
    with gr.Row():
        input_imgs = gr.File(
            label="Input Images (3 or more required)",
            file_types=[".jpg", ".png", ".jpeg", ".heic", ".heif"],
            file_count="multiple"
        )
        cameras_txt = gr.Code(label="Optional cameras.txt (paste content)")
        images_txt = gr.Code(label="Optional images.txt (paste content)")
        with gr.Tabs() as tabs:
            with gr.TabItem(label="Gallery", id=0):
                gallery_imgs = gr.Gallery()
            with gr.TabItem(label="Outputs", id=1):
                splat_3d = gr.Model3D()
                splat_output = gr.File(label="Output Splat")
                json_output = gr.File(label="Intersection Data (JSON)")
                npy_output = gr.File(label="Pose File (.npy)")

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
            with gr.Column():
                gr.Markdown("### Custom Coordinates per Image")
                gr.Markdown("Enter x,y coordinates for each image (one set per row). Add rows as needed.")
                coords_df = gr.Dataframe(
                    headers=["Image Index", "X", "Y"],
                    datatype=["number", "number", "number"],
                    row_count=3,
                    col_count=(3, "fixed"),
                    interactive=True,
                    label="Coordinates (Image Index starts at 0)"
                )

    with gr.Row():
        viewer = Rerun(streaming=True)

    splat_event = splat_btn.click(
        fn=change_tab_to_output,
        inputs=None,
        outputs=tabs,
    ).then(
        fn=train_splat_fn,
        inputs=[input_imgs, dust3r_conf, coords_df, cameras_txt, images_txt],
        outputs=[viewer, splat_output, splat_3d, json_output, npy_output],
    )
    stop_splat_btn.click(fn=None, inputs=[], outputs=[], cancels=[splat_event])

    input_imgs.change(
        fn=preview_input,
        inputs=[input_imgs],
        outputs=[gallery_imgs]
    )

    car_example_path = Path("data/custom/car_landscape/4_views")
    car_image_paths = [str(path) for path in car_example_path.glob("images/*")]
    guitars_example_path = Path("data/custom/guitars/5_views")
    guitar_image_paths = [str(path) for path in guitars_example_path.glob("images/*")]
    headphones_example_path = Path("data/custom/headphones/4_views")
    headphones_image_paths = [str(path) for path in headphones_example_path.glob("images/*")]
    
    gr.Examples(
        examples=[
            [guitar_image_paths],
            [car_image_paths],
            [headphones_image_paths]
        ],
        fn=train_splat_fn,
        inputs=[input_imgs, dust3r_conf, coords_df],
        outputs=[viewer, splat_output, splat_3d, json_output, npy_output],
        cache_examples=False,
    )
