import os
import torch
import numpy as np
import time

from mini_dust3r.inference import inference
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.device import to_numpy
from mini_dust3r.image_pairs import make_pairs
from mini_dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from typing import Optional

from instant_splat.utils.dust3r_utils import (
    compute_global_alignment,
    load_images,
    storePly,
    save_colmap_cameras,
    save_colmap_images,
    parse_cam_and_img_txt
)


def coarse_infer(
    model_path: str,
    device,
    batch_size,
    schedule,
    lr,
    niter,
    n_views,
    img_base_path,
    focal_avg,
    confidence: float = 2.0,
    cameras_txt: Optional[str] = None,
    images_txt: Optional[str] = None
) -> None:
    img_folder_path = os.path.join(img_base_path, "images")
    os.makedirs(img_folder_path, exist_ok=True)

    assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    ##########################################################################################################################################################################################

    train_img_list = sorted(os.listdir(img_folder_path))
    assert (
        len(train_img_list) == n_views
    ), f"Number of images ({len(train_img_list)}) in the folder ({img_folder_path}) is not equal to {n_views}"

    images, ori_size = load_images(img_folder_path, size=512)
    print("ori_size", ori_size)

    start_time = time.time()
    ##########################################################################################################################################################################################
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    output_colmap_path = img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )
    if cameras_txt is not None and images_txt is not None:
        poses = parse_cam_and_img_txt(cameras_txt_path=cameras_txt, images_txt_path=images_txt)
        poses = poses[::-1]
        scene.preset_pose(poses, pose_msk=None)
        loss = compute_global_alignment(
            scene=scene,
            init="mst",
            niter=niter,
            schedule=schedule,
            lr=lr,
            focal_avg=focal_avg,
            use_colmap_poses=False,
            # cameras_txt_path=cameras_txt,
            # images_txt_path=images_txt,
        )

    else:
            
        loss = compute_global_alignment(
            scene=scene,
            init="mst",
            niter=niter,
            schedule=schedule,
            lr=lr,
            focal_avg=focal_avg,
            use_colmap_poses=False,
        )
    scene = scene.clean_pointcloud()

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(confidence)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())
    ##########################################################################################################################################################################################
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

    # save
    save_colmap_cameras(
        ori_size, intrinsics, os.path.join(output_colmap_path, "cameras.txt")
    )
    save_colmap_images(
        poses, os.path.join(output_colmap_path, "images.txt"), train_img_list
    )

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    if cameras_txt is not None and images_txt is not None:
        np.save(output_colmap_path + "/focal.npy", focals.detach().cpu().numpy())
    else:
        np.save(output_colmap_path + "/focal.npy", np.array(focals.cpu()))
