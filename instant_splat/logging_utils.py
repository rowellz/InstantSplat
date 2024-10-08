from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32
from torch import Tensor

def log_3d_splats(parent_log_path: Path, gaussians: GaussianModel) -> None:
    initial_gaussians: Float32[Tensor, "num_gaussians 3"] = gaussians.get_xyz
    colors: Float32[Tensor, "num_gaussians 3"] = SH2RGB(gaussians.get_features)[:, 0, :]

    # get only the first 10 gaussians
    rr.log(
        f"{parent_log_path}/gaussian_points",
        rr.Points3D(
            positions=initial_gaussians.numpy(force=True),
            colors=colors.numpy(force=True),
        ),
    )
    # ic(scales.shape, scales.dtype)
    # ic(rotations.shape, rotations.dtype)
    # scales = gaussians.get_scaling
    # rotations = gaussians.get_rotation
    # rr.log(
    #     f"{parent_log_path}/gaussian_ellipsoids",
    #     rr.Ellipsoids3D(
    #         centers=initial_gaussians.numpy(force=True),
    #         quaternions=rotations.numpy(force=True),
    #         half_sizes=scales.numpy(force=True),
    #         colors=colors.numpy(force=True),
    #         fill_mode=3,
    #     ),
    # )


def log_cameras(
    parent_log_path: Path,
    cameras: list[Camera],
    gaussians: GaussianModel,
    pipe: PipelineParams,
    bg: Float32[Tensor, "3"],
) -> None:
    cam: Camera
    for idx, cam in enumerate(cameras):
        quat_t: Float32[Tensor, "7"] = gaussians.get_RT(cam.uid)
        w2c: Float32[Tensor, "4 4"] = get_camera_from_tensor(quat_t)
        cam_T_world: Float32[Tensor, "3 4"] = w2c.numpy(force=True)
        cam_log_path: Path = parent_log_path / f"camera_{cam.uid}"
        FoVx = cam.FoVx
        FoVy = cam.FoVy
        # convert to principal point and focal length
        fx = cam.image_width / (2 * np.tan(FoVx / 2))
        fy = cam.image_height / (2 * np.tan(FoVy / 2))
        principal_point = (cam.image_width / 2, cam.image_height / 2)

        img_gt_viz: Float32[Tensor, "3 h w"] = cam.original_image * 255
        img_gt_viz: Float32[np.ndarray, "h w 3"] = (
            img_gt_viz.permute(1, 2, 0).numpy(force=True).astype(np.uint8)
        )

        render_pkg: dict[str, Any] = render(
            cam, gaussians, pipe, bg, camera_pose=quat_t
        )
        img_pred_viz: Float32[Tensor, "3 h w"] = render_pkg["render"] * 255
        img_pred_viz: Float32[np.ndarray, "h w 3"] = (
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
        rr.log(f"{cam_log_path}/pinhole/image", rr.Image(img_pred_viz))
        # log outside of camera to avoid cluttering the view
        rr.log(f"{parent_log_path}/gt_image_{cam.uid}", rr.Image(img_gt_viz))


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
        )
    )
    return blueprint