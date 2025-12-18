import torch
from typing import List
import numpy as np
import rerun as rr


def project_points(points: torch.Tensor, cam_intrinsics: List[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # world space
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # camera space
    fx, fy, cx, cy = cam_intrinsics[0], cam_intrinsics[1], cam_intrinsics[2], cam_intrinsics[3]

    ux = x / z * fx + cx
    uy = y / z * fy + cy

    uv = torch.stack([ux, uy], dim=-1)

    return uv, ux, uy, z


def point_cloud_rasterization(points: torch.Tensor, points_colors: torch.Tensor, H: int, W: int, cam_intrinsics: List[float], near: float, far: float) -> torch.Tensor:

    uv, _, _, z_cam_coords = project_points(points, cam_intrinsics)

    mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z_cam_coords > near) & (z_cam_coords < far)
    uv = uv[mask]

    image = torch.zeros(H, W, 3)

    u = uv[:, 0].round().clamp(0, W - 1).long()
    v = uv[:, 1].round().clamp(0, H - 1).long()

    image[v, u] = points_colors[mask]

    return image


if __name__ == "__main__":
    # Initialize Rerun
    rr.init("point_cloud_rasterization", spawn=True)

    # Generate random point cloud
    points = torch.randn(100000, 3)
    z_offset = 10.0
    points[:, 2] += z_offset  # Push points in front of camera
    points_colors = torch.rand(points.shape[0], 3)

    # Camera parameters
    H = 1000
    W = 1000
    fx, fy, cx, cy = 1000, 1000, H / 2, W / 2
    cam_intrinsics = [fx, fy, cx, cy]
    near = 0.1
    far = 100.0

    # Log the 3D point cloud
    rr.log(
        "world/points",
        rr.Points3D(
            positions=points.numpy(),
            colors=points_colors.numpy(),
            radii=0.02,
        ),
    )

    # Log the camera (pinhole model)
    rr.log(
        "world/camera",
        rr.Pinhole(
            resolution=[W, H],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
        ),
    )

    # Rasterize the point cloud
    image = point_cloud_rasterization(points, points_colors, H, W, cam_intrinsics, near, far)
    print(f"Image shape: {image.shape}, sum: {image.sum():.2f}")

    # Log the rendered image under the camera
    rr.log(
        "world/camera/image",
        rr.Image(image.numpy()),
    )
