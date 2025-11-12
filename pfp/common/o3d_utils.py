from __future__ import annotations
import functools
import numpy as np
import open3d as o3d


def make_pcd(
    xyz: np.ndarray,
    rgb: np.ndarray = None,
) -> o3d.geometry.PointCloud:
    """
    Make a point cloud from xyz and rgb.
    Args:   
        xyz: (H, W, 3) - Point cloud coordinates
        rgb: (H, W, 3) - Point cloud colors
        return_mapping: If True, also return a dict mapping point cloud indices to 2D pixel coordinates
    Returns:
        pcd: Point cloud object with H*W points
        mapping: dict {pcd_index: (x, y)} - Dictionary mapping point cloud index to 2D pixel coordinates (only if return_mapping=True)
    """
    H, W = xyz.shape[0], xyz.shape[1]
    points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3).astype(np.float64) / 255) if rgb is not None else None
    pcd = o3d.geometry.PointCloud(points)
    if colors is not None:
        pcd.colors = colors
    return pcd


def merge_pcds(
    pcds: list[o3d.geometry.PointCloud],
    voxel_size: float = None,
    n_points: int = None,
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox = None,
) -> o3d.geometry.PointCloud:
    merged_pcd = functools.reduce(lambda a, b: a + b, pcds, o3d.geometry.PointCloud())
    if ws_aabb is not None:
        merged_pcd = merged_pcd.crop(ws_aabb)
    if n_points is not None:
        downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)

        if len(downsampled_pcd.points) > n_points:
            ratio = n_points / len(downsampled_pcd.points)
            downsampled_pcd = downsampled_pcd.random_down_sample(ratio)
        if len(downsampled_pcd.points) < n_points:
            # Append zeros to make the point cloud have the desired number of points
            num_missing_points = n_points - len(downsampled_pcd.points)
            zeros = np.zeros((num_missing_points, 3))
            zeros_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zeros))
            zeros_pcd.colors = o3d.utility.Vector3dVector(zeros)
            downsampled_pcd += zeros_pcd
        merged_pcd = downsampled_pcd
    return merged_pcd


