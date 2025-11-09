from __future__ import annotations
import functools
import numpy as np
import open3d as o3d


def make_pcd(
    xyz: np.ndarray,
    rgb: np.ndarray,
    return_mapping: bool = False,
) -> o3d.geometry.PointCloud:
    """
    Make a point cloud from xyz and rgb.
    Args:   
        xyz: (N, 3) - Point cloud coordinates
        rgb: (N, 3) - Point cloud colors
    Returns:
        pcd: (N, 3) - Point cloud
        mapping: (N, 1) - Mapping from point cloud to image pixels
    """
    points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3).astype(np.float64) / 255)
    pcd = o3d.geometry.PointCloud(points)
    pcd.colors = colors
    if return_mapping:
        return pcd, mapping
    mapping = np.arange(len(xyz))

def merge_pcds(
    voxel_size: float,
    n_points: int,
    pcds: list[o3d.geometry.PointCloud],
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox,
) -> o3d.geometry.PointCloud:
    merged_pcd = functools.reduce(lambda a, b: a + b, pcds, o3d.geometry.PointCloud())
    merged_pcd = merged_pcd.crop(ws_aabb)
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
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
    return downsampled_pcd

def depth2pcd(
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> o3d.geometry.PointCloud:
    depth = depth.reshape(-1, 1)
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    points = np.concatenate([x, y, depth], axis=-1)
    pcd = make_pcd(points, np.ones_like(depth))
    return pcd