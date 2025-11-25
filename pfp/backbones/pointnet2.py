from types import DynamicClassAttribute
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    print(f"sample_and_group: xyz: {xyz.shape}")
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor):
    """
    Args:
        xyz: (B, N, 3)
        points: (B, N, D)
    Returns:
        new_xyz: (B, 1, 3)
        new_points: (B, 1, D)
     Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstractor(nn.Module):
    def __init__(
        self,
        npoints: int,
        radius: float,
        nsample: int,
        in_channels: int,
        mlp: list,
        group_all: bool,
    ):
        super().__init__()
        self.npoints = npoints
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all
        last_channel = in_channels
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor,
    ):
        """
        Args:
            xyz: (B, C, N)
            points: (B, D, N)
        Returns:
            new_xyz: (B, C, S)
            new_points: (B, D', S)
        """
        xyz = xyz.contiguous()
        if points is not None:
            points = points.contiguous()

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoints, self.radius, self.nsample, xyz, points
            )
        # new_xyz: sampled points position data, [B, npoints, C]
        # new_points: sampled points data, [B, npoints, nsample, D+C]
        new_points = new_points.permute(0, 3, 1, 2)  # [B, D+C, npoints, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 3)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2Backbone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_channels: int,
        n_points: int = 4096,
        use_group_norm: bool = False,
        sa_configs: list[dict] = None,
    ):
        super().__init__()
        assert input_channels in [3, 6], "Input channels must be 3 or 6"
        in_channels = 6 if input_channels == 6 else 3

        # Default SA configs
        if sa_configs is None:
            sa_configs = [
                {
                    "npoints": 512,
                    "radius": 0.2,
                    "nsample": 32,
                    "mlp": [64, 64, 128],
                    "in_channel": in_channels,
                },
                {
                    "npoints": 128,
                    "radius": 0.4,
                    "nsample": 64,
                    "mlp": [128, 128, 256],
                    "in_channel": 128+3,
                },
                {
                    "npoints": None,  # group all
                    "radius": None,
                    "nsample": None,
                    "mlp": [256, 512, 1024],
                    "in_channel": 256+3,
                },
            ]
        self.sa_layers = nn.ModuleList()
        in_channel = input_channels

        # Build SA layers
        for i, sa_config in enumerate(sa_configs):
            if sa_config["npoints"] is None:
                # last layer group all
                self.sa_layers.append(
                    PointNetSetAbstractor(
                        npoints=sa_config["npoints"],
                        radius=sa_config["radius"],
                        nsample=sa_config["nsample"],
                        in_channels=sa_config["in_channel"],
                        mlp=sa_config["mlp"],
                        group_all=True,
                    )
                )
            else:
                self.sa_layers.append(
                    PointNetSetAbstractor(
                        npoints=sa_config["npoints"],
                        radius=sa_config["radius"],
                        nsample=sa_config["nsample"],
                        in_channels=sa_config["in_channel"],
                        mlp=sa_config["mlp"],
                        group_all=False,
                    )
                )
            in_channel = sa_config["mlp"][-1]

        self.final_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, pcd: torch.Tensor, robot_state_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pcd: (B, T, N, C), where C = 3 or 6
            robot_state_obs: (B, T, 10)
        Returns:
            x: (B, T * embed_dim + 10)
        """
        B, T, _, _ = pcd.shape
        original_shape = pcd.shape

        # Flatten the batch and time dimensions
        if len(pcd.shape) == 4:
            pcd = pcd.float().reshape(-1, *pcd.shape[2:])  # (B * T, N, C)
            if robot_state_obs is not None:
                robot_state_obs = robot_state_obs.float().reshape(
                    -1, *robot_state_obs.shape[2:]
                )  # (B * T, 10)
        else:
            pcd = pcd.float()

        # Separate xyz and features
        xyz = pcd[..., :3]
        if pcd.shape[-1] > 3:
            points = pcd[..., 3:]
        else:
            points = None

        # Pass through SA layers
        for sa_layer in self.sa_layers:
            print(f"point2Backbone forward:{xyz.shape}")
            xyz, points = sa_layer(xyz, points)

        # Global feature: [B*T, 1, D] -> [B * T, D]
        if points.shape[1] == 1:
            points = points.squeeze(1)
        else:
            points = torch.max(points, dim=1)[0]

        encoded_pcd = self.final_mlp(points)  # [B * T, embed_dim]

        # Concatenate with robot state
        if robot_state_obs is not None:
            nx = torch.cat([encoded_pcd, robot_state_obs], dim=1)
        else:
            nx = encoded_pcd

        if len(original_shape) == 4:  # [B, T, N, C]
            nx = nx.reshape(B, -1)  # [B, T * embed_dim + 10]

        return nx
