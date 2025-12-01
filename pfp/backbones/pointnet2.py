import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import sample_farthest_points, ball_query



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


def old_farthest_point_sample(xyz, npoint):
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


def old_query_ball_point(radius, nsample, xyz, new_xyz):
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
    # valid_points = (group_idx != N).sum(dim=-1)  # [B, S]
    # print(f"\n=== Valid points per group (before padding) ===")
    # print(f"Shape: {valid_points.shape} (B={B}, S={S})")
    # print(f"Min: {valid_points.min().item()}, Max: {valid_points.max().item()}, Mean: {valid_points.float().mean().item():.2f}")
    # print(f"Groups with 0 valid points: {(valid_points == 0).sum().item()}")
    # print(f"Groups with <{nsample} valid points: {(valid_points < nsample).sum().item()}")
    # print(f"Groups with =={nsample} valid points: {(valid_points == nsample).sum().item()}")

    # --- Fallback to nearest point ---
    # nn_idx: shape [B, S, 1]
    _, nn_idx = torch.topk(sqrdists, 1, dim=-1, largest=False)
    nn_idx = nn_idx.repeat(1, 1, nsample)  # repeat to fill nsample

    mask = group_idx == N
    group_idx[mask] = nn_idx[mask]
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
    B, N, C = xyz.shape
    S = npoint
    new_xyz, fps_idx = sample_farthest_points(xyz, K=npoint)  # [B, npoint, C]
    _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True)  # [B, npoint, nsample]
    # grouped_xyz shape: [B, npoint, nsample, 3]
    # Normalize grouped_xyz relative to new_xyz (center at new_xyz)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, 3]
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
        points: (B, N, D) or None
    Returns:
        new_xyz: (B, 1, 3)
        new_points: (B, 1, 3+D) if points is not None, else (B, 1, 3)
     Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use mean as the centroid
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, 1, 1, C)
    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm   
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
        bn = True,
    ):
        super().__init__()
        self.npoints = npoints
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all
        self.bn = bn
        last_channel = in_channels
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor,
    ):
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, D) or None
        Returns:
            new_xyz: (B, S, 3)
            new_points: (B, S, D')
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
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D+C, nsample, npoints]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
            else:
                new_points = F.relu(conv(new_points)) # [B, C+D, nsample, npoints]
        

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2Backbone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_points: int = 4096,
        use_group_norm: bool = False,
        sa_configs: list[dict] = None,
    ):
        super().__init__()

        # Default SA configs
        if sa_configs is None:
            sa_configs = [
                {
                    "npoints": 512,
                    "radius": 0.06,
                    "nsample": 32,
                    "mlp": [64, 64, 128],
                    "in_channel": 3,
                },
                {
                    "npoints": 128,
                    "radius": 0.20,
                    "nsample": 64,
                    "mlp": [128, 128, 256],
                    "in_channel": 128 + 3,
                },
                {
                    "npoints": None,  # group all
                    "radius": None,
                    "nsample": None,
                    "mlp": [256, 512, 1024],
                    "in_channel": 256 + 3,
                },
            ]
        self.sa_layers = nn.ModuleList()

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
            nn.ReLU(),
            nn.Dropout(0.2),
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
        pcd = pcd.float().reshape(-1, *pcd.shape[2:])  # (B * T, N, C)
        if robot_state_obs is not None:
            robot_state_obs = robot_state_obs.float().reshape(
                -1, *robot_state_obs.shape[2:]
            )  # (B * T, 10)

        # Separate xyz and features
        xyz = pcd[..., :3]
        if pcd.shape[-1] > 3:
            points = pcd[..., 3:]
        else:
            points = None

        # Normalize xyz
        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        # Pass through SA layers
        for sa_layer in self.sa_layers:
            xyz, points = sa_layer(xyz, points)

        # Global feature: [B*T, 1, D] -> [B * T, D]
        # After SA layers, points should never be None (first layer outputs features even if input points is None)
        if points.shape[1] == 1:
            points = points.squeeze(1)
        else:
            points = torch.max(points, dim=1)[0]

        encoded_pcd = self.final_mlp(points)  # [B * T, embed_dim]

        # robot_state_obs should not be None based on function signature
        if robot_state_obs is None:
            raise ValueError("robot_state_obs cannot be None")
        nx = torch.cat([encoded_pcd, robot_state_obs], dim=1)

        if len(original_shape) == 4:  # [B, T, N, C]
            nx = nx.reshape(B, -1)  # [B, T * embed_dim + 10]

        return nx
