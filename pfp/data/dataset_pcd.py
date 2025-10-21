"""
Point cloud dataset for robot manipulation tasks.
Loads sequences of point clouds and robot states from replay buffers with temporal subsampling.
"""
from __future__ import annotations
import torch
import numpy as np
import pypose as pp
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.se3_utils import transform_th
from pfp import DATA_DIRS


def rand_range(low: float, high: float, size: tuple[int], device) -> torch.Tensor:
    return torch.rand(size, device=device) * (high - low) + low


def augment_pcd_data(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """
    Apply SE(3) data augmentation to point clouds and robot states.
    
    Input shapes:
        pcd: (B, T, N_points, 3 or 6) - point clouds with optional colors
        robot_state_obs: (B, T_obs, 10) - observed robot states
        robot_state_pred: (B, T_pred, 10) - predicted robot states
    
    Returns same shapes with random SE(3) transform applied and points shuffled.
    """
    pcd, robot_state_obs, robot_state_pred = batch
    BT_robot_obs = robot_state_obs.shape[:-1]
    BT_robot_pred = robot_state_pred.shape[:-1]

    # Apply random SE(3) transform: sigma=(sigma_transl, sigma_rot_rad)
    transform = pp.randn_SE3(sigma=(0.1, 0.2), device=pcd.device).matrix()

    # Transform point cloud XYZ coordinates
    pcd[..., :3] = transform_th(transform, pcd[..., :3])
    
    # Transform robot states (pos + rot6d representation)
    robot_obs_pseudoposes = robot_state_obs[..., :9].reshape(*BT_robot_obs, 3, 3)
    robot_pred_pseudoposes = robot_state_pred[..., :9].reshape(*BT_robot_pred, 3, 3)
    robot_obs_pseudoposes = transform_th(transform, robot_obs_pseudoposes)
    robot_pred_pseudoposes = transform_th(transform, robot_pred_pseudoposes)
    robot_state_obs[..., :9] = robot_obs_pseudoposes.reshape(*BT_robot_obs, 9)
    robot_state_pred[..., :9] = robot_pred_pseudoposes.reshape(*BT_robot_pred, 9)

    # Shuffle point order along point dimension
    idx = torch.randperm(pcd.shape[2])
    pcd = pcd[:, :, idx, :]
    return pcd, robot_state_obs, robot_state_pred


class RobotDatasetPcd(torch.utils.data.Dataset):
    """
    PyTorch dataset for point cloud-based robot manipulation sequences.
    
    Returns:
        pcd: (T_obs, N_points, 3 or 6) - Point clouds (XYZ or XYZRGB)
        robot_state_obs: (T_obs, 10) - Observed robot states [pos(3), rot6d(6), gripper(1)]
        robot_state_pred: (T_pred, 10) - Future robot states to predict
    """
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        use_pc_color: bool,
        n_points: int,
        subs_factor: int = 1,  # 1 means no subsampling
    ) -> None:
        """
        Args:
            data_path: Path to replay buffer directory
            n_obs_steps: Number of observation timesteps (T_obs)
            n_pred_steps: Number of prediction timesteps (T_pred)
            use_pc_color: If True, include RGB colors (6 channels), else XYZ only (3 channels)
            n_points: Maximum number of points to sample from point cloud
            subs_factor: Temporal subsampling factor (1 = no subsampling)
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "pcd_xyz"]
        data_key_first_k = {"pcd_xyz": n_obs_steps * subs_factor}
        if use_pc_color:
            data_keys.append("pcd_color")
            data_key_first_k["pcd_color"] = n_obs_steps * subs_factor
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
        self.n_obs_steps = n_obs_steps
        self.n_prediction_steps = n_pred_steps
        self.subs_factor = subs_factor
        self.use_pc_color = use_pc_color
        self.n_points = n_points
        self.rng = np.random.default_rng()
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get a single training sample.
        
        Returns:
            pcd: (T_obs, N_points, 3 or 6) - Point cloud observations
            robot_state_obs: (T_obs, 10) - Observed robot states
            robot_state_pred: (T_pred, 10) - Future robot states to predict
        """
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        
        # Extract point clouds: (T_obs, N, 3)
        pcd = sample["pcd_xyz"][: cur_step_i : self.subs_factor]
        
        # Optionally concatenate RGB colors: (T_obs, N, 6)
        if self.use_pc_color:
            pcd_color = sample["pcd_color"][: cur_step_i : self.subs_factor]
            pcd_color = pcd_color.astype(np.float32) / 255.0  # Normalize to [0, 1]
            pcd = np.concatenate([pcd, pcd_color], axis=-1)
        
        # Extract robot states
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor].astype(np.float32)    # (T_obs, 10)
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor].astype(np.float32)   # (T_pred, 10)
        
        # Randomly subsample points if too many
        if pcd.shape[1] > self.n_points:
            random_indices = np.random.choice(pcd.shape[1], self.n_points, replace=False)
            pcd = pcd[:, random_indices]
        
        return pcd, robot_state_obs, robot_state_pred


if __name__ == "__main__":
    dataset = RobotDatasetPcd(
        data_path=DATA_DIRS.PFP / "open_fridge" / "train",
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
        use_pc_color=False,
        n_points=4096,
    )
    i = 20
    obs, robot_state_obs, robot_state_pred = dataset[i]
    print("robot_state_obs: ", robot_state_obs)
    print("robot_state_pred: ", robot_state_pred)
    print("done")
