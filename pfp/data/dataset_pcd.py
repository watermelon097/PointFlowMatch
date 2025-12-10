"""
Point cloud dataset for robot manipulation tasks.
Loads sequences of point clouds and robot states from replay buffers with temporal subsampling.
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
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
        **kwargs,  # Accept additional kwargs for compatibility
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
            # sample N frames with interval subs_factor: [0, subs_factor, 2*subs_factor, ..., (N-1)*subs_factor]
            # (N-1)*subs_factor =
            # N *subs_factor - subs_factor + 1 = 
            # N *subs_factor - (subs_factor - 1)

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
        # based on curr setting, cur_step_i = 6
        cur_step_i = self.n_obs_steps * self.subs_factor
        
        # Extract point clouds: (T_obs, N, 3)
        # current subs_factor=3, 
        # shape of pcd = (2, N_points, 3)
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
            random_indices = self.rng.choice(pcd.shape[1], self.n_points, replace=False)
            pcd = pcd[:, random_indices]
        
        return pcd, robot_state_obs, robot_state_pred


class RobotDatasetPixelAlignedPcd(torch.utils.data.Dataset):
    """
    PyTorch dataset for pixel-aligned point cloud-based robot manipulation sequences.
    Keeps track of per-point pixel indices and source camera ids for pixel-level feature alignment.
    
    Returns:
        pcd: (T_obs, N_points, 3) - Point clouds
        pixel_idx: (T_obs, N_points, 2) - Pixel indices [u, v] for each point
        map_idx: (T_obs, N_points) - Camera/source map index for each point
        images: (T_obs, N_cam, H, W, 3) - RGB images from cameras
        robot_state_obs: (T_obs, 10) - Observed robot states [pos(3), rot6d(6), gripper(1)]
        robot_state_pred: (T_pred, 10) - Future robot states to predict
        dinov3_features: (T_obs, N_cam, feat_dim) - Optional DINOv3 features if return_dinov3_features=True
    """

    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        subs_factor: int = 1,
        use_pc_color: bool = False,
        n_points: int = 4096,
        dinov3_repo_root: str | Path | None = None,
        dinov3_weights_path: str | Path | None = None,
        dinov3_model_name: str = "dinov3_vits16",
        dinov3_device: str = "cuda",
        dinov3_image_size: int = 518,
        dinov3_feature_cache_dir: str | Path | None = None,
        return_dinov3_features: bool = False,
        dinov3_batch_size: int = 32,
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> None:
        """
        Args:
            data_path: Path to replay buffer directory
            n_obs_steps: Number of observation timesteps (T_obs)
            n_pred_steps: Number of prediction timesteps (T_pred)
            subs_factor: Temporal subsampling factor (1 = no subsampling)
            use_pc_color: If True, include RGB colors (currently not used in this dataset)
            n_points: Maximum number of points to sample from point cloud
            dinov3_repo_root: Path to DINOv3 repository root (for loading model)
            dinov3_weights_path: Path to DINOv3 model weights
            dinov3_model_name: Name of DINOv3 model to load
            dinov3_device: Device to run DINOv3 model on
            dinov3_image_size: Target image size for DINOv3 preprocessing
            dinov3_feature_cache_dir: Directory to cache DINOv3 features
            return_dinov3_features: If True, return DINOv3 features in __getitem__
            dinov3_batch_size: Batch size for DINOv3 feature computation
            **kwargs: Additional arguments for compatibility with other datasets
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "pcd", "pixel_idx", "map_idx", "images"]
        data_key_first_k = {
            "pcd": n_obs_steps * subs_factor,
            "pixel_idx": n_obs_steps * subs_factor,
            "map_idx": n_obs_steps * subs_factor,
            "images": n_obs_steps * subs_factor,
        }
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
        self.n_points = n_points
        self.return_dinov3_features = return_dinov3_features
        self.dinov3_batch_size = dinov3_batch_size
        self.rng = np.random.default_rng()  # Add rng for compatibility with other datasets

        self.dinov3_cache_dir = (
            Path(dinov3_feature_cache_dir).expanduser().resolve()
            if dinov3_feature_cache_dir is not None
            else None
        )
        if self.dinov3_cache_dir is not None:
            self.dinov3_cache_dir.mkdir(parents=True, exist_ok=True)

        self.dinov3 = None
        self.dinov3_device = dinov3_device
        self.dinov3_image_size = dinov3_image_size
        self.dinov3_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.dinov3_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        if dinov3_repo_root is not None and dinov3_weights_path is not None:
            self._init_dinov3_model(
                repo_root=dinov3_repo_root,
                weights_path=dinov3_weights_path,
                model_name=dinov3_model_name,
                device=dinov3_device,
            )
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get a single training sample.
        
        Returns:
            pcd: (T_obs, N_points, 3) - Point cloud observations
            pixel_idx: (T_obs, N_points, 2) - Pixel indices for each point
            map_idx: (T_obs, N_points) - Camera map indices for each point
            images: (T_obs, N_cam, H, W, 3) - RGB images from cameras
            robot_state_obs: (T_obs, 10) - Observed robot states
            robot_state_pred: (T_pred, 10) - Future robot states to predict
            dinov3_features: (T_obs, N_cam, feat_dim) - Optional DINOv3 features
        """
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        pcd = sample["pcd"][: cur_step_i : self.subs_factor]
        map_idx = sample["map_idx"][: cur_step_i : self.subs_factor]
        pixel_idx = sample["pixel_idx"][: cur_step_i : self.subs_factor]
        images = sample["images"][: cur_step_i : self.subs_factor]
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor].astype(np.float32)
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor].astype(np.float32)

        dinov3_features = None
        if self.dinov3 is not None:
            dinov3_features = self._get_or_compute_dinov3(idx, images)

        if pcd.shape[1] > self.n_points:
            random_indices = self.rng.choice(pcd.shape[1], self.n_points, replace=False)
            pcd = pcd[:, random_indices]
            pixel_idx = pixel_idx[:, random_indices]
            map_idx = map_idx[:, random_indices]
        sample_tuple: tuple = (
            pcd,
            pixel_idx,
            map_idx,
            images,
            robot_state_obs,
            robot_state_pred,
        )
        if self.return_dinov3_features:
            sample_tuple = sample_tuple + (dinov3_features,)
        return sample_tuple

    # DINO helpers -----------------------------------------------------------------
    def _init_dinov3_model(
        self,
        repo_root: str | Path,
        weights_path: str | Path,
        model_name: str,
        device: str,
    ) -> None:
        repo_root = str(Path(repo_root).expanduser().resolve())
        weights_path = str(Path(weights_path).expanduser().resolve())
        self.dinov3 = torch.hub.load(
            repo_or_dir=repo_root,
            model=model_name,
            source="local",
            weights=weights_path,
        )
        self.dinov3.eval().to(device)
        return

    def _get_or_compute_dinov3(self, idx: int, images: np.ndarray) -> np.ndarray | None:
        cached = self._load_cached_features(idx)
        if cached is not None:
            return cached

        feats = self._compute_dinov3_features(images)
        if feats is None:
            return None
        feats_np = feats.cpu().numpy()
        self._cache_features(idx, feats_np)
        return feats_np

    def _compute_dinov3_features(self, images: np.ndarray) -> torch.Tensor | None:
        if self.dinov3 is None:
            return None

        tensor = torch.from_numpy(images).float() / 255.0  # (T, N_cam, H, W, 3)
        T, N_cam, H, W, _ = tensor.shape
        tensor = tensor.permute(0, 1, 4, 2, 3).reshape(T * N_cam, 3, H, W)
        tensor = self._preprocess_for_dinov3(tensor)

        feats = []
        with torch.no_grad():
            for chunk in tensor.split(self.dinov3_batch_size):
                chunk = chunk.to(self.dinov3_device, non_blocking=True)
                outputs = self.dinov3(chunk)
                if isinstance(outputs, dict):
                    chunk_feats = (
                        outputs.get("x_norm_clstoken")
                        or outputs.get("last_hidden_state")
                        or next(iter(outputs.values()))
                    )
                else:
                    chunk_feats = outputs
                feats.append(chunk_feats.detach().cpu())
        feats_cat = torch.cat(feats, dim=0)
        return feats_cat.reshape(T, N_cam, -1)

    def _preprocess_for_dinov3(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.shape[-1] != self.dinov3_image_size or imgs.shape[-2] != self.dinov3_image_size:
            imgs = F.interpolate(
                imgs,
                size=(self.dinov3_image_size, self.dinov3_image_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = self.dinov3_mean.to(imgs.device)
        std = self.dinov3_std.to(imgs.device)
        imgs = (imgs - mean) / std
        return imgs

    def _cache_features(self, idx: int, feats: np.ndarray) -> None:
        if self.dinov3_cache_dir is None:
            return
        cache_path = self.dinov3_cache_dir / f"{idx:08d}.npy"
        np.save(cache_path, feats)
        return

    def _load_cached_features(self, idx: int) -> np.ndarray | None:
        if self.dinov3_cache_dir is None:
            return None
        cache_path = self.dinov3_cache_dir / f"{idx:08d}.npy"
        if cache_path.is_file():
            return np.load(cache_path)
        return None


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
