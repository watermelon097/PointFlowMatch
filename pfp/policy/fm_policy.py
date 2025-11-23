from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import pypose as pp
import torchvision.transforms as T
import torch.nn.functional as F
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import init_random_traj_th
from pfp.common.fm_utils import get_timesteps, extract_dino_features_from_map
from pfp.data.dataset_pcd import augment_pcd_data



class FMPolicy(ComposerModel, BasePolicy):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        time_conditioning: bool,
        obs_encoder: nn.Module,
        image_encoder: nn.Module,
        diffusion_net: nn.Module,
        augment_data: bool = False,
        loss_weights: dict[int] = None,
        pos_emb_scale: int = 20,
        norm_pcd_center: list = None,
        noise_type: str = "gaussian",
        noise_scale: float = 1.0,
        loss_type: str = "l2",
        flow_schedule: str = "linear",
        exp_scale: float = None,
        snr_sampler: str = "uniform",
        subs_factor: int = 1,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps, subs_factor)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.pos_emb_scale = pos_emb_scale
        self.num_k_infer = num_k_infer
        self.time_conditioning = time_conditioning
        self.obs_encoder = obs_encoder # pcd encoder
        self.image_encoder = image_encoder # image encoder
        self.diffusion_net = diffusion_net # velocity predictor
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        self.snr_sampler = snr_sampler
        self.image_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # self.dino_decoder = nn.Sequential(
        #     nn.Linear(384, 256),
        #     nn.Mish()
        # )
        self.img_proj = nn.Linear(384, 256)
        self.view_weight = nn.Parameter(torch.zeros(5))
        self.num_patches = 16
        if loss_type == "l2":
            self.loss_fun = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fun = nn.L1Loss()
        else:
            raise NotImplementedError
        return

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def set_flow_schedule(self, flow_schedule: str, exp_scale: float):
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        return

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] += torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred = batch
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)
        return pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return augment_pcd_data(batch)

    def _init_noise(self, batch_size: int) -> torch.Tensor:
        B = batch_size
        T = self.n_pred_steps
        if self.noise_type == "gaussian":
            noise = torch.randn((batch_size, *self.ny_shape), device=DEVICE)
            return noise * self.noise_scale
        elif self.noise_type == "trajectory":
            return init_random_traj_th(batch_size, self.n_pred_steps, self.noise_scale)
        elif self.noise_type == "igso3":
            noise_pos = torch.randn((B, T, 3), device=DEVICE)
            noise_rot = pp.randn_SO3((B, T), device=DEVICE).matrix()
            noise_gripper = torch.randn((B, T, 1), device=DEVICE)
            noise = torch.cat(
                [noise_pos, noise_rot[..., :3, 0], noise_rot[..., :3, 1], noise_gripper], dim=-1
            )
            return noise
        else:
            raise NotImplementedError

    def _sample_snr(self, batch_size: int) -> torch.Tensor:
        if self.snr_sampler == "uniform":
            return torch.rand((batch_size, 1, 1), device=DEVICE)
        elif self.snr_sampler == "logit_normal":
            return torch.sigmoid(torch.randn((batch_size, 1, 1), device=DEVICE))
        else:
            raise NotImplementedError

    # ############### Training ################

    def forward(self, batch):
        """batch is the output of the dataloader"""
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        outputs: the output of the forward pass
        batch: the output of the dataloader
        """
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred = batch
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred
        )
        loss = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/train/xyz": loss_xyz.item(),
                "loss/train/rot6d": loss_rot6d.item(),
                "loss/train/grip": loss_grip.item(),
            }
        )
        return loss

    def calculate_loss(
        self, 
        pcd: torch.Tensor, 
        pixel_idx: torch.Tensor, 
        map_idx: torch.Tensor, 
        images: torch.Tensor, 
        robot_state_obs: torch.Tensor, 
        robot_state_pred: torch.Tensor
    ):
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        # print("loss: nx shape: ", nx.shape)
        # print("loss: pcd shape: ", pcd.shape) 
        
        # Process images: [B, T, num_cameras, H, W, C]
        B, T, num_cameras, H, W, C = images.shape
        # print("images shape: ", images.shape)
        # print("pixel_idx shape:", pixel_idx.shape)
        # print("map_idx shape:", map_idx.shape)

        # Convert to float and normalize
        if images.dtype != torch.float32:
            images = images.float() / 255.0
        
        # Reshape: [B, T, num_cameras, H, W, C] -> [B*T*num_cameras, H, W, C]
        images = images.reshape(-1, H, W, C)

        # Permute: [B*T*num_cameras, H, W, C] -> [B*T*num_cameras, C, H, W]
        images = images.permute(0, 3, 1, 2).contiguous()
        images = images.to(DEVICE)

        # Apply transforms and extract DINOv2 features
        images = self.image_transform(images)  # [B*T*num_cameras, 3, 224, 224]
        with torch.no_grad():
            dino_feats = self.image_encoder(images) # [B*T*num_cameras, 384]

        # image_features = self.dino_decoder(dino_feats)
        # print("dino_feats shape: ", dino_feats.shape)
        image_features = dino_feats.view(B, T, num_cameras, 384)
        image_features = self.img_proj(image_features)
        weight = F.softmax(self.view_weight, dim=-1)
        weight = weight.view(1, 1, num_cameras, 1)
        image_features = (image_features * weight).sum(dim=2)
        # image_features,_ = torch.max(image_features, dim=2)
        image_features = image_features.reshape(B,-1)
        # print("image_features shape: ", image_features.shape)
        # # # Reshape to spatial tokens: [B*T*num_cameras, 384, 16, 16]
        # patch_map = dino_feats.view(B * T * num_cameras, self.num_patches, self.num_patches, 384)
        # print("patch_map shape: ", patch_map.shape)
        # patch_map = patch_map.permute(0, 3, 1, 2)  # [B*T*num_cameras, 384, 16, 16]
        # print("patch_map shape: ", patch_map.shape)
        # dense_feats = self.dino_decoder(patch_map) # [B*T*num_cameras, 128, 128, 128]
        # chunk_size = 16  # 或 64，根据显存调整
        # dense_chunks = []
        # for chunk in patch_map.split(chunk_size, dim=0):
        #     dense_chunks.append(
        #         F.interpolate(chunk, size=(128, 128), mode="bilinear", align_corners=False)
        #     )
        # dense_feats = torch.cat(dense_chunks, dim=0).view(B, T, num_cameras, 384, 128, 128)
        # Decode dense features in smaller chunks to avoid exceeding INT_MAX elements
        # chunk_size = 256
        # dense_chunks = []
        # for chunk in patch_map.split(chunk_size, dim=0):
        #     dense_chunks.append(self.dino_decoder(chunk))
        # dense_feats = torch.cat(dense_chunks, dim=0)  # [B*T*num_cameras, 128, 128, 128]
        # dense_feats = dense_feats.view(B*T, num_cameras, 128, 128, 128)
        # print("dense_feats shape: ", dense_feats.shape)
        # # Move to device
        # pixel_idx = pixel_idx.to(DEVICE)  # [B, T, N_points, 2]
        # map_idx = map_idx.to(DEVICE).long()  # [B, T, N_points]

        # _, _, N_points, _ = pcd.shape

        # # # Clamp pixel coordinates to valid range and flatten batch/time dims
        # rows = pixel_idx[..., 0].clamp(0, 127).long()  # [B, T, N_points]
        # cols = pixel_idx[..., 1].clamp(0, 127).long()

        # # # dense_feats_flat = dense_feats.view(B * T, num_cameras, 128, 128, 128)
        # rows_flat = rows.view(B * T, N_points)
        # cols_flat = cols.view(B * T, N_points)
        # map_idx_flat = map_idx.view(B * T, N_points)
        # print("map_idx_flat shape: ", map_idx_flat.shape)
        # batch_indices = torch.arange(B * T, device=DEVICE).unsqueeze(1).expand(-1, N_points)
        # point_features = dense_feats[
        #     batch_indices,
        #     map_idx_flat,
        #     :,
        #     rows_flat,
        #     cols_flat,
        # ]  # [B*T, N_points, 128]

        # # Reshape back and aggregate
        # print("point_features shape: ", point_features.shape)
        # point_features = point_features.view(B, T, N_points, 128)
        # print("point_features shape: ", point_features.shape)
        # # Concatenate the T dimension (dim=1) along the channel dimension
        # point_features_reshaped = torch.cat([point_features[:, t, :, :] for t in range(point_features.shape[1])], dim=-1)
        # print("point_features_reshaped shape: ", point_features_reshaped.shape)
        # # img_features = extract_dino_features_from_map(self.image_encoder, images, map_idx, pixel_idx)
        nx = torch.cat([nx, image_features], dim=1)
        # # sample features from patch_map


        ny: torch.Tensor = robot_state_pred

        t = self._sample_snr(B)
        z0 = self._init_noise(ny.shape[0])
        z1 = ny
        z_t = t * z1 + (1.0 - t) * z0
        target_vel = z1 - z0
        timesteps = t.squeeze() * self.pos_emb_scale if self.time_conditioning else None
        pred_vel = self.diffusion_net(z_t, timesteps, global_cond=nx)
        loss_xyz = self.loss_fun(pred_vel[..., :3], target_vel[..., :3])
        loss_rot6d = self.loss_fun(pred_vel[..., 3:9], target_vel[..., 3:9])
        loss_grip = self.loss_fun(pred_vel[..., 9], target_vel[..., 9])
        return loss_xyz, loss_rot6d, loss_grip
    # ############### Inference ################

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        """
        batch: the output of the eval dataloader
        outputs: the output of the forward pass
        """
        batch = self._norm_data(batch)
        pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred = batch

        # Eval loss
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred
        )
        loss_total = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/eval/xyz": loss_xyz.item(),
                "loss/eval/rot6d": loss_rot6d.item(),
                "loss/eval/grip": loss_grip.item(),
                "loss/eval/total": loss_total.item(),
            }
        )

        # Eval metrics
        pred_y = self.infer_y(pcd, robot_state_obs, images=images)
        mse_xyz = nn.functional.mse_loss(pred_y[..., :3], robot_state_pred[..., :3])
        mse_rot6d = nn.functional.mse_loss(pred_y[..., 3:9], robot_state_pred[..., 3:9])
        mse_grip = nn.functional.mse_loss(pred_y[..., 9], robot_state_pred[..., 9])
        self.logger.log_metrics(
            {
                "metrics/eval/mse_xyz": mse_xyz.item(),
                "metrics/eval/mse_rot6d": mse_rot6d.item(),
                "metrics/eval/mse_grip": mse_grip.item(),
            }
        )
        return pred_y

    def infer_from_np(self, obs, robot_state: np.ndarray) -> np.ndarray:
        # 处理元组格式的观测 (pcd_with_idx 模式)
        if isinstance(obs, tuple):
            pcd, images,pixel_idx, map_idx = obs
            pcd_th = torch.tensor(pcd, device=DEVICE).unsqueeze(0)  # (1, N, 3)
            robot_state_th = torch.tensor(robot_state, device=DEVICE).unsqueeze(0)  # (1, T, 10)
            pcd_th = self._norm_obs(pcd_th)
            robot_state_th = self._norm_robot_state(robot_state_th)
            images_th = torch.tensor(images, device=DEVICE).unsqueeze(0)
            # 注意：推理时没有 images，所以传入 None
            ny = self.infer_y(
                pcd_th,
                robot_state_th,
                images=images_th,
                return_traj=True,
            )
            ny = self._denorm_robot_state(ny)
            ny = ny.squeeze().detach().cpu().numpy()
            return ny  # (K, T, 10)
        else:
            # 如果不是元组，调用父类方法
            return super().infer_from_np(obs, robot_state)

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        images: torch.Tensor = None,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        # Process images if provided (same as in calculate_loss)
        if images is not None:
            # Process images: [B, T, num_cameras, H, W, C]
            B, T, num_cameras, H, W, C = images.shape
            
            # Convert to float and normalize
            if images.dtype != torch.float32:
                images = images.float() / 255.0
            
            # Reshape: [B, T, num_cameras, H, W, C] -> [B*T*num_cameras, H, W, C]
            images = images.reshape(-1, H, W, C)
            
            # Permute: [B*T*num_cameras, H, W, C] -> [B*T*num_cameras, C, H, W]
            images = images.permute(0, 3, 1, 2).contiguous()
            images = images.to(DEVICE)
            
            # Apply transforms and extract DINOv2 features
            images = self.image_transform(images)  # [B*T*num_cameras, 3, 224, 224]
            with torch.no_grad():
                dino_feats = self.image_encoder(images)  # [B*T*num_cameras, 384]
            
            image_features = dino_feats.view(B, T, num_cameras, 384)
            image_features = self.img_proj(image_features)
            weight = F.softmax(self.view_weight, dim=-1)
            weight = weight.view(1, 1, num_cameras, 1)
            image_features = (image_features * weight).sum(dim=2)
            image_features = image_features.reshape(B, -1)
            
            # Concatenate with obs_encoder output
            nx = torch.cat([nx, image_features], dim=1)
            print("nx shape: ", nx.shape)
        
        B = nx.shape[0]
        z = self._init_noise(B) if noise is None else noise
        traj = [z]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            timesteps = torch.ones((B), device=DEVICE) * t0[i]
            timesteps *= self.pos_emb_scale
            vel_pred = self.diffusion_net(z, timesteps, global_cond=nx)
            z = z.detach().clone() + vel_pred * dt[i]
            traj.append(z)

        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int,
        flow_schedule: str = None,
        exp_scale: float = None,
        subs_factor: int = 1,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE, weights_only=False)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        # cfg.model.obs_encoder.encoder.random_crop = False
        cfg.model.subs_factor = subs_factor
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: FMPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class FMPolicyImage(FMPolicy):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image normalization is already done in the backbone, so here we just make it float
        """
        image = image.float() / 255.0
        return image

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
