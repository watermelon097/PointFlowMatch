"""
Train PointFlowMatch policy on RLBench demonstrations using Composer framework.
Supports both point cloud and RGB image observations with configurable model architectures.
"""
import os
import hydra  # Configuration management framework - loads YAML configs and enables easy experimentation
import wandb
import subprocess
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Composer (MosaicML): Production-ready PyTorch training framework
from composer.trainer import Trainer          # High-level trainer that handles training loop, checkpointing, distributed training
from composer.loggers import WandBLogger      # WandB integration for experiment tracking
from composer.callbacks import LRMonitor      # Callback to log learning rate during training
from composer.models import ComposerModel     # Base class for models - requires forward() and loss() methods
from composer.algorithms import EMA           # Exponential Moving Average for model weights

from diffusion_policy.model.common.lr_scheduler import get_scheduler
from pfp import DEVICE, DATA_DIRS, set_seeds
from pfp.data.dataset_pcd import RobotDatasetPcd
from pfp.data.dataset_images import RobotDatasetImages
from pfp.data.dataset_images import RobotDatasetRGBD


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: OmegaConf):
    """
    Training workflow for PointFlowMatch policy on robot manipulation tasks.
    
    WORKFLOW OVERVIEW:
    ==================
    1. Configuration Setup (Hydra):
       - Loads hierarchical configs: train.yaml → model/flow.yaml → backbone/pointnet.yaml
       - Resolves variable references (e.g., ${backbone} → PointNetBackbone config)
       - Enables CLI overrides: python train.py task_name=open_box epochs=2000
    
    2. Data Loading:
       - Loads replay buffers from DATA_DIRS.PFP/{task_name}/train|valid
       - Dataset returns: (observations, robot_state_obs, robot_state_pred)
         * Point cloud mode: pcd (T_obs, N, 3|6), robot states (T_obs|T_pred, 10)
         * RGB mode: images (T_obs, 5, 128, 128, 3), robot states (T_obs|T_pred, 10)
       - Creates PyTorch DataLoaders with optional multiprocessing
    
    3. Model Instantiation (Hydra + Composer):
       - Recursively builds model from config using hydra.utils.instantiate()
       - Structure: FMPolicy(obs_encoder=PointNetBackbone, diffusion_net=UNet1D)
         * obs_encoder: Encodes point clouds → feature vectors (B, T, N, 3) → (B, 532)
         * diffusion_net: Conditional UNet predicting velocity fields for flow matching
       - Model inherits from ComposerModel (requires forward() and loss() methods)
    
    4. Training Setup:
       - Optimizer: AdamW with specified hyperparameters
       - LR Scheduler: Cosine schedule with warmup (stepped per batch)
       - Logger: WandB for experiment tracking
       - Algorithms: Optional EMA (Exponential Moving Average) for stable weights
    
    5. Composer Trainer:
       - Handles complete training loop (no manual epoch/batch loops needed)
       - Features: auto checkpointing, distributed training, mixed precision
       - Calls model.loss(outputs, batch) each iteration
       - Logs metrics to WandB, saves checkpoints every N epochs
    
    6. Training Execution:
       - trainer.fit() runs full training (validation + checkpointing automatic)
       - Saves final config.yaml with checkpoint for reproducibility
       - Auto-launches evaluation script after training completes
    
    MODEL ARCHITECTURE:
    ===================
    FMPolicy (Flow Matching):
      Input: Point clouds (B, T_obs, N_points, 3) + robot states (B, T_obs, 10)
      ↓
      obs_encoder (PointNetBackbone):
        PointNet (3→64→128→1024) → MLP (1024→512→256)
        Concat with robot_state_obs → Flatten across time
        Output: nx (B, obs_features_dim * T_obs) = (B, 532)
      ↓
      Flow Matching Training:
        z_t = t * target + (1-t) * noise      (interpolate between noise and target)
        target_vel = target - noise            (ground truth velocity)
      ↓
      diffusion_net (Conditional UNet1D):
        Input: z_t (B, T_pred, 10)
        Condition: nx (B, 532)
        Time embedding: t
        Output: pred_vel (B, T_pred, 10)
      ↓
      Loss: MSE(pred_vel, target_vel) for [xyz(3), rot6d(6), gripper(1)]
    
    HYDRA FRAMEWORK BEHAVIOR:
    =========================
    The @hydra.main decorator is what enables config-driven execution:
    
    How Hydra Works:
    1. Intercepts function call before main() executes
    2. Reads config_name='train' → loads conf/train.yaml
    3. Processes 'defaults' section → loads model/flow.yaml, backbone/pointnet.yaml
    4. Merges all configs into single OmegaConf object
    5. Resolves variable references like ${backbone}, ${obs_features_dim}
    6. Applies command-line overrides (python train.py epochs=2000)
    7. Creates output directory (outputs/YYYY-MM-DD/HH-MM-SS/)
    8. Changes working directory to output dir
    9. Passes merged config as 'cfg' parameter to main()
    
    Why Use Hydra?
    - Separation of code and configuration (no hardcoded hyperparameters)
    - Easy experiment management (change configs without editing code)
    - Reproducibility (config saved with each run)
    - Composition (mix and match model/backbone/dataset configs)
    - CLI overrides for quick experiments
    
    Example: python train.py task_name=open_box model=flow_so3 backbone=pointmlp epochs=2000
    → Hydra loads different configs and overrides epochs, all without code changes!
    """
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

    data_path_train = DATA_DIRS.PFP / cfg.task_name / "train"
    data_path_valid = DATA_DIRS.PFP / cfg.task_name / "valid"
    
    # Dataset returns batches with shapes:
    # - Point cloud mode: (pcd, robot_state_obs, robot_state_pred)
    #   * pcd: (T_obs, N_points, 3 or 6) - XYZ or XYZRGB point clouds
    #   * robot_state_obs: (T_obs, 10) - observed robot states [pos(3), rot6d(6), gripper(1)]
    #   * robot_state_pred: (T_pred, 10) - predicted robot states
    # - RGB mode: (images, robot_state_obs, robot_state_pred)
    #   * images: (T_obs, 5, 128, 128, 3) - images from 5 cameras
    #   * robot_state_obs: (T_obs, 10) - observed robot states
    #   * robot_state_pred: (T_pred, 10) - predicted robot states
    # - RGBD mode: (images, depths, robot_state_obs, robot_state_pred)
    #   * images: (T_obs, 5, 128, 128, 3) - images from 5 cameras
    #   * depths: (T_obs, 5, 128, 128) - depth images from 5 cameras
    #   * robot_state_obs: (T_obs, 10) - observed robot states
    #   * robot_state_pred: (T_pred, 10) - predicted robot states
    if cfg.obs_mode == "pcd":
        dataset_train = RobotDatasetPcd(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetPcd(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "rgb":
        dataset_train = RobotDatasetImages(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetImages(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "rgbd":
        dataset_train = RobotDatasetRGBD(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetRGBD(data_path_valid, **cfg.dataset)
    else:
        raise ValueError(f"Unknown observation mode: {cfg.obs_mode}")
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )

    # Hydra instantiates model from config (e.g., FMPolicy with specified backbone)
    # Model initialization hierarchy:
    # 1. train.yaml (defaults) → loads model/flow.yaml + backbone/pointnet.yaml
    # 2. model/flow.yaml defines FMPolicy with obs_encoder=${backbone}
    # 3. backbone/pointnet.yaml defines PointNetBackbone (gets injected into obs_encoder)
    # 4. hydra.utils.instantiate() recursively builds: FMPolicy(obs_encoder=PointNetBackbone(...), ...)
    composer_model: ComposerModel = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, composer_model.parameters())
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_training_steps=(len(dataloader_train) * cfg.epochs),
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
    )

    wandb_logger = WandBLogger(
        project="pfp-train-fixed",
        entity="rl-lab-chisari",
        init_kwargs={
            "config": OmegaConf.to_container(cfg),
            "mode": "online" if cfg.log_wandb else "disabled",
        },
    )

    # Composer Trainer: Handles complete training workflow
    # - Automatic mixed precision, gradient accumulation
    # - Distributed training support (multi-GPU)
    # - Checkpointing with auto-resume
    # - Integration with loggers (WandB) and callbacks
    trainer = Trainer(
        model=composer_model,              # Model must inherit from ComposerModel
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_valid,
        max_duration=cfg.epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,  # Step LR scheduler after each batch (not epoch)
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        loggers=[wandb_logger],            # Log metrics to WandB
        callbacks=[LRMonitor()],           # Monitor learning rate changes
        save_folder="ckpt/{run_name}",
        save_interval=f"{cfg.save_each_n_epochs}ep",  # Checkpoint every N epochs
        save_num_checkpoints_to_keep=3,    # Keep only last 3 checkpoints
        algorithms=[EMA()] if cfg.use_ema else None,  # Optional: Exponential Moving Average
        run_name=cfg.run_name,             # Set this to continue training from previous ckpt
        autoresume=True if cfg.run_name is not None else False,  # Auto-resume if run_name provided
        spin_dataloaders=False
    )
    wandb.watch(composer_model)
    # Save the used cfg for inference - important for reproducing exact model configuration
    OmegaConf.save(cfg, "ckpt/" + trainer.state.run_name + "/config.yaml")

    # Start training - Composer handles the entire training loop
    trainer.fit()
    run_name = trainer.state.run_name
    wandb.finish()
    trainer.close()

    _ = subprocess.Popen(
        [
            "bash",
            "bash/start_eval.sh",
            f"{os.environ['CUDA_VISIBLE_DEVICES']}",
            f"{run_name}",
        ],
        start_new_session=True,
    )
    return


if __name__ == "__main__":
    main()
