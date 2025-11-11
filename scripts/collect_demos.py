"""
Collect RLBench demonstration episodes and save them to disk as replay buffers.
Extracts point clouds, images, and robot states from expert demonstrations.
"""
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from rlbench.backend.observation import Observation
from pfp import DATA_DIRS, set_seeds
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.visualization import RerunViewer as RV


# def project_points_to_cameras(
#     points: np.ndarray,
#     obs: Observation,
#     image_size: tuple[int, int] = (128, 128),
# ) -> dict[str, np.ndarray]:
#     """
#     Project 3D points to pixel coordinates for each camera.
    
#     Args:
#         points: (N, 3) - 3D points in world coordinates
#         obs: Observation object containing camera intrinsics and extrinsics
#         image_size: (H, W) - Image dimensions for validity checking
        
#     Returns:
#         Dictionary with keys for each camera name, containing:
#             - 'pixel_coords': (N, 2) - Pixel coordinates [u, v] for each point
#             - 'valid_mask': (N,) - Boolean mask indicating which points are visible
#                                  (in front of camera and within image bounds)
#             - 'depth': (N,) - Depth values in camera space
#     """
#     camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
#     H, W = image_size
    
#     results = []
    
#     # Convert points to homogeneous coordinates (N, 4)
#     N = points.shape[0]
#     points_homogeneous = np.concatenate([points, np.ones((N, 1))], axis=1)  # (N, 4)
    
#     for cam_name in camera_names:
#         # Get camera intrinsics (3x3) and extrinsics (4x4)
#         intrinsics = obs.misc[f'{cam_name}_camera_intrinsics']  # (3, 3)
#         extrinsics = obs.misc[f'{cam_name}_camera_extrinsics']  # (4, 4)
        
#         # Transform points from world to camera space
#         # P_cam = extrinsics @ P_world (homogeneous)
#         points_cam = (extrinsics @ points_homogeneous.T).T  # (N, 4)
        
#         # Extract depth (z coordinate in camera space)
#         depth = points_cam[:, 2]  # (N,)
        
#         # Check if points are in front of camera (z > 0)
#         in_front = depth > 0
        
#         # Project to image plane using intrinsics
#         # Only project points that are in front of camera
#         points_cam_3d = points_cam[:, :3]  # (N, 3)
#         points_image = (intrinsics @ points_cam_3d.T).T  # (N, 3)
        
#         # Convert to pixel coordinates [u, v]
#         # u = x/z, v = y/z
#         pixel_coords = np.zeros((N, 2))
#         valid_depth = depth > 1e-6  # Avoid division by zero
#         pixel_coords[valid_depth, 0] = points_image[valid_depth, 0] / depth[valid_depth]  # u
#         pixel_coords[valid_depth, 1] = points_image[valid_depth, 1] / depth[valid_depth]  # v
        
#         # Check if pixels are within image bounds
#         in_bounds = (
#             (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
#             (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
#         )
        
#         # Points are valid if they're in front of camera AND within image bounds
#         valid_mask = in_front & in_bounds
        
#         results.append({
#             "pixel_coords": pixel_coords.astype(np.float32),  # (N, 2)
#             "valid_mask": valid_mask,  # (N,)
#         })
    
#     return results

# For valid, call it with: --config-name=collect_demos_valid
# To actually save the data, remember to call it with: save_data=True
@hydra.main(version_base=None, config_path="../conf", config_name="collect_demos_train")
def main(cfg: OmegaConf):
    set_seeds(cfg.seed)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    print(OmegaConf.to_yaml(cfg))

    assert cfg.mode in ["train", "valid"]
    if cfg.env_config.vis:
        RV("pfp_collect_demos")
    env = RLBenchEnv(use_pc_color=False, **cfg.env_config)
    if cfg.save_data:
        data_path = DATA_DIRS.PFP / cfg.env_config.task_name / cfg.mode
        if data_path.is_dir():
            print(f"ERROR: Data path {data_path} already exists! Exiting...")
            return
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="a")

    # cfg.num_episodes: number of episodes to collect/ 完整演示轨迹数量
    for _ in tqdm(range(cfg.num_episodes)):
        data_history = list()
        demo = env.task.get_demos(1, live_demos=True)[0]
        observations: list[Observation] = demo._observations
        for obs in observations:
            # Extract robot state: (10,) = [pos(3), rot6d(6), gripper(1)]
            robot_state = env.get_robot_state(obs)
            
            # Extract images from 5 cameras: (5, 128, 128, 3)
            # [right_shoulder_rgb, left_shoulder_rgb, overhead_rgb, front_rgb, wrist_rgb]
            images = env.get_images(obs)
            
            # Point cloud with rgb
            # pcd = env.get_pcd(obs)
            pt_maps, mask_list = env.get_pt_maps_with_mask(obs)
            # pcd_xyz = np.asarray(pcd.points)
            # pcd_color = np.asarray(pcd.colors)
            
            # Store data for this timestep
            data_history.append(
                {
                    "pt_maps": pt_maps,       # (N, 3)
                    "mask_list": mask_list,   # (N,)
                    "robot_state": robot_state.astype(np.float32),   # (10,) float32
                    "images": images,           # (5, 128, 128, 3)
                    # "pixel_projections": pixel_projections,   # dict of {cam_name: {'pixel_coords': (N, 2), 'valid_mask': (N,)}}
                }
            )
            env.vis_step(robot_state, images)

        if cfg.save_data:
            replay_buffer.add_episode_from_list(data_history, compressors="disk")
            print(f"Saved episode with {len(data_history)} steps to disk.")

        # while True:
        #     env.step(robot_state)
    return


if __name__ == "__main__":
    main()
