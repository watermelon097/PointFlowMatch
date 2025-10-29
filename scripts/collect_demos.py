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
            
            # Extract depths from 5 cameras: (5, 128, 128, 1)
            depths = env.get_depths(obs)
            
            # Store data for this timestep
            data_history.append(
                {
                    "images": images.astype(np.uint8),           # (5, 128, 128, 3)
                    "depth": depths.astype(np.float32),           # (5, 128, 128, 1)
                    "robot_state": robot_state.astype(np.float32),   # (10,) float32
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
