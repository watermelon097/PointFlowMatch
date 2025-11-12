import hydra
from omegaconf import OmegaConf
from pfp import DATA_DIRS, set_seeds
from pfp.data.dataset_pcd import RobotDatasetPcd, RobotDatasetPtMaps
from pfp.data.dataset_images import RobotDatasetImages

import rerun as rr
from pfp.common.visualization import RerunViewer as RV
from pfp.common.visualization import RerunTraj

import open3d as o3d
from pfp.common.o3d_utils import make_pcd, merge_pcds

TASK_NAME = "unplug_charger"
MODE = "train"  # "train" or "valid"


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

    data_path_train = DATA_DIRS.PFP / TASK_NAME / MODE
    # data_path_valid = DATA_DIRS.PFP_REAL / TASK_NAME / MODE
    if cfg.obs_mode == "pcd":
        dataset_train = RobotDatasetPcd(data_path_train, **cfg.dataset)
        # dataset_valid = RobotDatasetPcd(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "rgb":
        dataset_train = RobotDatasetImages(data_path_train, **cfg.dataset)
        # dataset_valid = RobotDatasetImages(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "pt_map":
        dataset_train = RobotDatasetPtMaps(data_path_train, **cfg.dataset)
        # dataset_valid = RobotDatasetPcdWithMask(data_path_valid, **cfg.dataset)
    else:
        raise ValueError(f"Unknown observation mode: {cfg.obs_mode}")

    # Visualize the dataset
    RV("Dataset visualization")
    obs_traj = RerunTraj()
    pred_traj = RerunTraj()
    for i in range(len(dataset_train)):
        # pcd: (2, 4096, 3)
        # robot_state_obs: (2, 10)
        # robot_state_pred: (32, 10)
        pt_maps, mask_list, images, robot_state_obs, robot_state_pred = dataset_train[i]
        pcd_list = [
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pt_map[mask])) 
                    for pt_map, mask in zip(pt_maps[-1], mask_list[-1])
                    ]
        merged_pcd = merge_pcds(pcd_list)
        rr.set_time_sequence("timestep", i)
        RV.add_o3d_pointcloud("vis/pointcloud", merged_pcd)
        for i, img in enumerate(images[-1]):
            RV.add_rgb(f"vis/rgb_obs_{i}", img)
        obs_traj.add_traj("vis/robot_state_obs", robot_state_obs, size=0.008)
        pred_traj.add_traj("vis/robot_state_pred", robot_state_pred, size=0.004)
        rr.log("plot/gripper_pred", rr.Scalar(robot_state_pred[0, -1]))
        break


if __name__ == "__main__":
    main()
