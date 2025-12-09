"""
Compatibility shim that re-exports dataset utilities.

Prefer importing from `pfp.data.dataset_pcd`.
"""
from pfp import DATA_DIRS
from pfp.data.dataset_pcd import (
    rand_range,
    augment_pcd_data,
    RobotDatasetPcd,
    RobotDatasetPixelAlignedPcd,
)

__all__ = [
    "rand_range",
    "augment_pcd_data",
    "RobotDatasetPcd",
    "RobotDatasetPixelAlignedPcd",
]


if __name__ == "__main__":

    dataset = RobotDatasetPixelAlignedPcd(
        data_path=DATA_DIRS.PFP / "unplug_charger" / "train",
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
        use_pc_color=False,
        n_points=4096,
    )
    i = 20
    pcd, pixel_idx, map_idx, images, robot_state_obs, robot_state_pred = dataset[i]
    print("pcd: ", pcd.shape)
    print("pixel_idx: ", pixel_idx.shape)
    print("map_idx: ", map_idx.shape)
    print("mask_list: ", map_idx.shape)
    print("images: ", images.shape)
    print("robot_state_obs: ", robot_state_obs)
    print("robot_state_pred: ", robot_state_pred)
    print("done")
