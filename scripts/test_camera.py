"""
测试获取相机内参和外参
"""
import numpy as np
from pfp.envs.rlbench_env import RLBenchEnv

def test_camera_params():
    """测试获取相机参数"""
    print("=" * 60)
    print("测试相机内参和外参")
    print("=" * 60)
    
    # 初始化环境
    env = RLBenchEnv(
        task_name="reach_target",  # 简单任务
        voxel_size=0.01,
        n_points=4096,
        use_pc_color=False,
        headless=True,  # 无 GUI
        vis=False,      # 不可视化
        obs_mode="pcd"
    )
    
    # 重置环境
    env.reset()
    
    # 获取观察
    obs = env.task.get_observation()
    
    # 相机列表
    cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']
    
    print("\n📸 相机参数：")
    print("-" * 60)
    
    for cam_name in cameras:
        print(f"\n{cam_name.upper().replace('_', ' ')} Camera:")
        
        # 获取内参
        intrinsics = obs.misc[f'{cam_name}_camera_intrinsics']
        print(f"  内参矩阵 (Intrinsics):")
        print(f"    {intrinsics}")
        print(f"    焦距 fx={intrinsics[0,0]:.2f}, fy={intrinsics[1,1]:.2f}")
        print(f"    主点 cx={intrinsics[0,2]:.2f}, cy={intrinsics[1,2]:.2f}")
        
        # 获取外参
        extrinsics = obs.misc[f'{cam_name}_camera_extrinsics']
        position = extrinsics[:3, 3]
        print(f"  外参矩阵 (Extrinsics):")
        print(f"    相机位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        
        # Near/Far 裁剪面
        near = obs.misc[f'{cam_name}_camera_near']
        far = obs.misc[f'{cam_name}_camera_far']
        print(f"    近/远裁剪面: {near:.3f} / {far:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    env.env.shutdown()

if __name__ == "__main__":
    test_camera_params()