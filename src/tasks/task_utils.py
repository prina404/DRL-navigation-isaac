from tasks.go2_lidar_only import *
from tasks.go2_vision_depth import *

_TASK_DICT = {
    "go2_lidar_empty": Go2LidarEmpty,
    "go2_lidar_full": Go2LidarFull,
    "go2_vision_empty": Go2VisionEmpty,
    "go2_vision_full": Go2VisionFull,
    "go2_depth_full": Go2DepthFull,
    "go2_vision_depth_full": Go2VisionDepthFull,
}


def get_env_config(task_name: str) -> ManagerBasedRLEnvCfg:
    if task_name not in _TASK_DICT:
        raise ValueError(f"Env {task_name} not found. Available envs: {list(_TASK_DICT.keys())}")

    return _TASK_DICT[task_name]()
