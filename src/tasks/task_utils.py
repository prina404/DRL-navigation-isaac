from tasks.go2_lidar_only import *
from tasks.go2_vision_depth import *

_TASK_DICT = {
    "go2_lidar_full": Go2LidarFull,
    "go2_lidar_empty": Go2LidarEmpty,
    "go2_vision_full": Go2VisionFull,
    "go2_vision_empty": Go2VisionEmpty,
    "go2_depth_full": Go2DepthFull,
}


def get_env_config(name: str) -> ManagerBasedRLEnvCfg:
    if name not in _TASK_DICT:
        raise ValueError(f"Env {name} not found. Available envs: {list(_TASK_DICT.keys())}")
    return _TASK_DICT[name]()
