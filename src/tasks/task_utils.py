from tasks.go2_lidar_only import *

_TASK_DICT = {
    "go2_lidar_empty": Go2LidarEmpty,
    "go2_lidar_full": Go2LidarFull,
}


def get_env_config(task_name: str) -> ManagerBasedRLEnvCfg:
    if task_name not in _TASK_DICT:
        raise ValueError(f"Env {task_name} not found. Available envs: {list(_TASK_DICT.keys())}")

    return _TASK_DICT[task_name]()
