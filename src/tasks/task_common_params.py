import isaaclab.sim as sim_utils

PARAMS = {
    "dt": 0.005,
    "device": "cuda:0",
    "use_fabric": True,
    "render": sim_utils.RenderCfg(
        rendering_mode="performance",
        enable_translucency=True,
        enable_global_illumination=True,
        enable_reflections=True,
        enable_shadows=True,
        enable_ambient_occlusion=True,
        dlss_mode="3",  # defaults to 1 in performance mode
    ),
    "episode_length_s": 15.0,
    "decimation": 16,
}
