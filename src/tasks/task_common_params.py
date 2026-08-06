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
        # The "performance" preset disables the DL denoiser, which blacks out the interactive
        # Kit viewport under the RTX Real-Time 2.0 (path tracing) renderer used by Isaac Sim 6.
        # Re-enable it here (applied after the preset) so `--viz kit` shows the scene.
        enable_dl_denoiser=True,
        dlss_mode=3,  # defaults to 1 in performance mode
        carb_settings={
            "/rtx/sceneDb/ambientLightIntensity": 0.1,
        },
    ),
    "episode_length_s": 15.0,
    "decimation": 16,
}
