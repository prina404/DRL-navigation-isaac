import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg

from cfg.CFG import ASSET_DIR

no_collision = CollisionPropertiesCfg(collision_enabled=False)

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/go2_rotated_joints.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # collision_props = CollisionPropertiesCfg(collision_enabled=False)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": -0.1,
            "RR_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": -0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": -1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": 1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": 1.5,
        },        
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=27.5,
            saturation_effort=27.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

# old asset cfg

# UNITREE_GO2_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ASSET_DIR}/go2_simplified_collisions.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#         ),
#         # collision_props = CollisionPropertiesCfg(collision_enabled=False)
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.6),
#         joint_pos={
#             ".*L_hip_joint": 0.1,
#             ".*R_hip_joint": -0.1,
#             "F[L,R]_thigh_joint": 0.8,
#             "R[L,R]_thigh_joint": 1.0,
#             ".*_calf_joint": -1.5,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "base_legs": DCMotorCfg(
#             joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
#             effort_limit=23.5,
#             saturation_effort=23.5,
#             velocity_limit=30.0,
#             stiffness=25.0,
#             damping=0.5,
#             friction=0.0,
#         ),
#     },
# )