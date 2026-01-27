from loguru import logger
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sim.schemas import CollisionPropertiesCfg, modify_collision_properties
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from cfg.CFG import ASSET_DIR
import omni

no_collision = CollisionPropertiesCfg(collision_enabled=False)

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/go2_simplified_collisions_lidar.usd",
        activate_contact_sensors=False,
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        #collision_props = CollisionPropertiesCfg(collision_enabled=False)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

def enable_foot_collisions(env: RslRlVecEnvWrapper):
    stage = omni.usd.get_context().get_stage()
    
    art = env.unwrapped.scene.articulations['unitree_go2']
    links_lst = art.root_physx_view.link_paths
    flattened_links = [link for inner_lst in links_lst for link in inner_lst]
    for prim_root in flattened_links:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_root)):
            if not prim.IsValid():
                continue
            enabled_attr = prim.GetAttribute("physics:collisionEnabled")
            enabled_attr.Set(False)
            # if prim.endswith('foot'):
            #     continue
            print(f"\nPrim: {prim.GetPath()}  Type: {prim.GetTypeName()}")
            print("APIs:", [api for api in prim.GetAppliedSchemas()])            
            # if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            #     # Remove the API
            #     print(f"CollisionAPI removed from {prim.GetPath()}")
            # else:
            #     print(f"No CollisionAPI found on {prim.GetPath()}")
