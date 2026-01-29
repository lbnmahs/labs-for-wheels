from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
import isaaclab.sim as sim_utils

from . import WHEELEDLAB_ASSETS_DATA_DIR

# MuSHR actuator configurations
MUSHR_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
        velocity_limit=10.0,
        effort_limit=3.2,
        stiffness=100.0,
        damping=10.0,
        friction=0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort=1.05,
        effort_limit=0.25,
        velocity_limit=450.0,
        stiffness=0,
        damping=1000.0,
        friction=0.0,
    ),
}

MUSHR_SUS_ACTUATOR_CFG = {  # 4WD with suspension
    **MUSHR_ACTUATOR_CFG,
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_suspension"],
        effort_limit=None,  # Passive joint
        velocity_limit=None,
        stiffness=1e8,
        damping=0.0,
        friction=0.5,
    ),
}

# MuSHR 2WD Configuration with suspension
MUSHR_SUS_2WD_ACTUATOR_CFG = {
    "steering_joints": MUSHR_SUS_ACTUATOR_CFG["steering_joints"],
    "suspension": MUSHR_SUS_ACTUATOR_CFG["suspension"],
    "throttle_joints": MUSHR_SUS_ACTUATOR_CFG["throttle_joints"].replace(
        joint_names_expr=["back_.*throttle"],
        effort_limit=0.5,  # More torque for two wheel drive
    ),
    "passive_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_.*throttle"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
}

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        'back_left_wheel_throttle': 0.0,
        'back_right_wheel_throttle': 0.0,
        'front_left_wheel_steer': 0.0,
        'front_right_wheel_steer': 0.0,
        'front_left_wheel_throttle': 0.0,
        'front_right_wheel_throttle': 0.0,
    },
)

MUSHR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/UWPRL/mushr_nano.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,  # m/s
            max_angular_velocity=100000.0,  # deg/s
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=MUSHR_ACTUATOR_CFG,
)

MUSHR_SUS_CFG = MUSHR_CFG.replace(
    spawn=MUSHR_CFG.spawn.replace(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/UWRLL/mushr_nano_v2.usd",
    ),
    init_state=_ZERO_INIT_STATES.replace(
        joint_pos={
            **_ZERO_INIT_STATES.joint_pos,
            'front_left_wheel_suspension': 0.0,
            'front_right_wheel_suspension': 0.0,
            'back_left_wheel_suspension': 0.0,
            'back_right_wheel_suspension': 0.0,
        }
    ),
    actuators=MUSHR_SUS_ACTUATOR_CFG,
)

MUSHR_SUS_2WD_CFG = MUSHR_SUS_CFG.replace(
    actuators=MUSHR_SUS_2WD_ACTUATOR_CFG,
)
