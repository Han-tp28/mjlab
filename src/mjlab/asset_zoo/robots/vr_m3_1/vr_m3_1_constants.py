"""VR M3.1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##


VR_M3_1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "vr_m3_1" / "xmls" / "vr_m3_1_rl_scene.xml"
)

VR_M3_1_27DOF_XML = VR_M3_1_XML

assert VR_M3_1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  del meshdir
  return {}


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(VR_M3_1_XML))


##
# Actuator config.
##


MOTOR_SPECS = {
  "TD7080": {
    "armature": 0.5804369,
    "max_vel": 4.18,
    "max_tau": 102.0,
    "saturation_tau": 153.0,
  },
  "TD6070": {
    "armature": 0.3896782,
    "max_vel": 4.29,
    "max_tau": 66.0,
    "saturation_tau": 99.0,
  },
  "TD5060": {
    "armature": 0.1142512,
    "max_vel": 5.13,
    "max_tau": 34.0,
    "saturation_tau": 51.0,
  },
  "TD4052": {
    "armature": 0.0836482,
    "max_vel": 6.17,
    "max_tau": 11.0,
    "saturation_tau": 16.5,
  },
  "P110": {
    "armature": 0.1404,
    "max_vel": 14.653,
    "max_tau": 360.0,
    "saturation_tau": 816.667,
  },
  "P90N20": {
    "armature": 0.02864,
    "max_vel": 31.4,
    "max_tau": 130.0,
    "saturation_tau": 389.875,
  },
  "P60N30": {
    "armature": 0.03006,
    "max_vel": 16.747,
    "max_tau": 120.0,
    "saturation_tau": 479.97,
  },
}

VR_M3_1_27DOF_ACTUATOR_MOTORS = {
  "hip_pitch": "P110",
  "hip_roll": "P110",
  "hip_yaw": "P90N20",
  "knee_pitch": "P110",
  "ankle_pitch": "P60N30",
  "ankle_roll": "P60N30",
  "waist_yaw": "TD7080",
  "shoulder_pitch": "TD6070",
  "shoulder_roll": "TD6070",
  "shoulder_yaw": "TD5060",
  "elbow_pitch": "TD5060",
  "wrist_yaw": "TD4052",
  "wrist_roll": "TD4052",
  "wrist_pitch": "TD4052",
}

VR_M3_1_27DOF_STIFFNESS = {
  "hip_pitch": 150.0,
  "hip_roll": 150.0,
  "hip_yaw": 100.0,
  "knee_pitch": 200.0,
  "ankle_pitch": 200.0,
  "ankle_roll": 200.0,
  "waist_yaw": 367.0,
  "shoulder_pitch": 362.0,
  "shoulder_roll": 330.0,
  "shoulder_yaw": 278.0,
  "elbow_pitch": 278.0,
  "wrist_yaw": 292.0,
  "wrist_roll": 218.0,
  "wrist_pitch": 212.0,
}

VR_M3_1_27DOF_DAMPING = {
  "hip_pitch": 25.0,
  "hip_roll": 25.0,
  "hip_yaw": 4.0,
  "knee_pitch": 10.0,
  "ankle_pitch": 5.0,
  "ankle_roll": 5.0,
  "waist_yaw": 29.0,
  "shoulder_pitch": 14.0,
  "shoulder_roll": 13.0,
  "shoulder_yaw": 22.0,
  "elbow_pitch": 22.0,
  "wrist_yaw": 12.0,
  "wrist_roll": 9.0,
  "wrist_pitch": 8.0,
}

VR_M3_1_27DOF_STATIC_FRICTION = {
  "hip_pitch": 1.5,
  "hip_roll": 1.5,
  "hip_yaw": 0.8,
  "knee_pitch": 1.5,
  "ankle_pitch": 0.5,
  "ankle_roll": 0.5,
  "waist_yaw": 3.0,
  "shoulder_pitch": 3.0,
  "shoulder_roll": 3.0,
  "shoulder_yaw": 1.5,
  "elbow_pitch": 1.5,
  "wrist_yaw": 1.2,
  "wrist_roll": 0.8,
  "wrist_pitch": 1.2,
}

VR_M3_1_27DOF_VISCOUS_FRICTION = {
  "hip_pitch": 0.7,
  "hip_roll": 0.8,
  "hip_yaw": 0.2,
  "knee_pitch": 0.7,
  "ankle_pitch": 0.5,
  "ankle_roll": 0.1,
  "waist_yaw": 1.5,
  "shoulder_pitch": 3.0,
  "shoulder_roll": 3.0,
  "shoulder_yaw": 1.0,
  "elbow_pitch": 1.0,
  "wrist_yaw": 0.6,
  "wrist_roll": 0.6,
  "wrist_pitch": 0.6,
}

_ACTUATOR_TARGETS = {
  "hip_pitch": (".*hip_pitch_joint",),
  "hip_roll": (".*hip_roll_joint",),
  "hip_yaw": (".*hip_yaw_joint",),
  "knee_pitch": (".*knee_pitch_joint",),
  "ankle_pitch": (".*ankle_pitch_joint",),
  "ankle_roll": (".*ankle_roll_joint",),
  "waist_yaw": (".*waist_yaw_joint",),
  "shoulder_pitch": (".*shoulder_pitch_joint",),
  "shoulder_roll": (".*shoulder_roll_joint",),
  "shoulder_yaw": (".*shoulder_yaw_joint",),
  "elbow_pitch": (".*elbow_pitch_joint",),
  "wrist_yaw": (".*wrist_yaw_joint",),
  "wrist_roll": (".*wrist_roll_joint",),
  "wrist_pitch": (".*wrist_pitch_joint",),
}


def _make_actuator(joint_type: str) -> DcMotorActuatorCfg:
  motor = MOTOR_SPECS[VR_M3_1_27DOF_ACTUATOR_MOTORS[joint_type]]
  return DcMotorActuatorCfg(
    target_names_expr=_ACTUATOR_TARGETS[joint_type],
    armature=motor["armature"],
    effort_limit=motor["max_tau"],
    saturation_effort=motor["saturation_tau"],
    velocity_limit=motor["max_vel"],
    frictionloss=VR_M3_1_27DOF_STATIC_FRICTION[joint_type],
    viscous_damping=VR_M3_1_27DOF_VISCOUS_FRICTION[joint_type],
    damping=VR_M3_1_27DOF_DAMPING[joint_type],
    stiffness=VR_M3_1_27DOF_STIFFNESS[joint_type],
  )


VR_M3_1_27DOF_ACTUATORS = {
  joint_type: _make_actuator(joint_type) for joint_type in VR_M3_1_27DOF_ACTUATOR_MOTORS
}


##
# Keyframe config.
##


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.87),
  joint_pos={
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_pitch_joint": 0.4,
    ".*_ankle_pitch_joint": -0.2,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 1.2,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 1.2,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##


FOOT_GEOM_PATTERN = r"^(left|right)_ankle_roll_link_collision_([1-9]|10)$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={FOOT_GEOM_PATTERN: 3, ".*_collision": 3},
  priority={FOOT_GEOM_PATTERN: 1},
  friction={FOOT_GEOM_PATTERN: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT_GEOM_PATTERN: 3, ".*_collision": 3},
  priority={FOOT_GEOM_PATTERN: 1},
  friction={FOOT_GEOM_PATTERN: (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_GEOM_PATTERN,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


##
# Final config.
##


VR_M3_1_27DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(VR_M3_1_27DOF_ACTUATORS.values()),
  soft_joint_pos_limit_factor=0.9,
)


def get_vr_m3_1_27dof_robot_cfg() -> EntityCfg:
  """Get a fresh VR M3.1 27-DOF robot configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=VR_M3_1_27DOF_ARTICULATION,
  )


VR_M3_1_27DOF_ACTION_SCALE: dict[str, float] = {}

for actuator in VR_M3_1_27DOF_ARTICULATION.actuators:
  assert isinstance(actuator, DcMotorActuatorCfg)
  assert actuator.effort_limit is not None
  for name in actuator.target_names_expr:
    VR_M3_1_27DOF_ACTION_SCALE[name] = 0.25 * actuator.effort_limit / actuator.stiffness


VR_M3_1_ARTICULATION = VR_M3_1_27DOF_ARTICULATION


def get_vr_m3_1_robot_cfg() -> EntityCfg:
  """Get a fresh VR M3.1 robot configuration instance."""
  return get_vr_m3_1_27dof_robot_cfg()


VR_M3_1_ACTION_SCALE = VR_M3_1_27DOF_ACTION_SCALE


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_vr_m3_1_robot_cfg())
  viewer.launch(robot.spec.compile())
