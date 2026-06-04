"""VR H3.1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##

# MJCF and assets.

##


VR_H3_1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "vr_h3_1" / "xmls" / "vr_h3_1.xml"
)

VR_H3_1_28DOF_XML = VR_H3_1_XML

assert VR_H3_1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  del meshdir
  return {}


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(VR_H3_1_XML))


##

# Actuator config.

##

ARMATURE_EROB110H100 = 3.08632

ARMATURE_EROB90H100 = 0.95791

ARMATURE_EROB70H100 = 0.22674

ARMATURE_TD7080 = 0.5804369

ARMATURE_TD6070 = 0.3896782

ARMATURE_TD5060 = 0.1142512

ARMATURE_TD4052 = 0.0836482


# Joint types: legs (6) + waist (1) + arms (4) + wrists (3) = 14 actuator groups, 27 DOF total.

VR_H3_1_28DOF_ARMATURE = {
  "hip_pitch": 0.1404,
  "hip_roll": 0.1404,
  "hip_yaw": 0.02864,
  "knee_pitch": 0.1404,
  "ankle_pitch": 0.03006,
  "ankle_roll": 0.03006,
  "waist_yaw": ARMATURE_TD7080,
  "waist_roll": ARMATURE_TD7080,
  "shoulder_pitch": ARMATURE_TD6070,
  "shoulder_roll": ARMATURE_TD6070,
  "shoulder_yaw": ARMATURE_TD5060,
  "elbow_pitch": ARMATURE_TD5060,
  "wrist_yaw": ARMATURE_TD4052,
  "wrist_roll": ARMATURE_TD4052,
  "wrist_pitch": ARMATURE_TD4052,
}


VR_H3_1_28DOF_EFFORT_LIMITS = {
  "hip_pitch": 360.0,
  "hip_roll": 360.0,
  "hip_yaw": 130.0,
  "knee_pitch": 360.0,
  "ankle_pitch": 120.0,
  "ankle_roll": 120.0,
  "waist_yaw": 102.0,
  "waist_roll": 102.0,
  "shoulder_pitch": 66.0,
  "shoulder_roll": 66.0,
  "shoulder_yaw": 34.0,
  "elbow_pitch": 34.0,
  "wrist_yaw": 11.0,
  "wrist_roll": 11.0,
  "wrist_pitch": 11.0,
}


VR_H3_1_28DOF_SATURATION_EFFORTS = {
  "hip_pitch": 816.667,
  "hip_roll": 816.667,
  "hip_yaw": 389.875,
  "knee_pitch": 816.667,
  "ankle_pitch": 479.97,
  "ankle_roll": 479.97,
  "waist_yaw": 153.0,
  "waist_roll": 153.0,
  "shoulder_pitch": 99.0,
  "shoulder_roll": 99.0,
  "shoulder_yaw": 51.0,
  "elbow_pitch": 51.0,
  "wrist_yaw": 16.5,
  "wrist_roll": 16.5,
  "wrist_pitch": 16.5,
}


VR_H3_1_28DOF_VELOCITY_LIMITS = {
  "hip_pitch": 14.653,
  "hip_roll": 14.653,
  "hip_yaw": 31.4,
  "knee_pitch": 14.653,
  "ankle_pitch": 16.747,
  "ankle_roll": 16.747,
  "waist_yaw": 4.18,
  "waist_roll": 4.18,
  "shoulder_pitch": 4.29,
  "shoulder_roll": 4.29,
  "shoulder_yaw": 5.13,
  "elbow_pitch": 5.13,
  "wrist_yaw": 6.17,
  "wrist_roll": 6.17,
  "wrist_pitch": 6.17,
}


VR_H3_1_28DOF_STIFFNESS = {
  "hip_pitch": 150.0,
  "hip_roll": 150.0,
  "hip_yaw": 100.0,
  "knee_pitch": 200.0,
  "ankle_pitch": 200.0,
  "ankle_roll": 200.0,
  "waist_yaw": 367.0,
  "waist_roll": 367.0,
  "shoulder_pitch": 362.0,
  "shoulder_roll": 330.0,
  "shoulder_yaw": 278.0,
  "elbow_pitch": 278.0,
  "wrist_yaw": 292.0,
  "wrist_roll": 218.0,
  "wrist_pitch": 212.0,
}


VR_H3_1_28DOF_DAMPING = {
  "hip_pitch": 25.0,
  "hip_roll": 25.0,
  "hip_yaw": 4.0,
  "knee_pitch": 10.0,
  "ankle_pitch": 5.0,
  "ankle_roll": 5.0,
  "waist_yaw": 29.0,
  "waist_roll": 29.0,
  "shoulder_pitch": 14.0,
  "shoulder_roll": 13.0,
  "shoulder_yaw": 22.0,
  "elbow_pitch": 22.0,
  "wrist_yaw": 12.0,
  "wrist_roll": 9.0,
  "wrist_pitch": 8.0,
}


VR_H3_1_28DOF_STATIC_FRICTION = {
  "hip_pitch": 1.5,
  "hip_roll": 1.5,
  "hip_yaw": 0.8,
  "knee_pitch": 1.5,
  "ankle_pitch": 0.5,
  "ankle_roll": 0.5,
  "waist_yaw": 3.0,
  "waist_roll": 0.5,
  "shoulder_pitch": 3.0,
  "shoulder_roll": 3.0,
  "shoulder_yaw": 1.5,
  "elbow_pitch": 1.5,
  "wrist_yaw": 1.2,
  "wrist_roll": 0.8,
  "wrist_pitch": 1.2,
}


VR_H3_1_28DOF_DYNAMIC_FRICTION = {  # Coulomb / kinetic (avg of left & right Dc), N·m
  "hip_pitch": 16.217010,
  "hip_roll": 9.983655,
  "hip_yaw": 12.231030,
  "knee_pitch": 6.604992,
  "ankle_pitch": 3.235630,
  "ankle_roll": 1.138469,
  "waist_yaw": 0.5,
  "waist_roll": 0.5,
  "shoulder_pitch": 0.3,
  "shoulder_roll": 0.3,
  "shoulder_yaw": 0.3,
  "elbow_pitch": 0.3,
  "wrist_yaw": 0.3,
  "wrist_roll": 0.3,
  "wrist_pitch": 0.3,
}


VR_H3_1_28DOF_VISCOUS_FRICTION = {  # Viscous (avg of left & right Dv), N·m·s/rad
  "hip_pitch": 0.7,
  "hip_roll": 0.8,
  "hip_yaw": 0.2,
  "knee_pitch": 0.7,
  "ankle_pitch": 0.5,
  "ankle_roll": 0.1,
  "waist_yaw": 1.5,
  "waist_roll": 0.3,
  "shoulder_pitch": 3.0,
  "shoulder_roll": 3.0,
  "shoulder_yaw": 1.0,
  "elbow_pitch": 1.0,
  "wrist_yaw": 0.6,
  "wrist_roll": 0.6,
  "wrist_pitch": 0.6,
}


VR_H3_1_28DOF_ACTUATORS = {
  "hip_pitch": DcMotorActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["hip_pitch"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["hip_pitch"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["hip_pitch"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["hip_pitch"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["hip_pitch"],
    damping=VR_H3_1_28DOF_DAMPING["hip_pitch"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["hip_pitch"],
  ),
  "hip_roll": DcMotorActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["hip_roll"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["hip_roll"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["hip_roll"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["hip_roll"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["hip_roll"],
    damping=VR_H3_1_28DOF_DAMPING["hip_roll"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["hip_roll"],
  ),
  "hip_yaw": DcMotorActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["hip_yaw"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["hip_yaw"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["hip_yaw"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["hip_yaw"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["hip_yaw"],
    damping=VR_H3_1_28DOF_DAMPING["hip_yaw"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["hip_yaw"],
  ),
  "knee_pitch": DcMotorActuatorCfg(
    target_names_expr=(".*_knee_pitch_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["knee_pitch"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["knee_pitch"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["knee_pitch"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["knee_pitch"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["knee_pitch"],
    damping=VR_H3_1_28DOF_DAMPING["knee_pitch"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["knee_pitch"],
  ),
  "ankle_pitch": DcMotorActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["ankle_pitch"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["ankle_pitch"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["ankle_pitch"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["ankle_pitch"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["ankle_pitch"],
    damping=VR_H3_1_28DOF_DAMPING["ankle_pitch"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["ankle_pitch"],
  ),
  "ankle_roll": DcMotorActuatorCfg(
    target_names_expr=(".*_ankle_roll_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["ankle_roll"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["ankle_roll"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["ankle_roll"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["ankle_roll"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["ankle_roll"],
    damping=VR_H3_1_28DOF_DAMPING["ankle_roll"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["ankle_roll"],
  ),
  "waist_yaw": DcMotorActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["waist_yaw"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["waist_yaw"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["waist_yaw"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["waist_yaw"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["waist_yaw"],
    damping=VR_H3_1_28DOF_DAMPING["waist_yaw"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["waist_yaw"],
  ),
  "shoulder_pitch": DcMotorActuatorCfg(
    target_names_expr=(".*_shoulder_pitch_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["shoulder_pitch"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["shoulder_pitch"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["shoulder_pitch"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["shoulder_pitch"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["shoulder_pitch"],
    damping=VR_H3_1_28DOF_DAMPING["shoulder_pitch"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["shoulder_pitch"],
  ),
  "shoulder_roll": DcMotorActuatorCfg(
    target_names_expr=(".*_shoulder_roll_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["shoulder_roll"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["shoulder_roll"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["shoulder_roll"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["shoulder_roll"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["shoulder_roll"],
    damping=VR_H3_1_28DOF_DAMPING["shoulder_roll"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["shoulder_roll"],
  ),
  "shoulder_yaw": DcMotorActuatorCfg(
    target_names_expr=(".*_shoulder_yaw_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["shoulder_yaw"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["shoulder_yaw"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["shoulder_yaw"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["shoulder_yaw"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["shoulder_yaw"],
    damping=VR_H3_1_28DOF_DAMPING["shoulder_yaw"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["shoulder_yaw"],
  ),
  "elbow_pitch": DcMotorActuatorCfg(
    target_names_expr=(".*_elbow_pitch_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["elbow_pitch"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["elbow_pitch"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["elbow_pitch"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["elbow_pitch"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["elbow_pitch"],
    damping=VR_H3_1_28DOF_DAMPING["elbow_pitch"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["elbow_pitch"],
  ),
  "wrist_yaw": DcMotorActuatorCfg(
    target_names_expr=(".*_wrist_yaw_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["wrist_yaw"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["wrist_yaw"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["wrist_yaw"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["wrist_yaw"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["wrist_yaw"],
    damping=VR_H3_1_28DOF_DAMPING["wrist_yaw"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["wrist_yaw"],
  ),
  "wrist_roll": DcMotorActuatorCfg(
    target_names_expr=(".*_wrist_roll_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["wrist_roll"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["wrist_roll"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["wrist_roll"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["wrist_roll"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["wrist_roll"],
    damping=VR_H3_1_28DOF_DAMPING["wrist_roll"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["wrist_roll"],
  ),
  "wrist_pitch": DcMotorActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint",),
    armature=VR_H3_1_28DOF_ARMATURE["wrist_pitch"],
    effort_limit=VR_H3_1_28DOF_EFFORT_LIMITS["wrist_pitch"],
    saturation_effort=VR_H3_1_28DOF_SATURATION_EFFORTS["wrist_pitch"],
    velocity_limit=VR_H3_1_28DOF_VELOCITY_LIMITS["wrist_pitch"],
    frictionloss=VR_H3_1_28DOF_STATIC_FRICTION["wrist_pitch"],
    damping=VR_H3_1_28DOF_DAMPING["wrist_pitch"],
    stiffness=VR_H3_1_28DOF_STIFFNESS["wrist_pitch"],
  ),
}


##

# Keyframe config.

##


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.87),
  joint_pos={
    # --- Left leg (6) ---
    "left_hip_pitch_joint": -0.2,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_pitch_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    # --- Right leg (6) ---
    "right_hip_pitch_joint": -0.2,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_pitch_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    # --- Waist (1) ---
    "waist_yaw_joint": 0.0,
    # --- Left arm (7) ---
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 1.2,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    # --- Right arm (7) ---
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


# This enables all collisions, including self collisions.

# Self-collisions are given condim=1 while foot collisions

# are given condim=3.

# Note: Foot collision geoms are named left/right_ankle_roll_link_collision_*

FOOT_GEOM_PATTERN = r"^(left|right)_ankle_roll_link_collision_[1-9][0-9]*$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision_.*",),
  condim={FOOT_GEOM_PATTERN: 3, ".*_collision_.*": 1},
  priority={FOOT_GEOM_PATTERN: 1},
  friction={FOOT_GEOM_PATTERN: (0.6,)},
)


FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision_.*",),
  contype=0,
  conaffinity=1,
  condim={FOOT_GEOM_PATTERN: 3, ".*_collision_.*": 1},
  priority={FOOT_GEOM_PATTERN: 1},
  friction={FOOT_GEOM_PATTERN: (0.6,)},
)


# This disables all collisions except the feet.

# Feet get condim=3, all other geoms are disabled.

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


VR_H3_1_28DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(VR_H3_1_28DOF_ACTUATORS.values()),
  soft_joint_pos_limit_factor=0.9,
)


def get_vr_h3_1_28dof_robot_cfg() -> EntityCfg:
  """Get a fresh VR H3.1 27-DOF robot configuration."""

  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=VR_H3_1_28DOF_ARTICULATION,
  )


VR_H3_1_28DOF_ACTION_SCALE: dict[str, float] = {}

for a in VR_H3_1_28DOF_ARTICULATION.actuators:
  assert isinstance(a, DcMotorActuatorCfg)

  e = a.effort_limit

  s = a.stiffness

  names = a.target_names_expr

  assert e is not None

  for n in names:
    VR_H3_1_28DOF_ACTION_SCALE[n] = 0.25 * e / s


VR_H3_1_ARTICULATION = VR_H3_1_28DOF_ARTICULATION


def get_vr_h3_1_robot_cfg() -> EntityCfg:
  """Get a fresh VR H3.1 robot configuration instance."""
  return get_vr_h3_1_28dof_robot_cfg()


VR_H3_1_ACTION_SCALE = {
  ".*_hip_pitch_joint": 0.60,
  ".*_hip_roll_joint": 0.60,
  ".*_hip_yaw_joint": 0.325,
  ".*_knee_pitch_joint": 0.45,
  ".*_ankle_pitch_joint": 0.15,
  ".*_ankle_roll_joint": 0.15,
  "waist_yaw_joint": 0.12,
  ".*_shoulder_pitch_joint": 0.18,
  ".*_shoulder_roll_joint": 0.16,
  ".*_shoulder_yaw_joint": 0.16,
  ".*_elbow_pitch_joint": 0.16,
  ".*_wrist_yaw_joint": 0.12,
  ".*_wrist_roll_joint": 0.10,
  ".*_wrist_pitch_joint": 0.10,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_vr_h3_1_robot_cfg())

  viewer.launch(robot.spec.compile())
