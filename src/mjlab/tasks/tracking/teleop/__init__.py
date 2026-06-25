"""VR teleoperation for the VR-M3.1 tracking policy.

Public API:

* :class:`PoseSource` and its implementations (:class:`OpenXrPoseSource`,
  :class:`MockPoseSource`, :class:`ReplayPoseSource`) yield VR head/hand poses.
* :class:`TeleopRetargeter` maps those poses to the ``teleop_3point_*`` targets
  consumed by ``MotionCommand.set_live_teleop``.
"""

from mjlab.tasks.tracking.teleop.pose_source import (
  MockPoseSource,
  OpenXrPoseSource,
  PoseSource,
  ReplayPoseSource,
  TeleopFrame,
)
from mjlab.tasks.tracking.teleop.retarget import (
  TELEOP_BODY_NAMES,
  TeleopCalibration,
  TeleopRetargeter,
  TeleopTargets,
)

__all__ = [
  "TELEOP_BODY_NAMES",
  "MockPoseSource",
  "OpenXrPoseSource",
  "PoseSource",
  "ReplayPoseSource",
  "TeleopCalibration",
  "TeleopFrame",
  "TeleopRetargeter",
  "TeleopTargets",
]
