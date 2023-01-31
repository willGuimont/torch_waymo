"""
Based on https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/keypoint.proto
"""
from dataclasses import dataclass

from .utils import ReversibleIntEnum


@dataclass
class Vec2d:
    x: float
    y: float


@dataclass
class Vec3d:
    x: float
    y: float
    z: float


@dataclass
class KeypointVisibility:
    is_occluded: bool


@dataclass
class Keypoint2d:
    location_px: Vec2d
    visibility: KeypointVisibility


@dataclass
class Keypoint3d:
    visibility: KeypointVisibility
    location_px: Vec3d = None


class KeypointType(ReversibleIntEnum):
    KEYPOINT_TYPE_UNSPECIFIED = 0
    # Tip of nose.
    KEYPOINT_TYPE_NOSE = 1
    KEYPOINT_TYPE_LEFT_SHOULDER = 5
    KEYPOINT_TYPE_LEFT_ELBOW = 6
    KEYPOINT_TYPE_LEFT_WRIST = 7
    KEYPOINT_TYPE_LEFT_HIP = 8
    KEYPOINT_TYPE_LEFT_KNEE = 9
    KEYPOINT_TYPE_LEFT_ANKLE = 10
    KEYPOINT_TYPE_RIGHT_SHOULDER = 13
    KEYPOINT_TYPE_RIGHT_ELBOW = 14
    KEYPOINT_TYPE_RIGHT_WRIST = 15
    KEYPOINT_TYPE_RIGHT_HIP = 16
    KEYPOINT_TYPE_RIGHT_KNEE = 17
    KEYPOINT_TYPE_RIGHT_ANKLE = 18
    # Center of the forehead area.
    KEYPOINT_TYPE_FOREHEAD = 19
    # A point in the center of head - a point in the middle between two ears.
    # The nose and head center together create an imaginary line in the direction
    # that the person is looking (i.e. head orientation).
    KEYPOINT_TYPE_HEAD_CENTER = 20


@dataclass
class CameraKeypoint:
    type: KeypointType
    keypoint_2d: Keypoint2d = None
    keypoint_3d: Keypoint3d = None


@dataclass
class CameraKeypoints:
    keypoint: [CameraKeypoint]


@dataclass
class LaserKeypoint:
    type: KeypointType
    keypoint_3d: Keypoint3d


@dataclass
class LaserKeypoints:
    keypoint: [LaserKeypoint]
