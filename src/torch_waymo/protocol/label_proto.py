"""
Based on https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
"""
from dataclasses import dataclass

from .keypoint_proto import LaserKeypoints, CameraKeypoints
from .utils import ReversibleIntEnum


class BoxType(ReversibleIntEnum):
    TYPE_UNKNOWN = 0
    # 7-DOF 3D (a.k.a upright 3D box).
    TYPE_3D = 1
    # 5-DOF 2D. Mostly used for laser top down representation.
    TYPE_2D = 2
    # Axis aligned 2D. Mostly used for image.
    TYPE_AA_2D = 3


@dataclass
class Box:
    center_x: float
    center_y: float
    center_z: float

    length: float
    width: float
    height: float

    # angle in rad (-pi, pi(, rotate +x to the box front face
    heading: float


@dataclass
class Metadata:
    speed_x: float
    speed_y: float
    speed_z: float

    accel_x: float
    accel_y: float
    accel_z: float


class Type(ReversibleIntEnum):
    TYPE_UNKNOWN = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_SIGN = 3
    TYPE_CYCLIST = 4


class DifficultyLevel(ReversibleIntEnum):
    UNKNOWN = 0
    LEVEL_1 = 1
    LEVEL_2 = 2


@dataclass
class Association:
    laser_object_id: str


@dataclass
class Label:
    box: Box
    metadata: Metadata
    type: Type
    id: str
    detection_difficulty_level: DifficultyLevel
    tracking_difficulty_level: DifficultyLevel
    num_lidar_points_in_box: int
    num_top_lidar_points_in_box: int
    laser_keypoints: LaserKeypoints
    camera_keypoints: CameraKeypoints
    association: Association
    most_visible_camera_name: str
    camera_synced_box: Box


@dataclass
class Polygon2dProto:
    x: [float]
    y: [float]
    id: str
