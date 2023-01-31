"""
Based on https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto#L145
"""
import typing
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

from .label_proto import Type, Label, Polygon2dProto
from .utils import ReversibleIntEnum

T = typing.TypeVar('T')


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


def from_data(cls: typing.Type[T], data) -> T:
    from waymo_open_dataset import dataset_pb2
    from google.protobuf.pyext._message import RepeatedCompositeContainer, RepeatedScalarContainer

    if isinstance(data, dataset_pb2.Transform):
        return np.array(data.transform).reshape((4, 4))
    if isinstance(data, list) or isinstance(data, RepeatedCompositeContainer) or isinstance(data,
                                                                                            RepeatedScalarContainer):
        return [from_data(cls[0], d) for d in data]
    if not is_dataclass(cls):
        return data

    field_names = [f.name for f in fields(cls)]
    field_types = {f.name: f.type for f in fields(cls)}

    attributes = dict()
    for name in dir(data):
        if name in field_names:
            field_type = field_types[name]
            field_data = getattr(data, name)
            attributes[name] = from_data(field_type, field_data)

    return cls(**attributes)


Transform = np.ndarray


class CameraName(ReversibleIntEnum):
    UNKNOWN = 0
    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5


class LaserName(ReversibleIntEnum):
    UNKNOWN = 0
    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5


@dataclass
class Velocity:
    # velocity in m/s
    v_x: float
    v_y: float
    v_z: float

    # angular velocity in rad/s
    w_x: float
    w_y: float
    w_z: float


class RollingShutterReadOutDirection(ReversibleIntEnum):
    UNKNOWN = 0
    TOP_TO_BOTTOM = 1
    LEFT_TO_RIGHT = 2
    BOTTOM_TO_TOP = 3
    RIGHT_TO_LEFT = 4
    GLOBAL_SHUTTER = 5


@dataclass
class CameraCalibration:
    name: CameraName
    # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}], follows the same definition as OpenCV
    intrinsic: [float]
    extrinsic: Transform
    width: int
    height: int
    rolling_shutter_direction: RollingShutterReadOutDirection


@dataclass
class LaserCalibration:
    name: LaserName
    beam_inclinations: [float]
    beam_inclination_min: float
    beam_inclination_max: float
    extrinsic: Transform


@dataclass
class ObjectCount:
    type: Type
    count: int


@dataclass
class Stats:
    laser_object_counts: [ObjectCount]
    camera_object_counts: [ObjectCount]
    time_of_day: str
    location: str
    weather: str


@dataclass
class Context:
    name: str
    camera_calibrations: [CameraCalibration]
    laser_calibrations: [LaserCalibration]
    stats: Stats


@dataclass
class RangeImage:
    """
    *_compressed are compressed using Zlib (val = ZlibDecompress(range_image_compressed))
    """
    range_image_compressed: np.ndarray
    camera_projection_compressed: np.ndarray
    range_image_pose_compressed: np.ndarray
    range_image_flow_compressed: np.ndarray
    segmentation_label_compressed: np.ndarray


@dataclass
class InstanceIDToGlobalIDMapping:
    local_instance_id: int
    global_instance_id: int
    is_tracked: bool


@dataclass
class CameraSegmentationLabel:
    panoptic_label_divisor: int
    panoptic_label: np.ndarray
    instance_id_to_global_id_mapping: [InstanceIDToGlobalIDMapping]
    sequence_id: str


@dataclass
class CameraImage:
    name: CameraName
    image: np.ndarray
    pose: Transform
    velocity: Velocity
    pose_timestamp: float
    shutter: float
    camera_trigger_time: float
    camera_readout_done_time: float
    camera_segmentation_label: CameraSegmentationLabel


@dataclass
class CameraLabels:
    name: CameraName
    labels: [Label]


@dataclass
class Laser:
    name: LaserName
    ri_return1: RangeImage
    ri_return2: RangeImage


@dataclass
class Frame:
    context: Context
    timestamp_micros: int
    pose: Transform
    images: [CameraImage]
    lasers: [Laser]
    laser_labels: [Label]
    projected_lidar_labels: [CameraLabels]
    camera_labels: [CameraLabels]
    no_label_zones: [Polygon2dProto]
