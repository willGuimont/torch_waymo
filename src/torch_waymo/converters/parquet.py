"""TensorFlow-free converter for the Waymo v2 modular Parquet format."""

import pathlib
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import tqdm

from torch_waymo.converters.common import output_split_path, write_frame, write_sequence_lengths
from torch_waymo.converters.geometry import beam_inclinations, range_image_to_point_cloud
from torch_waymo.dataset import SimplifiedFrame
from torch_waymo.protocol.dataset_proto import (
    CameraCalibration,
    CameraImage,
    CameraLabels,
    CameraName,
    CameraSegmentationLabel,
    Context,
    Frame,
    InstanceIDToGlobalIDMapping,
    Laser,
    LaserCalibration,
    LaserName,
    ObjectCount,
    RangeImage,
    RollingShutterReadOutDirection,
    Stats,
    Velocity,
)
from torch_waymo.protocol.keypoint_proto import (
    CameraKeypoint,
    CameraKeypoints,
    Keypoint2d,
    Keypoint3d,
    KeypointType,
    KeypointVisibility,
    LaserKeypoint,
    LaserKeypoints,
    Vec2d,
    Vec3d,
)
from torch_waymo.protocol.label_proto import (
    Association,
    Box,
    DifficultyLevel,
    Label,
    Metadata,
    Type,
)

_TIMESTAMP = "key.frame_timestamp_micros"
_CAMERA = "key.camera_name"
_LASER = "key.laser_name"
_CAMERA_OBJECT = "key.camera_object_id"
_LASER_OBJECT = "key.laser_object_id"


def _column(component: str, field: str) -> str:
    return f"[{component}].{field}"


def _value(row: dict, component: str, field: str, default=None):
    value = row.get(_column(component, field), default)
    return default if value is None else value


def _read_rows(split_path: pathlib.Path, component: str, filename: str, *, required: bool = False) -> list[dict]:
    path = split_path / component / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required Waymo v2 component: {path}")
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Parquet conversion requires the 'waymo' extra: uv sync --extra waymo") from error
    return pq.read_table(path, memory_map=True).to_pylist()


def _group_timestamp(rows: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row[_TIMESTAMP])].append(row)
    return grouped


def _index(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _reshape(row: dict | None, component: str, field: str, *, dtype=None) -> np.ndarray | None:
    if row is None:
        return None
    values = row.get(_column(component, f"{field}.values"))
    shape = row.get(_column(component, f"{field}.shape"))
    if values is None or shape is None:
        return None
    return np.asarray(values, dtype=dtype).reshape(shape)


def _transform(row: dict, component: str, field: str) -> np.ndarray:
    return np.asarray(_value(row, component, f"{field}.transform"), dtype=np.float64).reshape(4, 4)


def _box3d(row: dict, component: str, field: str = "box") -> Box:
    return Box(
        center_x=float(_value(row, component, f"{field}.center.x", 0.0)),
        center_y=float(_value(row, component, f"{field}.center.y", 0.0)),
        center_z=float(_value(row, component, f"{field}.center.z", 0.0)),
        length=float(_value(row, component, f"{field}.size.x", 0.0)),
        width=float(_value(row, component, f"{field}.size.y", 0.0)),
        height=float(_value(row, component, f"{field}.size.z", 0.0)),
        heading=float(_value(row, component, f"{field}.heading", 0.0)),
    )


def _box2d(row: dict, component: str) -> Box:
    return Box(
        center_x=float(_value(row, component, "box.center.x", 0.0)),
        center_y=float(_value(row, component, "box.center.y", 0.0)),
        center_z=0.0,
        length=float(_value(row, component, "box.size.x", 0.0)),
        width=float(_value(row, component, "box.size.y", 0.0)),
        height=0.0,
        heading=0.0,
    )


def _difficulty(row: dict, component: str) -> tuple[DifficultyLevel, DifficultyLevel]:
    detection = int(_value(row, component, "difficulty_level.detection", 0))
    tracking = int(_value(row, component, "difficulty_level.tracking", 0))
    return DifficultyLevel(detection), DifficultyLevel(tracking)


def _metadata(row: dict, component: str) -> Metadata:
    return Metadata(
        speed_x=float(_value(row, component, "speed.x", 0.0)),
        speed_y=float(_value(row, component, "speed.y", 0.0)),
        speed_z=float(_value(row, component, "speed.z", 0.0)),
        accel_x=float(_value(row, component, "acceleration.x", 0.0)),
        accel_y=float(_value(row, component, "acceleration.y", 0.0)),
        accel_z=float(_value(row, component, "acceleration.z", 0.0)),
    )


def _camera_keypoints(row: dict | None) -> CameraKeypoints:
    if row is None:
        return CameraKeypoints()
    component = "CameraHumanKeypointsComponent"
    prefix = "camera_keypoints[*]"
    types = _value(row, component, f"{prefix}.type", [])
    x2 = _value(row, component, f"{prefix}.keypoint_2d.location_px.x", [])
    y2 = _value(row, component, f"{prefix}.keypoint_2d.location_px.y", [])
    o2 = _value(row, component, f"{prefix}.keypoint_2d.visibility.is_occluded", [])
    x3 = _value(row, component, f"{prefix}.keypoint_3d.location_m.x", [])
    y3 = _value(row, component, f"{prefix}.keypoint_3d.location_m.y", [])
    z3 = _value(row, component, f"{prefix}.keypoint_3d.location_m.z", [])
    o3 = _value(row, component, f"{prefix}.keypoint_3d.visibility.is_occluded", [])
    keypoints = []
    for index, keypoint_type in enumerate(types):
        point2d = None
        if index < len(x2) and x2[index] is not None:
            point2d = Keypoint2d(Vec2d(float(x2[index]), float(y2[index])), KeypointVisibility(bool(o2[index])))
        point3d = None
        if index < len(x3) and x3[index] is not None:
            point3d = Keypoint3d(
                KeypointVisibility(bool(o3[index])),
                Vec3d(float(x3[index]), float(y3[index]), float(z3[index])),
            )
        keypoints.append(CameraKeypoint(KeypointType(int(keypoint_type)), point2d, point3d))
    return CameraKeypoints(keypoints)


def _lidar_keypoints(row: dict | None) -> LaserKeypoints:
    if row is None:
        return LaserKeypoints()
    component = "LiDARHumanKeypointsComponent"
    prefix = "lidar_keypoints[*]"
    types = _value(row, component, f"{prefix}.type", [])
    xs = _value(row, component, f"{prefix}.keypoint_3d.location_m.x", [])
    ys = _value(row, component, f"{prefix}.keypoint_3d.location_m.y", [])
    zs = _value(row, component, f"{prefix}.keypoint_3d.location_m.z", [])
    occluded = _value(row, component, f"{prefix}.keypoint_3d.visibility.is_occluded", [])
    return LaserKeypoints(
        [
            LaserKeypoint(
                KeypointType(int(keypoint_type)),
                Keypoint3d(
                    KeypointVisibility(bool(occluded[index])),
                    Vec3d(float(xs[index]), float(ys[index]), float(zs[index])),
                ),
            )
            for index, keypoint_type in enumerate(types)
        ]
    )


def _object_counts(row: dict, field: str) -> list[ObjectCount]:
    component = "StatsComponent"
    types = _value(row, component, f"{field}.types", [])
    counts = _value(row, component, f"{field}.counts", [])
    return [ObjectCount(Type(int(object_type)), int(count)) for object_type, count in zip(types, counts)]


def _context(
    segment_name: str,
    stats_row: dict,
    camera_calibrations: list[CameraCalibration],
    lidar_calibrations: list[LaserCalibration],
) -> Context:
    return Context(
        name=segment_name,
        camera_calibrations=camera_calibrations,
        laser_calibrations=lidar_calibrations,
        stats=Stats(
            laser_object_counts=_object_counts(stats_row, "lidar_object_counts"),
            camera_object_counts=_object_counts(stats_row, "camera_object_counts"),
            time_of_day=str(_value(stats_row, "StatsComponent", "time_of_day", "")),
            location=str(_value(stats_row, "StatsComponent", "location", "")),
            weather=str(_value(stats_row, "StatsComponent", "weather", "")),
        ),
    )


def _camera_calibrations(rows: list[dict]) -> list[CameraCalibration]:
    component = "CameraCalibrationComponent"
    intrinsic_fields = ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3")
    return [
        CameraCalibration(
            name=CameraName(int(row[_CAMERA])),
            intrinsic=[float(_value(row, component, f"intrinsic.{field}", 0.0)) for field in intrinsic_fields],
            extrinsic=_transform(row, component, "extrinsic"),
            width=int(_value(row, component, "width", 0)),
            height=int(_value(row, component, "height", 0)),
            rolling_shutter_direction=RollingShutterReadOutDirection(
                int(_value(row, component, "rolling_shutter_direction", 0))
            ),
        )
        for row in sorted(rows, key=lambda item: item[_CAMERA])
    ]


def _lidar_calibrations(rows: list[dict]) -> list[LaserCalibration]:
    component = "LiDARCalibrationComponent"
    return [
        LaserCalibration(
            name=LaserName(int(row[_LASER])),
            beam_inclinations=list(_value(row, component, "beam_inclination.values", [])),
            beam_inclination_min=float(_value(row, component, "beam_inclination.min", 0.0)),
            beam_inclination_max=float(_value(row, component, "beam_inclination.max", 0.0)),
            extrinsic=_transform(row, component, "extrinsic"),
        )
        for row in sorted(rows, key=lambda item: item[_LASER])
    ]


def _segmentation(row: dict | None) -> CameraSegmentationLabel:
    if row is None:
        return CameraSegmentationLabel(0, b"", [], "", b"")
    component = "CameraSegmentationLabelComponent"
    locals_ = _value(row, component, "instance_id_to_global_id_mapping.local_instance_ids", [])
    globals_ = _value(row, component, "instance_id_to_global_id_mapping.global_instance_ids", [])
    tracked = _value(row, component, "instance_id_to_global_id_mapping.is_tracked", [])
    mappings = [
        InstanceIDToGlobalIDMapping(int(local), int(global_), bool(is_tracked))
        for local, global_, is_tracked in zip(locals_, globals_, tracked)
    ]
    return CameraSegmentationLabel(
        panoptic_label_divisor=int(_value(row, component, "panoptic_label_divisor", 0)),
        panoptic_label=bytes(_value(row, component, "panoptic_label", b"")),
        instance_id_to_global_id_mapping=mappings,
        sequence_id=str(_value(row, component, "sequence_id", "")),
        num_cameras_covered=bytes(_value(row, component, "num_cameras_covered", b"")),
    )


def _images(rows: list[dict], segmentation: dict[tuple, dict]) -> list[CameraImage]:
    component = "CameraImageComponent"
    images = []
    for row in sorted(rows, key=lambda item: item[_CAMERA]):
        camera_name = int(row[_CAMERA])
        images.append(
            CameraImage(
                name=CameraName(camera_name),
                image=bytes(_value(row, component, "image", b"")),
                pose=_transform(row, component, "pose"),
                velocity=Velocity(
                    v_x=float(_value(row, component, "velocity.linear_velocity.x", 0.0)),
                    v_y=float(_value(row, component, "velocity.linear_velocity.y", 0.0)),
                    v_z=float(_value(row, component, "velocity.linear_velocity.z", 0.0)),
                    w_x=float(_value(row, component, "velocity.angular_velocity.x", 0.0)),
                    w_y=float(_value(row, component, "velocity.angular_velocity.y", 0.0)),
                    w_z=float(_value(row, component, "velocity.angular_velocity.z", 0.0)),
                ),
                pose_timestamp=float(_value(row, component, "pose_timestamp", 0.0)),
                shutter=float(_value(row, component, "rolling_shutter_params.shutter", 0.0)),
                camera_trigger_time=float(_value(row, component, "rolling_shutter_params.camera_trigger_time", 0.0)),
                camera_readout_done_time=float(
                    _value(row, component, "rolling_shutter_params.camera_readout_done_time", 0.0)
                ),
                camera_segmentation_label=_segmentation(segmentation.get((camera_name,))),
            )
        )
    return images


def _range_image(
    lidar_row: dict,
    projection_row: dict | None,
    pose_row: dict | None,
    segmentation_row: dict | None,
    return_number: int,
) -> RangeImage:
    field = f"range_image_return{return_number}"
    return RangeImage(
        values=_reshape(lidar_row, "LiDARComponent", field, dtype=np.float32),
        camera_projection=_reshape(projection_row, "LiDARCameraProjectionComponent", field, dtype=np.float32),
        pose=(
            _reshape(pose_row, "LiDARPoseComponent", "range_image_return1", dtype=np.float32)
            if return_number == 1
            else None
        ),
        segmentation_label=_reshape(segmentation_row, "LiDARSegmentationLabelComponent", field, dtype=np.int32),
    )


def _lasers(
    rows: list[dict],
    projections: dict[tuple, dict],
    poses: dict[tuple, dict],
    segmentations: dict[tuple, dict],
) -> list[Laser]:
    lasers = []
    for row in sorted(rows, key=lambda item: item[_LASER]):
        laser_name = int(row[_LASER])
        key = (laser_name,)
        lasers.append(
            Laser(
                name=LaserName(laser_name),
                ri_return1=_range_image(row, projections.get(key), poses.get(key), segmentations.get(key), 1),
                ri_return2=_range_image(row, projections.get(key), poses.get(key), segmentations.get(key), 2),
            )
        )
    return lasers


def _laser_labels(
    rows: list[dict],
    synced: dict[tuple, dict],
    keypoints: dict[tuple, dict],
) -> list[Label]:
    component = "LiDARBoxComponent"
    labels = []
    for row in rows:
        object_id = str(row[_LASER_OBJECT])
        detection, tracking = _difficulty(row, component)
        synced_row = synced.get((object_id,))
        labels.append(
            Label(
                box=_box3d(row, component),
                metadata=_metadata(row, component),
                type=Type(int(_value(row, component, "type", 0))),
                id=object_id,
                detection_difficulty_level=detection,
                tracking_difficulty_level=tracking,
                num_lidar_points_in_box=int(_value(row, component, "num_lidar_points_in_box", 0)),
                num_top_lidar_points_in_box=int(_value(row, component, "num_top_lidar_points_in_box", 0)),
                laser_keypoints=_lidar_keypoints(keypoints.get((object_id,))),
                most_visible_camera_name=(
                    CameraName(
                        int(_value(synced_row, "LiDARCameraSyncedBoxComponent", "most_visible_camera_name"))
                    ).name
                    if synced_row is not None
                    else ""
                ),
                camera_synced_box=(
                    _box3d(synced_row, "LiDARCameraSyncedBoxComponent", "camera_synced_box")
                    if synced_row is not None
                    else None
                ),
            )
        )
    return labels


def _camera_labels(
    rows: list[dict],
    associations: dict[tuple, dict],
    keypoints: dict[tuple, dict],
) -> list[CameraLabels]:
    grouped = defaultdict(list)
    component = "CameraBoxComponent"
    for row in rows:
        camera_name = int(row[_CAMERA])
        object_id = str(row[_CAMERA_OBJECT])
        detection, tracking = _difficulty(row, component)
        association_row = associations.get((camera_name, object_id))
        grouped[camera_name].append(
            Label(
                box=_box2d(row, component),
                metadata=Metadata(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                type=Type(int(_value(row, component, "type", 0))),
                id=object_id,
                detection_difficulty_level=detection,
                tracking_difficulty_level=tracking,
                num_lidar_points_in_box=0,
                num_top_lidar_points_in_box=0,
                camera_keypoints=_camera_keypoints(keypoints.get((camera_name, object_id))),
                association=(Association(str(association_row[_LASER_OBJECT])) if association_row is not None else None),
            )
        )
    return [CameraLabels(CameraName(name), grouped[name]) for name in sorted(grouped)]


def _projected_labels(rows: list[dict]) -> list[CameraLabels]:
    grouped = defaultdict(list)
    component = "ProjectedLiDARBoxComponent"
    for row in rows:
        camera_name = int(row[_CAMERA])
        object_id = str(row[_LASER_OBJECT])
        grouped[camera_name].append(
            Label(
                box=_box2d(row, component),
                metadata=Metadata(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                type=Type(int(_value(row, component, "type", 0))),
                id=f"{object_id}_{CameraName(camera_name).name}",
                detection_difficulty_level=DifficultyLevel.UNKNOWN,
                tracking_difficulty_level=DifficultyLevel.UNKNOWN,
                num_lidar_points_in_box=0,
                num_top_lidar_points_in_box=0,
            )
        )
    return [CameraLabels(CameraName(name), grouped[name]) for name in sorted(grouped)]


def _points(lasers: list[Laser], calibrations: list[LaserCalibration], frame_pose: np.ndarray) -> list[np.ndarray]:
    calibration_by_name = {calibration.name: calibration for calibration in calibrations}
    points = []
    for laser in lasers:
        range_image = laser.ri_return1.values
        calibration = calibration_by_name[laser.name]
        if range_image is None:
            points.append(np.empty((0, 3), dtype=np.float32))
            continue
        inclinations = beam_inclinations(
            range_image.shape[0],
            calibration.beam_inclinations,
            calibration.beam_inclination_min,
            calibration.beam_inclination_max,
        )
        pixel_pose = laser.ri_return1.pose if laser.name == LaserName.TOP else None
        points.append(
            range_image_to_point_cloud(
                range_image,
                calibration.extrinsic,
                inclinations,
                pixel_pose=pixel_pose,
                frame_pose=frame_pose if pixel_pose is not None else None,
            )
        )
    return points


def _convert_sequence(
    split_path: pathlib.Path, pose_path: pathlib.Path, simplified: bool
) -> Iterator[Frame | SimplifiedFrame]:
    filename = pose_path.name
    vehicle_pose_rows = _read_rows(split_path, "vehicle_pose", filename, required=True)
    stats_rows = _group_timestamp(_read_rows(split_path, "stats", filename, required=True))
    camera_calibrations = _camera_calibrations(
        _read_rows(split_path, "camera_calibration", filename, required=not simplified)
    )
    lidar_calibrations = _lidar_calibrations(_read_rows(split_path, "lidar_calibration", filename, required=True))

    required_components = {"lidar"}
    if not simplified:
        required_components.add("camera_image")
    per_timestamp = {
        component: _group_timestamp(
            _read_rows(split_path, component, filename, required=component in required_components)
        )
        for component in (
            "camera_image",
            "camera_segmentation",
            "camera_box",
            "camera_to_lidar_box_association",
            "camera_hkp",
            "lidar",
            "lidar_camera_projection",
            "lidar_pose",
            "lidar_segmentation",
            "lidar_box",
            "lidar_camera_synced_box",
            "lidar_hkp",
            "projected_lidar_box",
        )
    }

    for pose_row in sorted(vehicle_pose_rows, key=lambda row: row[_TIMESTAMP]):
        timestamp = int(pose_row[_TIMESTAMP])
        stats_for_frame = stats_rows.get(timestamp)
        if not stats_for_frame:
            raise ValueError(f"Missing stats for frame {timestamp} in {filename}")
        frame_pose = _transform(pose_row, "VehiclePoseComponent", "world_from_vehicle")

        def rows(component: str) -> list[dict]:
            return per_timestamp[component].get(timestamp, [])

        segment_name = str(pose_row["key.segment_context_name"])
        context = _context(segment_name, stats_for_frame[0], camera_calibrations, lidar_calibrations)
        laser_rows = rows("lidar")
        lasers = _lasers(
            laser_rows,
            _index(rows("lidar_camera_projection"), _LASER),
            _index(rows("lidar_pose"), _LASER),
            _index(rows("lidar_segmentation"), _LASER),
        )
        points = _points(lasers, lidar_calibrations, frame_pose)
        laser_labels = _laser_labels(
            rows("lidar_box"),
            _index(rows("lidar_camera_synced_box"), _LASER_OBJECT),
            _index(rows("lidar_hkp"), _LASER_OBJECT),
        )
        if simplified:
            yield SimplifiedFrame(context, timestamp, frame_pose, laser_labels, [], points)
            continue
        yield Frame(
            context=context,
            timestamp_micros=timestamp,
            pose=frame_pose,
            images=_images(
                rows("camera_image"),
                _index(rows("camera_segmentation"), _CAMERA),
            ),
            lasers=lasers,
            laser_labels=laser_labels,
            projected_lidar_labels=_projected_labels(rows("projected_lidar_box")),
            camera_labels=_camera_labels(
                rows("camera_box"),
                _index(rows("camera_to_lidar_box_association"), _CAMERA, _CAMERA_OBJECT),
                _index(rows("camera_hkp"), _CAMERA, _CAMERA_OBJECT),
            ),
            no_label_zones=[],
            points=points,
        )


def generate_parquet_cache(root_path: pathlib.Path, split: str, simplified: bool = False) -> None:
    """Convert one split of the Waymo v2 modular dataset into frame pickles."""
    split_path = root_path / split
    pose_path = split_path / "vehicle_pose"
    sequence_paths = sorted(pose_path.glob("*.parquet"))
    if not sequence_paths:
        raise FileNotFoundError(f"No Waymo v2 vehicle-pose Parquet files found in {pose_path}")

    output_path = output_split_path(root_path, split, simplified)
    frame_index = 0
    sequence_lengths = []
    for sequence_path in tqdm.tqdm(sequence_paths, desc=f"Converting {split}"):
        sequence_length = 0
        for frame in _convert_sequence(split_path, sequence_path, simplified):
            write_frame(output_path, frame_index, frame)
            frame_index += 1
            sequence_length += 1
        sequence_lengths.append(sequence_length)
    write_sequence_lengths(output_path, sequence_lengths)
