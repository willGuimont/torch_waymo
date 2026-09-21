import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from torch_waymo import WaymoDataset
from torch_waymo.converters.geometry import range_image_to_point_cloud
from torch_waymo.converters.parquet import generate_parquet_cache
from torch_waymo.protocol.dataset_proto import CameraName, LaserName

IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
SEGMENT = "segment-1"
TIMESTAMP = 123456
FILENAME = f"{SEGMENT}.parquet"


def _write(root, component, rows):
    path = root / "training" / component
    path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path / FILENAME)


def _base(**values):
    return {"key.segment_context_name": SEGMENT, **values}


def _frame(**values):
    return _base(**{"key.frame_timestamp_micros": TIMESTAMP}, **values)


def _component(component, **values):
    return {f"[{component}].{key}": value for key, value in values.items()}


def _make_dataset(root):
    _write(
        root,
        "vehicle_pose",
        [_frame(**_component("VehiclePoseComponent", **{"world_from_vehicle.transform": IDENTITY}))],
    )
    _write(
        root,
        "stats",
        [
            _frame(
                **_component(
                    "StatsComponent",
                    time_of_day="Day",
                    location="SF",
                    weather="Sunny",
                    **{
                        "lidar_object_counts.types": [1],
                        "lidar_object_counts.counts": [1],
                        "camera_object_counts.types": [1],
                        "camera_object_counts.counts": [1],
                    },
                )
            )
        ],
    )
    _write(
        root,
        "camera_calibration",
        [
            _base(
                **{"key.camera_name": 1},
                **_component(
                    "CameraCalibrationComponent",
                    **{
                        "intrinsic.f_u": 1000.0,
                        "intrinsic.f_v": 1000.0,
                        "intrinsic.c_u": 960.0,
                        "intrinsic.c_v": 640.0,
                        "intrinsic.k1": 0.0,
                        "intrinsic.k2": 0.0,
                        "intrinsic.p1": 0.0,
                        "intrinsic.p2": 0.0,
                        "intrinsic.k3": 0.0,
                        "extrinsic.transform": IDENTITY,
                        "width": 1920,
                        "height": 1280,
                        "rolling_shutter_direction": 2,
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_calibration",
        [
            _base(
                **{"key.laser_name": 1},
                **_component(
                    "LiDARCalibrationComponent",
                    **{
                        "extrinsic.transform": IDENTITY,
                        "beam_inclination.min": 0.0,
                        "beam_inclination.max": 0.0,
                        "beam_inclination.values": [0.0],
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "camera_image",
        [
            _frame(
                **{"key.camera_name": 1},
                **_component(
                    "CameraImageComponent",
                    **{
                        "image": b"jpeg",
                        "pose.transform": IDENTITY,
                        "velocity.linear_velocity.x": 1.0,
                        "velocity.linear_velocity.y": 2.0,
                        "velocity.linear_velocity.z": 3.0,
                        "velocity.angular_velocity.x": 4.0,
                        "velocity.angular_velocity.y": 5.0,
                        "velocity.angular_velocity.z": 6.0,
                        "pose_timestamp": 1.25,
                        "rolling_shutter_params.shutter": 0.01,
                        "rolling_shutter_params.camera_trigger_time": 1.0,
                        "rolling_shutter_params.camera_readout_done_time": 1.1,
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "camera_segmentation",
        [
            _frame(
                **{"key.camera_name": 1},
                **_component(
                    "CameraSegmentationLabelComponent",
                    **{
                        "panoptic_label_divisor": 1000,
                        "panoptic_label": b"panoptic",
                        "instance_id_to_global_id_mapping.local_instance_ids": [7],
                        "instance_id_to_global_id_mapping.global_instance_ids": [70],
                        "instance_id_to_global_id_mapping.is_tracked": [True],
                        "sequence_id": "sequence",
                        "num_cameras_covered": b"coverage",
                    },
                ),
            )
        ],
    )
    range_return1 = [1.0, 0.1, 0.2, 0.0, 2.0, 0.3, 0.4, 1.0]
    range_return2 = [0.0] * 8
    _write(
        root,
        "lidar",
        [
            _frame(
                **{"key.laser_name": 1},
                **_component(
                    "LiDARComponent",
                    **{
                        "range_image_return1.values": range_return1,
                        "range_image_return1.shape": [1, 2, 4],
                        "range_image_return2.values": range_return2,
                        "range_image_return2.shape": [1, 2, 4],
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_camera_projection",
        [
            _frame(
                **{"key.laser_name": 1},
                **_component(
                    "LiDARCameraProjectionComponent",
                    **{
                        "range_image_return1.values": [1.0, 10.0, 20.0, 0.0, 0.0, 0.0] * 2,
                        "range_image_return1.shape": [1, 2, 6],
                        "range_image_return2.values": [0.0] * 12,
                        "range_image_return2.shape": [1, 2, 6],
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_pose",
        [
            _frame(
                **{"key.laser_name": 1},
                **_component(
                    "LiDARPoseComponent",
                    **{"range_image_return1.values": [0.0] * 12, "range_image_return1.shape": [1, 2, 6]},
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_segmentation",
        [
            _frame(
                **{"key.laser_name": 1},
                **_component(
                    "LiDARSegmentationLabelComponent",
                    **{
                        "range_image_return1.values": [1, 2, 3, 4],
                        "range_image_return1.shape": [1, 2, 2],
                        "range_image_return2.values": [0, 0, 0, 0],
                        "range_image_return2.shape": [1, 2, 2],
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_box",
        [
            _frame(
                **{"key.laser_object_id": "lidar-object"},
                **_component(
                    "LiDARBoxComponent",
                    **{
                        "box.center.x": 1.0,
                        "box.center.y": 2.0,
                        "box.center.z": 3.0,
                        "box.size.x": 4.0,
                        "box.size.y": 5.0,
                        "box.size.z": 6.0,
                        "box.heading": 0.5,
                        "type": 1,
                        "speed.x": 7.0,
                        "speed.y": 8.0,
                        "speed.z": 9.0,
                        "acceleration.x": 0.1,
                        "acceleration.y": 0.2,
                        "acceleration.z": 0.3,
                        "num_lidar_points_in_box": 12,
                        "num_top_lidar_points_in_box": 10,
                        "difficulty_level.detection": 1,
                        "difficulty_level.tracking": 2,
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_camera_synced_box",
        [
            _frame(
                **{"key.laser_object_id": "lidar-object"},
                **_component(
                    "LiDARCameraSyncedBoxComponent",
                    **{
                        "most_visible_camera_name": 1,
                        "camera_synced_box.center.x": 1.5,
                        "camera_synced_box.center.y": 2.0,
                        "camera_synced_box.center.z": 3.0,
                        "camera_synced_box.size.x": 4.0,
                        "camera_synced_box.size.y": 5.0,
                        "camera_synced_box.size.z": 6.0,
                        "camera_synced_box.heading": 0.5,
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "lidar_hkp",
        [
            _frame(
                **{"key.laser_object_id": "lidar-object"},
                **_component(
                    "LiDARHumanKeypointsComponent",
                    **{
                        "lidar_keypoints[*].type": [1],
                        "lidar_keypoints[*].keypoint_3d.location_m.x": [1.0],
                        "lidar_keypoints[*].keypoint_3d.location_m.y": [2.0],
                        "lidar_keypoints[*].keypoint_3d.location_m.z": [3.0],
                        "lidar_keypoints[*].keypoint_3d.visibility.is_occluded": [False],
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "camera_box",
        [
            _frame(
                **{"key.camera_name": 1, "key.camera_object_id": "camera-object"},
                **_component(
                    "CameraBoxComponent",
                    **{
                        "box.center.x": 10.0,
                        "box.center.y": 20.0,
                        "box.size.x": 30.0,
                        "box.size.y": 40.0,
                        "type": 1,
                        "difficulty_level.detection": 1,
                        "difficulty_level.tracking": 2,
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "camera_to_lidar_box_association",
        [
            _frame(
                **{
                    "key.camera_name": 1,
                    "key.camera_object_id": "camera-object",
                    "key.laser_object_id": "lidar-object",
                }
            )
        ],
    )
    _write(
        root,
        "camera_hkp",
        [
            _frame(
                **{"key.camera_name": 1, "key.camera_object_id": "camera-object"},
                **_component(
                    "CameraHumanKeypointsComponent",
                    **{
                        "camera_keypoints[*].type": [1],
                        "camera_keypoints[*].keypoint_2d.location_px.x": [11.0],
                        "camera_keypoints[*].keypoint_2d.location_px.y": [22.0],
                        "camera_keypoints[*].keypoint_2d.visibility.is_occluded": [False],
                        "camera_keypoints[*].keypoint_3d.location_m.x": [1.0],
                        "camera_keypoints[*].keypoint_3d.location_m.y": [2.0],
                        "camera_keypoints[*].keypoint_3d.location_m.z": [3.0],
                        "camera_keypoints[*].keypoint_3d.visibility.is_occluded": [True],
                    },
                ),
            )
        ],
    )
    _write(
        root,
        "projected_lidar_box",
        [
            _frame(
                **{"key.camera_name": 1, "key.laser_object_id": "lidar-object"},
                **_component(
                    "ProjectedLiDARBoxComponent",
                    **{
                        "box.center.x": 10.0,
                        "box.center.y": 20.0,
                        "box.size.x": 30.0,
                        "box.size.y": 40.0,
                        "type": 1,
                    },
                ),
            )
        ],
    )


def test_range_image_geometry_matches_waymo_azimuth_order():
    ranges = np.ones((1, 4, 4), dtype=np.float32)
    points = range_image_to_point_cloud(ranges, np.eye(4), np.array([0.0]))
    root_half = np.sqrt(0.5)
    expected = np.array(
        [
            [-root_half, root_half, 0.0],
            [root_half, root_half, 0.0],
            [root_half, -root_half, 0.0],
            [-root_half, -root_half, 0.0],
        ]
    )
    np.testing.assert_allclose(points, expected, atol=1e-6)


def test_converts_full_parquet_frame(tmp_path):
    _make_dataset(tmp_path)

    generate_parquet_cache(tmp_path, "training")

    dataset = WaymoDataset(tmp_path / "converted", "training")
    frame = dataset[0]
    assert (tmp_path / "converted" / "training" / "0.pkl.gz").is_file()
    assert frame.context.name == SEGMENT
    assert frame.context.stats.location == "SF"
    assert frame.timestamp_micros == TIMESTAMP
    assert frame.images[0].name == CameraName.FRONT
    assert frame.images[0].image == b"jpeg"
    assert frame.images[0].camera_segmentation_label.num_cameras_covered == b"coverage"
    assert frame.lasers[0].name == LaserName.TOP
    assert frame.lasers[0].ri_return1.values.shape == (1, 2, 4)
    assert frame.lasers[0].ri_return1.camera_projection.shape == (1, 2, 6)
    assert frame.lasers[0].ri_return1.segmentation_label.shape == (1, 2, 2)
    assert frame.points[0].shape == (2, 3)
    assert frame.laser_labels[0].camera_synced_box.center_x == 1.5
    assert frame.laser_labels[0].laser_keypoints.keypoint[0].keypoint_3d.location_m.z == 3.0
    assert frame.camera_labels[0].labels[0].association.laser_object_id == "lidar-object"
    assert frame.camera_labels[0].labels[0].camera_keypoints.keypoint[0].keypoint_2d.location_px.x == 11.0
    assert frame.projected_lidar_labels[0].labels[0].id == "lidar-object_FRONT"
    assert frame.no_label_zones == []
    with (tmp_path / "converted" / "training" / "len.pkl").open("rb") as file:
        assert pickle.load(file) == [1]


def test_converts_simplified_parquet_frame_without_camera_components(tmp_path):
    _make_dataset(tmp_path)
    for component in (
        "camera_calibration",
        "camera_image",
        "camera_segmentation",
        "camera_box",
        "camera_to_lidar_box_association",
        "camera_hkp",
        "projected_lidar_box",
    ):
        path = tmp_path / "training" / component / FILENAME
        path.unlink()

    generate_parquet_cache(tmp_path, "training", simplified=True)

    frame = WaymoDataset(tmp_path / "converted_simplified", "training")[0]
    assert frame.context.camera_calibrations == []
    assert frame.points[0].shape == (2, 3)
    assert frame.laser_labels[0].id == "lidar-object"
