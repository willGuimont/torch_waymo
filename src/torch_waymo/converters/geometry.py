"""NumPy implementation of Waymo range-image geometry."""

import numpy as np


def rotation_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll) for arrays of Euler angles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    result = np.empty(roll.shape + (3, 3), dtype=np.result_type(roll, pitch, yaw))
    result[..., 0, 0] = cy * cp
    result[..., 0, 1] = cy * sp * sr - sy * cr
    result[..., 0, 2] = cy * sp * cr + sy * sr
    result[..., 1, 0] = sy * cp
    result[..., 1, 1] = sy * sp * sr + cy * cr
    result[..., 1, 2] = sy * sp * cr - cy * sr
    result[..., 2, 0] = -sp
    result[..., 2, 1] = cp * sr
    result[..., 2, 2] = cp * cr
    return result


def range_image_to_point_cloud(
    range_image: np.ndarray,
    extrinsic: np.ndarray,
    beam_inclinations: np.ndarray,
    *,
    pixel_pose: np.ndarray | None = None,
    frame_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a Waymo ``[H, W, 4]`` range image to vehicle-frame XYZ."""
    height, width = range_image.shape[:2]
    if beam_inclinations.shape != (height,):
        raise ValueError(f"Expected {height} beam inclinations, got {beam_inclinations.shape}")

    azimuth_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (np.arange(width, 0, -1, dtype=np.float64) - 0.5) / width
    azimuth = (ratios * 2.0 - 1.0) * np.pi - azimuth_correction
    inclination = beam_inclinations[:, None]
    distance = range_image[..., 0].astype(np.float64, copy=False)

    cos_inclination = np.cos(inclination)
    sensor_points = np.stack(
        (
            np.cos(azimuth)[None, :] * cos_inclination * distance,
            np.sin(azimuth)[None, :] * cos_inclination * distance,
            np.sin(inclination) * distance,
        ),
        axis=-1,
    )
    points = np.einsum("ij,hwj->hwi", extrinsic[:3, :3], sensor_points) + extrinsic[:3, 3]

    if pixel_pose is not None:
        if frame_pose is None:
            raise ValueError("frame_pose is required when pixel_pose is provided")
        rotations = rotation_matrix(pixel_pose[..., 0], pixel_pose[..., 1], pixel_pose[..., 2])
        world_points = np.einsum("hwij,hwj->hwi", rotations, points) + pixel_pose[..., 3:6]
        world_to_vehicle = np.linalg.inv(frame_pose)
        points = np.einsum("ij,hwj->hwi", world_to_vehicle[:3, :3], world_points) + world_to_vehicle[:3, 3]

    return points[distance > 0].astype(range_image.dtype, copy=False)


def beam_inclinations(height: int, values, minimum: float, maximum: float) -> np.ndarray:
    """Build inclinations in the row order used by Waymo range images."""
    if values is None or len(values) == 0:
        inclinations = np.linspace(minimum, maximum, height, dtype=np.float64)
    else:
        inclinations = np.asarray(values, dtype=np.float64)
    return inclinations[::-1]
