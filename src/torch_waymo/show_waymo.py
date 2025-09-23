import argparse
import io
import math

from torch_waymo import WaymoDataset

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required for this demo. Install with `pip install pillow`.") from e

import matplotlib.pyplot as plt
import numpy as np

from torch_waymo.protocol.dataset_proto import CameraName
from torch_waymo.protocol.label_proto import Type as LabelType


def _decode_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGB"))


def _describe_frame(frame, index: int) -> str:
    lines = []
    frame_type = frame.__class__.__name__
    timestamp = getattr(frame, "timestamp_micros", "N/A")

    # Cameras
    if hasattr(frame, "images"):
        cameras = frame.images
        cam_names = []
        for ci in cameras:
            try:
                cam_names.append(CameraName(ci.name).name)
            except Exception:
                cam_names.append(str(ci.name))
        cam_summary = f"{len(cameras)} cameras: [" + ", ".join(cam_names) + "]"
    else:
        cam_summary = "0 cameras (simplified frame)"

    # Lasers
    if hasattr(frame, "lasers"):
        laser_count = len(frame.lasers)
    else:
        laser_count = 0

    # Labels
    labels = getattr(frame, "laser_labels", []) or []
    total_labels = len(labels)
    label_counts = {lt: 0 for lt in LabelType}
    for lbl in labels:
        try:
            label_counts[lbl.type] += 1
        except Exception:
            pass
    label_parts = []
    for lt, cnt in label_counts.items():
        if cnt > 0:
            label_parts.append(f"{lt.name.replace('TYPE_', '')}:{cnt}")
    label_summary = f"labels: {total_labels}" + (" (" + ", ".join(label_parts) + ")" if label_parts else "")

    # Points
    pts_obj = frame.points
    if pts_obj is None:
        point_summary = "points: N/A"
    else:
        if isinstance(pts_obj, (list, tuple)):
            total_pts = sum(getattr(p, "shape", [0])[0] if hasattr(p, "shape") else 0 for p in pts_obj)
        else:
            total_pts = getattr(pts_obj, "shape", [0])[0] if hasattr(pts_obj, "shape") else 0
        point_summary = f"points: {total_pts}"

    lines.append(f"Frame {index} | {frame_type}")
    lines.append(f" timestamp: {timestamp}")
    lines.append(f" {cam_summary}")
    lines.append(f" lasers: {laser_count}")
    lines.append(f" {label_summary}")
    lines.append(f" {point_summary}")
    return "\n" + "\n".join(lines) + "\n"


def _show_frames(dataset_path: str, split: str, num_frames: int):
    dataset = WaymoDataset(dataset_path, split)
    for i in range(num_frames):
        frame = dataset[i]
        print(_describe_frame(frame, i))
        if hasattr(frame, "images"):
            cam_images = frame.images
            if len(cam_images) == 0:
                print(f"Frame {i} has no images.")
            else:
                n = len(cam_images)
                cols = min(3, n)
                rows = math.ceil(n / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
                if not isinstance(axes, np.ndarray):  # single axis
                    axes = np.array([[axes]])
                axes = axes.reshape(rows, cols)

                for idx, cam_img in enumerate(cam_images):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r, c]
                    data = cam_img.image
                    if isinstance(data, bytes):
                        try:
                            arr = _decode_image(data)
                        except Exception as e:
                            print(f"Decode failure for frame {i} image {idx}: {e}")
                            arr = np.zeros((32, 32, 3), dtype=np.uint8)
                    else:
                        arr = data
                        if arr.ndim == 2:
                            arr = np.repeat(arr[..., None], 3, axis=-1)
                        elif arr.shape[2] == 4:
                            arr = arr[..., :3]
                    # Title from enum name
                    try:
                        cam_name = CameraName(cam_img.name).name
                    except Exception:
                        cam_name = str(cam_img.name)
                    ax.imshow(arr)
                    ax.set_title(cam_name)
                    ax.axis("off")

                for extra_idx in range(n, rows * cols):
                    r = extra_idx // cols
                    c = extra_idx % cols
                    axes[r, c].axis("off")

                fig.suptitle(f"Frame {i} - All camera images ({n})")
                plt.tight_layout()
                plt.show()
        else:
            print(f"Frame {i} is a SimplifiedFrame (no images available).")


def main():
    parser = argparse.ArgumentParser(description="Show Waymo dataset camera images in a grid per frame.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to converted or converted_simplified dataset root")
    parser.add_argument(
        "-s", "--split", choices=["training", "validation", "testing"], default="training", help="Dataset split"
    )
    parser.add_argument("-n", "--num-frames", type=int, default=10, help="Number of frames to display")
    args = parser.parse_args()

    _show_frames(args.dataset, args.split, args.num_frames)


if __name__ == "__main__":
    main()
