# torch_waymo

Load converted [Waymo Open Dataset](https://waymo.com/open/) Perception frames with a PyTorch `Dataset`.

The recommended Waymo v2 Parquet converter supports Python 3.10 or newer and does not require TensorFlow. The legacy v1 TFRecord converter remains available on Python 3.10.

## Download the dataset

After accepting the Waymo dataset terms, authenticate the Google Cloud CLI and download the modular Perception v2.0.1 components. The modular format lets you omit components that you do not need.

```shell
gcloud auth login
mkdir -p ~/Datasets/Waymo-v2/training

for component in \
  camera_box camera_calibration camera_hkp camera_image camera_segmentation \
  camera_to_lidar_box_association lidar lidar_box lidar_calibration \
  lidar_camera_projection lidar_camera_synced_box lidar_hkp lidar_pose \
  lidar_segmentation projected_lidar_box stats vehicle_pose
do
  gcloud storage rsync --recursive \
    "gs://waymo_open_dataset_v_2_0_1/training/${component}" \
    "${HOME}/Datasets/Waymo-v2/training/${component}"
done
```

Repeat for `validation` or `testing` as needed. Camera/LiDAR segmentation and human-keypoint components are sparse; absent rows are handled normally.

## Convert raw frames

Create the conversion environment with [uv](https://docs.astral.sh/uv/):

```shell
uv sync --extra waymo

# Auto-detects v2 Parquet and converts training and validation.
uv run torch-waymo-convert --dataset ~/Datasets/Waymo-v2

# Point clouds and labels only; writes <dataset>/converted_simplified.
uv run torch-waymo-convert --dataset ~/Datasets/Waymo-v2 --simplified

# Select one or more splits explicitly.
uv run torch-waymo-convert --dataset ~/Datasets/Waymo-v2 --splits training
```

Conversion is resumable: existing frame files are skipped. Parquet frames are stored as compressed `.pkl.gz` files, and each converted split contains a `len.pkl` index. `WaymoDataset` also continues to read uncompressed `.pkl` caches produced by the legacy converter.

Full conversion preserves v2 camera images, both LiDAR range-image returns, camera projections, per-pixel poses, camera and LiDAR segmentation, calibrations, 2D/3D boxes, synchronized boxes, associations, keypoints, statistics, and generated first-return point clouds. Waymo v2 does not include the v1 polygonal `no_label_zones`, so `Frame.no_label_zones` is empty; the equivalent per-pixel no-label-zone flag remains in channel 3 of each `RangeImage.values` array. Maps are also only available in the v1 dataset.

To convert legacy v1.4.x TFRecords instead:

```shell
uv sync --python 3.10 --extra waymo-v1
uv run --python 3.10 torch-waymo-convert \
  --format tfrecord \
  --dataset ~/Datasets/Waymo-v1
```

## Load converted frames

Add only the runtime package to a downstream uv project:

```shell
uv add torch_waymo
```

```python
from torch_waymo import WaymoDataset

train_dataset = WaymoDataset(
    "~/Datasets/Waymo-v2/converted_simplified",
    "training",
)

frame = train_dataset[0]
print(frame.timestamp_micros)
print(sum(points.shape[0] for points in frame.points))
```

Use `~/Datasets/Waymo-v2/converted` instead when camera images and the complete frame are needed. Home-directory (`~`) paths are expanded automatically.

## Citation

```bibtex
@software{Guimont-Martin_A_PyTorch_dataloader_2023,
    author = {Guimont-Martin, William},
    month = {1},
    title = {{A PyTorch dataloader for Waymo Open Dataset}},
    version = {0.1.1},
    year = {2023}
}
```
