# torch_waymo

Load converted [Waymo Open Dataset](https://waymo.com/open/) Perception frames with a PyTorch `Dataset`.

The recommended Waymo v2 Parquet converter supports Python 3.10 or newer and does not require TensorFlow. The legacy v1 TFRecord converter remains available on Python 3.10.

## Citation

If this package is useful in your work, please cite it:

```bibtex
@software{Guimont-Martin_A_PyTorch_dataloader_2023,
    author = {Guimont-Martin, William},
    month = {1},
    title = {{A PyTorch dataloader for Waymo Open Dataset}},
    version = {0.1.1},
    year = {2023}
}
```

Remember to also follow the [Waymo Open Dataset citation requirements](https://waymo.com/open/terms/) for the dataset release you use.

## Download the dataset

Accept the Waymo dataset terms and authenticate the Google Cloud CLI first:

```shell
gcloud auth login
```

### One-segment v2 subset

The segment below contains 199 frames and all supported component types. It is the quickest way to try the full conversion without downloading an entire split (approximately 589 MB raw and 1.3 GB converted):

```shell
SEGMENT=10023947602400723454_1120_000_1140_000
WAYMO_V2_ROOT="${HOME}/Datasets/Waymo-v2-subset"

for component in \
  camera_box camera_calibration camera_hkp camera_image camera_segmentation \
  camera_to_lidar_box_association lidar lidar_box lidar_calibration \
  lidar_camera_projection lidar_camera_synced_box lidar_hkp lidar_pose \
  lidar_segmentation projected_lidar_box stats vehicle_pose
do
  mkdir -p "${WAYMO_V2_ROOT}/training/${component}"
  gcloud storage cp \
    "gs://waymo_open_dataset_v_2_0_1/training/${component}/${SEGMENT}.parquet" \
    "${WAYMO_V2_ROOT}/training/${component}/"
done
```

Convert and smoke-test the subset:

```shell
uv sync --extra waymo
uv run torch-waymo-convert \
  --dataset "${HOME}/Datasets/Waymo-v2-subset" \
  --splits training

uv run python -c "from torch_waymo import WaymoDataset; d = WaymoDataset('${HOME}/Datasets/Waymo-v2-subset/converted', 'training'); f = d[0]; assert len(d) == 199 and len(f.images) == len(f.lasers) == len(f.points) == 5; print(len(d), sum(map(len, f.points)))"
```

### Complete v2 split

Use recursive synchronization when you need every segment. The modular format lets you omit components that your application does not use.

```shell
WAYMO_V2_ROOT="${HOME}/Datasets/Waymo-v2"

for component in \
  camera_box camera_calibration camera_hkp camera_image camera_segmentation \
  camera_to_lidar_box_association lidar lidar_box lidar_calibration \
  lidar_camera_projection lidar_camera_synced_box lidar_hkp lidar_pose \
  lidar_segmentation projected_lidar_box stats vehicle_pose
do
  gcloud storage rsync --recursive \
    "gs://waymo_open_dataset_v_2_0_1/training/${component}" \
    "${WAYMO_V2_ROOT}/training/${component}"
done
```

Repeat with `validation` or `testing` as needed. Camera/LiDAR segmentation and human-keypoint components are sparse; absent rows are handled normally.

### One-segment legacy v1 subset

The matching v1 TFRecord is useful for checking legacy compatibility:

```shell
WAYMO_V1_ROOT="${HOME}/Datasets/Waymo-v1-subset"
mkdir -p "${WAYMO_V1_ROOT}/training"
gcloud storage cp \
  gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord \
  "${WAYMO_V1_ROOT}/training/"
```

## Convert raw frames

Create the v2 conversion environment with [uv](https://docs.astral.sh/uv/):

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

Full conversion preserves v2 camera images, both LiDAR returns, camera projections, per-pixel poses, camera and LiDAR segmentation, calibrations, 2D/3D boxes, synchronized boxes, associations, keypoints, statistics, and generated first-return point clouds. Waymo v2 does not include the v1 polygonal `no_label_zones`, so `Frame.no_label_zones` is empty; the equivalent per-pixel no-label-zone flag remains in channel 3 of each `RangeImage.values` array. Maps are also only available in the v1 dataset.

To convert legacy v1.4.x TFRecords, use a separate Python 3.10 environment so the TensorFlow dependency stack does not replace your main environment:

```shell
UV_PROJECT_ENVIRONMENT=.venv-waymo-v1 uv sync --python 3.10 --extra waymo-v1
UV_PROJECT_ENVIRONMENT=.venv-waymo-v1 uv run torch-waymo-convert \
  --format tfrecord \
  --dataset ~/Datasets/Waymo-v1 \
  --splits training
```

## Run the tests

The regular test suite uses generated fixtures, so it does not require downloading Waymo data:

```shell
uv sync --extra waymo
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

To test the real v1 subset after downloading it above:

```shell
UV_PROJECT_ENVIRONMENT=.venv-waymo-v1 uv run torch-waymo-convert \
  --format tfrecord \
  --dataset ~/Datasets/Waymo-v1-subset \
  --splits training

UV_PROJECT_ENVIRONMENT=.venv-waymo-v1 uv run python -c "from torch_waymo import WaymoDataset; d = WaymoDataset('~/Datasets/Waymo-v1-subset/converted', 'training'); f = d[0]; assert len(d) == 199 and len(f.images) == len(f.lasers) == len(f.points) == 5; print(len(d), sum(map(len, f.points)))"
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

For a guided example covering camera tensors, LiDAR tensors, range images, 2D and 3D annotations, segmentation, and simple visualization, open [`examples/waymo_torch_walkthrough.ipynb`](https://github.com/willGuimont/torch_waymo/blob/main/examples/waymo_torch_walkthrough.ipynb):

```shell
TORCH_WAYMO_DATASET="${HOME}/Datasets/Waymo-v2-subset/converted" \
  uv run --with jupyter --with "matplotlib>=3.9" --with "pillow>=11" \
  jupyter lab examples/waymo_torch_walkthrough.ipynb
```
