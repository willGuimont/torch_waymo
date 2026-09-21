# torch_waymo

Load converted [Waymo Open Dataset](https://waymo.com/open/) frames with a PyTorch `Dataset`.

The runtime package supports Python 3.10 or newer and does not require TensorFlow. Raw-data conversion uses Python 3.10 on Linux x86-64 with Waymo Open Dataset SDK 1.6.7 and TensorFlow 2.13.

## Download the dataset

After accepting the Waymo dataset terms, authenticate the Google Cloud CLI and download the Perception dataset. Training and validation require approximately 1.02 TB.

```shell
gcloud auth login
mkdir -p ~/Datasets/Waymo
gcloud storage rsync --recursive \
  gs://waymo_open_dataset_v_1_4_3/individual_files/training \
  ~/Datasets/Waymo/training
gcloud storage rsync --recursive \
  gs://waymo_open_dataset_v_1_4_3/individual_files/validation \
  ~/Datasets/Waymo/validation
```

This checkout expects the optional local symlink `waymo -> ~/Datasets/Waymo`. It is ignored by Git.

## Convert raw frames

Create a Python 3.10 environment for the Waymo SDK:

```shell
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Converts training and validation to ./waymo/converted.
torch-waymo-convert --dataset ./waymo

# Point clouds and labels only; writes ./waymo/converted_simplified.
torch-waymo-convert --dataset ./waymo --simplified

# Select one or more splits explicitly.
torch-waymo-convert --dataset ./waymo --splits training
```

Conversion is resumable: existing frame pickle files are skipped. Each converted split contains a `len.pkl` index.

## Load converted frames

Install only the runtime dependencies in downstream projects:

```shell
pip install torch_waymo
```

```python
from torch_waymo import WaymoDataset

train_dataset = WaymoDataset(
    "~/Datasets/Waymo/converted_simplified",
    "training",
)

frame = train_dataset[0]
print(frame.timestamp_micros)
print(sum(points.shape[0] for points in frame.points))
```

Use `~/Datasets/Waymo/converted` instead when camera images and the complete frame are needed. Home-directory (`~`) paths are expanded automatically.

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
