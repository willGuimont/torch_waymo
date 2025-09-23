# torch_waymo

Load Waymo Open Dataset in PyTorch

Cite this repository:
```
@software{Guimont-Martin_A_PyTorch_dataloader_2023,
    author = {Guimont-Martin, William},
    month = {1},
    title = {{A PyTorch dataloader for Waymo Open Dataset}},
    version = {0.1.1},
    year = {2023}
}
```

## Usage

Requires:
- Python < 3.10

### Download the dataset

```shell
# Login to gcloud
gcloud auth login

# Download the full dataset
cd <path/to/waymo>
gsutil -m cp -r \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/training" \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/validation" \
  .
```

### Convert it

```shell
# Make a tf venv with Python < 3.10
python3.9 -m venv venv_tf
source venv_tf/bin/activate

# We recommend using uv for faster installs
pip install uv
uv pip install "torch_waymo[waymo]"

# Convert all splits (FULL frames with images & lasers) -> writes to <path>/converted
torch-waymo-convert --dataset <path/to/waymo>

# Convert only training split (FULL frames)
torch-waymo-convert --dataset <path/to/waymo> --splits training

# Convert multiple splits (FULL frames)
torch-waymo-convert --dataset <path/to/waymo> --splits training validation

# (NEW) Convert to SIMPLIFIED frames (no camera images stored, point cloud + labels only)
# Writes to <path>/converted_simplified
torch-waymo-convert --dataset <path/to/waymo> --simplified

# Simplified + specific splits
torch-waymo-convert --dataset <path/to/waymo> --simplified --splits training validation
```

### Load it in your project

Now that the dataset is converted, you don't have to depend on `waymo-open-dataset-tf-2-11-0` in your downstream project.
You can simply install `torch_waymo` in your *runtime* environment.

```shell
pip install torch_waymo
```
Example usage:
train_dataset = WaymoDataset('~/Datasets/Waymo/converted_simplified', 'training')
Example usage (Full conversion):
```python
from torch_waymo import WaymoDataset

# Simplified frames (no images, only point clouds + labels)
train_dataset = WaymoDataset('~/Datasets/Waymo/converted_simplified', 'training')
for i in range(10):
    # frame is of type SimplifiedFrame
    frame = train_dataset[i]
    print(frame.timestamp_micros)
    print(frame.timestamp_micros, len(frame.lasers))

# Full frames (with images)
train_dataset = WaymoDataset('~/Datasets/Waymo/converted', 'training')
for i in range(10):
    # frame is of type Frame
    frame = train_dataset[i]
    print(frame.timestamp_micros)
    print(frame.timestamp_micros, len(frame.images))
```

Notes:
- Paths with `~` are supported; they will expand to your home directory.
- `len.pkl` inside each split directory stores cumulative frame counts for indexing.
- If you re-run conversion, existing frames are skipped (idempotent per frame file).
