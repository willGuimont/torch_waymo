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
# Make a tf venv
python -m venv venv_tf
source venv_tf/bin/activate
pip install torch_waymo[waymo]

# Convert all the dataset
torch-waymo-convert --dataset <path/to/waymo>
# Or only convert the training split
torch-waymo-convert --dataset <path/to/waymo> --split training
# Or convert multiple splits
torch-waymo-convert --dataset <path/to/waymo> --split training validation
```

### Load it in your project

Now that the dataset is converted, you don't have to depend on `waymo-open-dataset-tf-2-6-0` in your project.
You can simply install `torch_waymo` in your project.

```shell
pip install torch_waymo
```

Example usage:

```python
from torch_waymo import WaymoDataset

train_dataset = WaymoDataset('~/Datasets/Waymo/converted', 'training')

for i in range(10):
    # frame is of type SimplifiedFrame
    frame = train_dataset[i]
    print(frame.timestamp_micros)
```
