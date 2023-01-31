# torch_waymo

Load Waymo Open Dataset in PyTorch

## Usage

Requires:
- Python < 3.10
- nvidia-container-toolkit (to run inside of docker)
- nvidia-container-runtime (also for docker)

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
pip install -r requirements_tf.txt

# Convert all the dataset
python convert_waymo.py --dataset <path/to/waymo>
# Or only convert the training split
python convert_waymo.py --dataset <path/to/waymo> --split training
# Or only convert multiple splits
python convert_waymo.py --dataset <path/to/waymo> --split training validation
```

### Load it in your project

```python
from src.torch_waymo.dataset.waymo_dataset import WaymoDataset

train_dataset = WaymoDataset('~/Datasets/Waymo/converted', 'training')

for i in range(10):
    # frame is of type SimplifiedFrame
    frame = train_dataset[i]
    print(frame.timestamp_micros)
```
