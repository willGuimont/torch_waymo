import pathlib
import pickle

import tqdm

from torch_waymo.converters.common import output_split_path, write_frame, write_sequence_lengths
from torch_waymo.dataset import SimplifiedFrame
from torch_waymo.protocol import dataset_proto
from torch_waymo.protocol.dataset_proto import Frame


def generate_tfrecord_cache(root_path: pathlib.Path, split: str, simplified: bool = False) -> None:
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2 as open_dataset
        from waymo_open_dataset.utils import frame_utils
    except ImportError as error:
        raise ImportError(
            "TFRecord conversion requires Python 3.10 and the 'waymo-v1' extra: uv sync --python 3.10 --extra waymo-v1"
        ) from error

    split_path = root_path / split
    output_path = output_split_path(root_path, split, simplified)
    split_path.mkdir(parents=True, exist_ok=True)
    sequence_paths = sorted(path for path in split_path.iterdir() if path.is_file())
    if not sequence_paths:
        raise FileNotFoundError(f"No TFRecord files found in {split_path}")

    lengths_path = output_path / "len.pkl"
    if lengths_path.exists():
        with lengths_path.open("rb") as file:
            sequence_lengths = pickle.load(file)
    else:
        print("Computing sequence lengths, might take a while")
        sequence_lengths = [
            sum(1 for _ in tf.data.TFRecordDataset(path, compression_type="")) for path in tqdm.tqdm(sequence_paths)
        ]
        write_sequence_lengths(output_path, sequence_lengths)

    frame_index = 0
    for sequence_path in tqdm.tqdm(sequence_paths):
        sequence = tf.data.TFRecordDataset(sequence_path, compression_type="")
        for data in sequence:
            frame_path = output_path / f"{frame_index}.pkl"
            if not frame_path.exists():
                proto = open_dataset.Frame()
                proto.ParseFromString(data.numpy())
                frame = dataset_proto.from_data(Frame, proto)
                range_images, camera_projections, _, top_pose = frame_utils.parse_range_image_and_camera_projection(
                    proto
                )
                points, _ = frame_utils.convert_range_image_to_point_cloud(
                    proto, range_images, camera_projections, top_pose
                )
                frame.points = points
                if simplified:
                    frame = SimplifiedFrame(
                        frame.context,
                        frame.timestamp_micros,
                        frame.pose,
                        frame.laser_labels,
                        frame.no_label_zones,
                        frame.points,
                    )
                write_frame(output_path, frame_index, frame)
            frame_index += 1
