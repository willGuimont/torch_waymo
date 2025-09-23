import argparse
import pathlib
import pickle

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

from torch_waymo.dataset import SimplifiedFrame
from torch_waymo.protocol import dataset_proto
from torch_waymo.protocol.dataset_proto import Frame


def generate_cache(root_path: pathlib.Path, split: str, simplified: bool = False):
    """
    Convert the Waymo Open Dataset to a format without Tensorflow dependencies.
    :param root_path: Path to the Waymo dataset root directory.
    :param split: One of "training", "validation", "testing".
    :param simplified: If True, store simplified frames (no images) into 'converted_simplified' instead of full frames.
    :return: None
    """
    output_root_name = "converted_simplified" if simplified else "converted"
    split_path = root_path.joinpath(split)
    split_cache_path = root_path.joinpath(output_root_name).joinpath(split)

    split_path.mkdir(parents=True, exist_ok=True)
    split_cache_path.mkdir(parents=True, exist_ok=True)

    # Cache sequence lengths
    sequence_paths = sorted(list(split_path.iterdir()))
    seq_lens = _cache_seq_lens(sequence_paths, split_cache_path)

    # Generate the new dataset
    frame_count = 0
    for seq_idx, seq_len in tqdm.tqdm(enumerate(seq_lens), total=len(seq_lens)):
        seq_path = sequence_paths[seq_idx]
        seq = tf.data.TFRecordDataset(seq_path, compression_type="")
        for data in seq:
            frame_path = split_cache_path.joinpath(f"{frame_count}.pkl")
            frame_count += 1
            if not frame_path.exists():
                obj = _load_frame(data, simplified=simplified)
                with open(frame_path, "wb") as f:
                    pickle.dump(obj, f)


def _cache_seq_lens(sequence_paths, split_cache_path):
    seq_len_cache_path = split_cache_path.joinpath("len.pkl")
    if seq_len_cache_path.exists():
        with open(seq_len_cache_path, "rb") as f:
            seq_lens = pickle.load(f)
    else:
        print("Computing sequence lengths, might take a while")
        seq_lens = list(tqdm.tqdm((_get_size(s) for s in sequence_paths), total=len(sequence_paths)))
        with open(seq_len_cache_path, "wb") as f:
            pickle.dump(seq_lens, f)
    return seq_lens


def _get_size(s: pathlib.Path) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(s, compression_type=""))


def _load_frame(data, simplified: bool):
    """
    Load a single frame from TFRecord data.
    :param data: TFRecord data
    :param simplified: If True, return a SimplifiedFrame (no images), else return full Frame.
    :return: Frame or SimplifiedFrame
    """
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    converted_frame = dataset_proto.from_data(Frame, frame)

    # Generate point cloud
    (
        range_images,
        camera_projections,
        _,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )
    converted_frame.points = points

    if not simplified:
        # Return full frame (images, lasers, labels). Point cloud generation skipped for speed.
        return converted_frame

    # Simplified path: compute point cloud and build SimplifiedFrame (no images stored)
    simple_frame = SimplifiedFrame(
        converted_frame.context,
        converted_frame.timestamp_micros,
        converted_frame.pose,
        converted_frame.laser_labels,
        converted_frame.no_label_zones,
        converted_frame.points,
    )
    return simple_frame


def main():
    SPLITS = ["training", "validation", "testing"]

    parser = argparse.ArgumentParser(
        prog="Convert Waymo",
        description="Convert the Waymo Open Dataset to remove all dependencies to Tensorflow",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to the Waymo Open Dataset",
    )
    parser.add_argument(
        "-s",
        "--splits",
        type=str,
        choices=SPLITS,
        nargs="+",
        default=SPLITS,
        help="Specify the splits you want to process",
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Store simplified frames (no images) into 'converted_simplified' instead of full frames.",
    )

    args = parser.parse_args()
    dataset_path = pathlib.Path(args.dataset).expanduser()
    splits = args.splits
    simplified = args.simplified

    for split in splits:
        mode = "simplified" if simplified else "full"
        print(f"Processing {split} in {mode} mode...")
        generate_cache(dataset_path, split, simplified=simplified)


if __name__ == "__main__":
    main()
