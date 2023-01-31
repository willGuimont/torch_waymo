import argparse
import pathlib
import pickle

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
import tqdm

from . import SimplifiedFrame
from .protocol import dataset_proto
from .protocol.dataset_proto import Frame

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


def generate_cache(root_path: pathlib.Path, split: str):
    split_path = root_path.joinpath(split)
    split_cache_path = root_path.joinpath('converted').joinpath(split)

    split_path.mkdir(parents=True, exist_ok=True)
    split_cache_path.mkdir(parents=True, exist_ok=True)

    # Cache sequence lengths
    sequence_paths = sorted(list(split_path.iterdir()))
    seq_lens = _cache_seq_lens(sequence_paths, split_cache_path)

    # Generate the new dataset
    frame_count = 0
    for seq_idx, seq_len in tqdm.tqdm(enumerate(seq_lens), total=len(seq_lens)):
        seq_path = sequence_paths[seq_idx]
        seq = tf.data.TFRecordDataset(seq_path, compression_type='')
        for data in seq:
            frame_path = split_cache_path.joinpath(f'{frame_count}.pkl')
            frame_count += 1
            if not frame_path.exists():
                simple_frame = _load_frame(data)
                with open(frame_path, 'wb') as f:
                    pickle.dump(simple_frame, f)


def _cache_seq_lens(sequence_paths, split_cache_path):
    seq_len_cache_path = split_cache_path.joinpath('len.pkl')
    if seq_len_cache_path.exists():
        with open(seq_len_cache_path, 'rb') as f:
            seq_lens = pickle.load(f)
    else:
        print('Computing sequence lengths, might take a while')
        seq_lens = list(
            tqdm.tqdm((_get_size(s) for s in sequence_paths), total=len(sequence_paths)))
        with open(seq_len_cache_path, 'wb') as f:
            pickle.dump(seq_lens, f)
    return seq_lens


def _get_size(s: pathlib.Path) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(s, compression_type=''))


def _load_frame(data):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    (range_images, camera_projections, _,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose)
    clean_frame = dataset_proto.from_data(Frame, frame)
    simple_frame = SimplifiedFrame(clean_frame.context, clean_frame.timestamp_micros, clean_frame.pose,
                                   clean_frame.laser_labels, clean_frame.no_label_zones, points)
    return simple_frame


SPLITS = ['training', 'validation', 'testing']


def main():
    global parser, args, split
    parser = argparse.ArgumentParser(prog='Convert Waymo',
                                     description='Convert the Waymo Open Dataset to remove all dependencies to Tensorflow')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the Waymo Open Dataset')
    parser.add_argument('-s', '--splits', type=str, choices=SPLITS,
                        nargs='+', default=SPLITS, help='Specify the splits you want to process')
    args = parser.parse_args()
    dataset_path = pathlib.Path(args.dataset)
    splits = args.splits
    for split in splits:
        print(f'Processing {split}...')
        generate_cache(dataset_path, split)


if __name__ == '__main__':
    main()
