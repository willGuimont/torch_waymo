import argparse
import pathlib


def detect_format(root_path: pathlib.Path, split: str) -> str:
    if (root_path / split / "vehicle_pose").is_dir():
        return "parquet"
    return "tfrecord"


def generate_cache(
    root_path: pathlib.Path,
    split: str,
    simplified: bool = False,
    source_format: str = "auto",
) -> None:
    """Convert a Waymo v1 TFRecord or v2 Parquet split into frame pickles."""
    selected_format = detect_format(root_path, split) if source_format == "auto" else source_format
    if selected_format == "parquet":
        from torch_waymo.converters.parquet import generate_parquet_cache

        generate_parquet_cache(root_path, split, simplified)
    elif selected_format == "tfrecord":
        from torch_waymo.converters.tfrecord import generate_tfrecord_cache

        generate_tfrecord_cache(root_path, split, simplified)
    else:
        raise ValueError(f"Unknown source format: {selected_format}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Convert Waymo",
        description="Convert Waymo v1 TFRecord or v2 Parquet data into TensorFlow-free frame files.",
    )
    parser.add_argument("-d", "--dataset", required=True, help="Path to the Waymo dataset root")
    parser.add_argument(
        "-s",
        "--splits",
        choices=("training", "validation", "testing"),
        nargs="+",
        default=("training", "validation"),
        help="Splits to process",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "parquet", "tfrecord"),
        default="auto",
        help="Raw dataset format; auto detects v2 from the vehicle_pose directory",
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Store point clouds and labels without camera images",
    )
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset).expanduser()
    for split in args.splits:
        selected_format = detect_format(dataset_path, split) if args.format == "auto" else args.format
        mode = "simplified" if args.simplified else "full"
        print(f"Processing {split} from {selected_format} in {mode} mode...")
        generate_cache(dataset_path, split, args.simplified, selected_format)


if __name__ == "__main__":
    main()
