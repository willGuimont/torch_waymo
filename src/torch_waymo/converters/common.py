import gzip
import pathlib
import pickle
from collections.abc import Sequence


def output_split_path(root_path: pathlib.Path, split: str, simplified: bool) -> pathlib.Path:
    output_root_name = "converted_simplified" if simplified else "converted"
    path = root_path / output_root_name / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_sequence_lengths(path: pathlib.Path, lengths: Sequence[int]) -> None:
    with (path / "len.pkl").open("wb") as file:
        pickle.dump(list(lengths), file)


def write_frame(path: pathlib.Path, index: int, frame, *, compress: bool = False) -> None:
    plain_path = path / f"{index}.pkl"
    compressed_path = path / f"{index}.pkl.gz"
    if plain_path.exists() or compressed_path.exists():
        return
    frame_path = compressed_path if compress else plain_path
    temporary_path = frame_path.with_name(f".{frame_path.name}.tmp")
    if compress:
        file = gzip.open(temporary_path, "wb", compresslevel=1)
    else:
        file = temporary_path.open("wb")
    with file:
        pickle.dump(frame, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(frame_path)
