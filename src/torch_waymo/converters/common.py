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


def write_frame(path: pathlib.Path, index: int, frame) -> None:
    frame_path = path / f"{index}.pkl"
    if not frame_path.exists():
        with frame_path.open("wb") as file:
            pickle.dump(frame, file)
