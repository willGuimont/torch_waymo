import pathlib
import pickle
from typing import Callable, Optional, Union

from torch.utils.data import Dataset

from ..protocol.dataset_proto import Frame
from .simplified_frame import SimplifiedFrame


class WaymoDataset(Dataset):
    def __init__(self, root_path: str, split: str, transform: Optional[Callable] = None):
        self._root_path = pathlib.Path(root_path).expanduser()
        self._split = split
        self._split_path = self._root_path.joinpath(split)
        self._transform = transform

        # Validate root path existence
        if not self._root_path.exists() or not self._root_path.is_dir():
            raise FileNotFoundError(f"Dataset root path does not exist or is not a directory: {self._root_path}")

        # Validate split directory existence
        if not self._split_path.exists() or not self._split_path.is_dir():
            raise FileNotFoundError(
                f"Split path does not exist: {self._split_path}. You may have the wrong root or forgot to convert this split."
            )

        self._seq_len_cache_path = self._split_path.joinpath("len.pkl")
        if self._seq_len_cache_path.exists():
            with open(self._seq_len_cache_path, "rb") as f:
                self._seq_lens = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Could not find sequence length cache file: {self._seq_len_cache_path}. Conversion may be incomplete."
            )

    def __len__(self) -> int:
        return sum(self._seq_lens)

    def __getitem__(self, idx: int) -> Union[SimplifiedFrame, Frame]:
        path = self._split_path.joinpath(f"{idx}.pkl")
        if path.exists():
            return self._get_frame(path)
        else:
            raise IndexError(f"Could not load frame at index {idx} (missing file {path}).")

    def _get_frame(self, path):
        with open(path, "rb") as f:
            x = pickle.load(f)
        if self._transform is not None:
            x = self._transform(x)
        return x
