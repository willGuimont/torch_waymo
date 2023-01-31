import pathlib
import pickle
from typing import Optional, Callable

from torch.utils.data import Dataset

from .simplified_frame import SimplifiedFrame


class WaymoDataset(Dataset):
    def __init__(self, root_path: str, split: str, transform: Optional[Callable] = None):
        self._root_path = pathlib.Path(root_path).expanduser()
        self._split = split
        self._split_path = self._root_path.joinpath(split)
        self._transform = transform

        self._seq_len_cache_path = self._split_path.joinpath('len.pkl')
        if self._seq_len_cache_path.exists():
            with open(self._seq_len_cache_path, 'rb') as f:
                self._seq_lens = pickle.load(f)
        else:
            raise RuntimeError(f'Could not find {self._seq_len_cache_path}')

    def __len__(self) -> int:
        return self._seq_lens[-1]

    def __getitem__(self, idx: int) -> SimplifiedFrame:
        path = self._split_path.joinpath(f'{idx}.pkl')
        if path.exists():
            return self._get_frame(path)
        else:
            raise IndexError(f'Could not load frame at index {idx}')

    def _get_frame(self, path):
        with open(path, 'rb') as f:
            x = pickle.load(f)
        if self._transform is not None:
            x = self._transform(x)
        return x
