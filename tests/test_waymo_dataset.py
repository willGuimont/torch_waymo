import pickle

import pytest

from torch_waymo import WaymoDataset


@pytest.fixture
def converted_dataset(tmp_path):
    split_path = tmp_path / "training"
    split_path.mkdir()
    with (split_path / "len.pkl").open("wb") as file:
        pickle.dump([1, 2], file)
    for index in range(3):
        with (split_path / f"{index}.pkl").open("wb") as file:
            pickle.dump(index, file)
    return tmp_path


def test_loads_converted_frames(converted_dataset):
    dataset = WaymoDataset(converted_dataset, "training", transform=lambda value: value + 10)

    assert len(dataset) == 3
    assert dataset[1] == 11


def test_rejects_missing_split(converted_dataset):
    with pytest.raises(FileNotFoundError, match="Split path does not exist"):
        WaymoDataset(converted_dataset, "validation")


def test_rejects_missing_frame(converted_dataset):
    dataset = WaymoDataset(converted_dataset, "training")

    with pytest.raises(IndexError, match="missing file"):
        dataset[3]
