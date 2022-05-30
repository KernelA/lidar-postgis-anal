import h5py
import numpy as np

DATASET_PATH = "/points"


def append_points(file: h5py.File, points: np.ndarray) -> None:
    dataset = file.get(DATASET_PATH)

    if dataset is None:
        file.create_dataset(DATASET_PATH, maxshape=(None, points.shape[1]), data=points)
    else:
        last_index = dataset.shape[0]
        dataset[last_index:] = points
