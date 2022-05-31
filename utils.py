from typing import Dict

import h5py
import numpy as np


def append_points(file: h5py.File, points_data: Dict[str, np.ndarray]) -> None:
    for dataset_name in points_data:
        dataset = file.get(dataset_name)

        points = points_data[dataset_name]

        if dataset is None:
            file.create_dataset(dataset_name, maxshape=(None, points.shape[1]), data=points)
        else:
            last_index = dataset.shape[0]
            dataset[last_index:] = points
