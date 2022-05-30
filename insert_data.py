import tempfile
import pathlib
import os

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPoint
import h5py
import hydra
import numpy as np
from tqdm import tqdm
from hydra.utils import instantiate

from db import LazPoints
from db.base import Base

from loader import BasePointCloudLoader
from utils import append_points, DATASET_PATH


def insert_points(session: Session, file_path: str, file: h5py.File, chunk_id: int):
    res = session.query(LazPoints).filter(LazPoints.chunk_id == chunk_id).filter(
        LazPoints.file == file_path).one_or_none()

    if res is not None:
        return

    points = file.get(DATASET_PATH)[:]

    multi_point = MultiPoint(points)
    session.add(LazPoints(file=file_path, chunk_id=chunk_id, points=from_shape(multi_point)))


def generate_bounds(min_value: float, max_value: float, step: float):
    values = []
    value = min_value

    while value < max_value:
        values.append(value)
        value += step

    values.append(max_value)

    return np.array(values)


def get_chunk_indices(points: np.ndarray, x_bounds: np.ndarray, y_bounds: np.ndarray, z_bounds: np.ndarray):
    x_indices = np.searchsorted(x_bounds, points[:, 0], side="right") - 1  # 0 to n - 1
    y_indices = np.searchsorted(y_bounds, points[:, 1], side="right") - 1
    z_indices = np.searchsorted(z_bounds, points[:, 2], side="right") - 1

    # Convert three dimensional index to the one dimensional
    width = len(x_bounds) - 1
    height = len(y_bounds) - 1

    return z_indices * width * height + y_indices * width + x_indices


def insert_data(session_factory, path_to_file: str, loader_config, chunk_size: int, voxel_size: float):
    loader: BasePointCloudLoader = instantiate(loader_config, path_to_file)

    file_loc = pathlib.Path(path_to_file).as_posix()

    min_bounds, max_bounds = loader.get_bounds()

    x_intervals = generate_bounds(min_bounds[0], max_bounds[0], voxel_size)
    y_intervals = generate_bounds(min_bounds[1], max_bounds[1], voxel_size)
    z_intervals = generate_bounds(min_bounds[2], max_bounds[2], voxel_size)

    files = []
    chunk_ids = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for chunk_xyz in loader.iter_chunks(chunk_size):
            chunk_index_per_point = get_chunk_indices(
                chunk_xyz, x_intervals, y_intervals, z_intervals)

            for chunk_index in tqdm(set(chunk_index_per_point), desc="Save to hdf"):
                file_path = os.path.join(tmp_dir, f"chunk_{chunk_index}.h5")
                files.append(file_path)
                chunk_ids.append(int(chunk_index))

                with h5py.File(file_path, "a") as hdf_file:
                    local_chunk = chunk_xyz[chunk_index_per_point == chunk_index]
                    append_points(hdf_file, local_chunk)

                del file_path

        with session_factory() as session:
            for file, chunk_id in tqdm(zip(files, chunk_ids), total=len(files), desc="Insert chunks"):
                with h5py.File(file, "r") as hdf_file:
                    insert_points(session, file_loc, hdf_file, chunk_id)
                session.commit()


@hydra.main(config_path="configs", config_name="insert_data")
def main(config):

    engine = create_engine(config.db.url)

    try:
        Base.metadata.create_all(engine)
        CustomSession = sessionmaker(engine)
        insert_data(CustomSession, config.data.filepath, config.loader,
                    config.data_splitting.chunk_size, config.data_splitting.voxel_size)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
