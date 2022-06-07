from typing import Iterable, Tuple, Dict

from plyfile import PlyData
import numpy as np

from .base_loader import BasePointCloudLoader, COORDS_KEY, COLOR_KEY


class PLYColorLoader(BasePointCloudLoader):
    def __init__(self, path_to_file: str):
        super().__init__(path_to_file)

    def iter_chunks(self, chunk_size: int) -> Iterable[Dict[str, np.ndarray]]:
        assert chunk_size > 0
        with open(self.path_to_file, "rb") as file:
            data = PlyData.read(file)

        total_vertices = len(data["vertex"]["x"])

        for i in range(0, total_vertices, chunk_size):
            end = i + chunk_size
            x = data["vertex"]["x"][i:end]
            y = data["vertex"]["y"][i:end]
            z = data["vertex"]["z"][i:end]
            red = data["vertex"]["red"][i:end]
            green = data["vertex"]["green"][i:end]
            blue = data["vertex"]["blue"][i:end]

            colors = np.vstack((red, green, blue)).T

            yield {COORDS_KEY: np.vstack((x, y, z)).T, COLOR_KEY: colors}

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        min_bounds = None
        max_bounds = None

        for chunk_data in self.iter_chunks(10_000):
            coords = chunk_data[COORDS_KEY]
            if min_bounds is None:
                min_bounds = coords.min(axis=0)
                max_bounds = coords.max(axis=0)
            else:
                min_bounds = np.minimum(min_bounds, coords.min(axis=0))
                max_bounds = np.maximum(max_bounds, coords.max(axis=0))

        return min_bounds, max_bounds
