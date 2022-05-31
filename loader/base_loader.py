from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict

import numpy as np


class BasePointCloudLoader(ABC):

    def __init__(self, path_to_file: str):
        self.path_to_file = path_to_file

    @abstractmethod
    def iter_chunks(self, chunk_size: int) -> Iterable[Dict[str, np.ndarray]]:
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get [min_x, min_y, min_z] and [max_x, max_y, max_z] extents of entire point cloud.
        """
        pass
