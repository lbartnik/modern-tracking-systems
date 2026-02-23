import numpy as np
from numpy.typing import ArrayLike

from .frame import Frame

class Point3D:
    # measurement coordinates are (0,0,range)
    range: float    
    frame: Frame

    x_stdev: float
    y_stdev: float
    z_stdev: float
    
    def __init__(self, vec: ArrayLike, stdev: ArrayLike = [1, 1, 1]):
        vec = np.asarray(vec)
        rng = np.linalg.norm(vec)
        self.range = float(rng)
        self.frame = Frame([0, 0, 0], vec)

        self.x_stdev, self.y_stdev, self.z_stdev = stdev

    def __repr__(self) -> str:
        return f"Point3D(range={self.range:.2f}, frame={self.frame})"
