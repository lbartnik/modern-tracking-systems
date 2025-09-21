import numpy as np
from numpy.typing import ArrayLike
from typing import List

from ..sensor import SensorMeasurement


__all__ = ['Tracker', 'Track']


class Track(object):
    track_id: int
    time: float
    mean: np.ndarray
    cov: np.ndarray

    def __init__(self, track_id: int, time: float, mean: ArrayLike, cov: ArrayLike):
        self.track_id = track_id
        self.time = time
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)


class Tracker(object):
    def add_measurements(self, measurements: List[SensorMeasurement]):
        raise Exception(f"add_measurements() method not implemented in {self.__class__.__name__}")

    def estimate_tracks(self, time: float) -> List[Track]:
        raise Exception(f"estimate_tracks() method not implemented in {self.__class__.__name__}")

    def reset(self) -> List[Track]:
        raise Exception(f"reset() method not implemented in {self.__class__.__name__}")
