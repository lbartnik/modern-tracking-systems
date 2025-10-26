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


def initialize_velocity(m0, m1):
    """Initialize position+velocity mean and covariance from two measurements.
    """
    dt = m1.time - m0.time
    dp = m1.z - m0.z
    
    vel = dp / dt
    P_vel = (m0.R + m1.R) / (dt * dt)
    P_pos_vel = m1.R / dt
    
    x = np.concatenate((m1.z.squeeze(), vel.squeeze()))
    P = np.zeros((6, 6))
    P[:3, :3] = m1.R
    P[3:, 3:] = P_vel
    P[:3, 3:] = P_pos_vel
    P[3:, :3] = P_pos_vel

    return x, P
