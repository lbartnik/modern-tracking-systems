from .linear import LinearKalmanFilter
from ..motion import ConstantVelocityModel


def linear_ncv(noise_intensity: float) -> LinearKalmanFilter:
    return LinearKalmanFilter(ConstantVelocityModel(noise_intensity),
                              [[1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0]])