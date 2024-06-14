import numpy as np
from typing import List


class ConstantVelocityModel:
    def __init__(self, noise_intensity: float = 1):
        self.spatial_dim = 3    # number of spatial dimensions
        self.state_dim = 6      # size of the state
        self.space_name = "CartesianPositionVelocity"
        self.noise_intensity = noise_intensity

    @property
    def name(self):
        """Motion model name"""
        return f"cv_{self.noise_intensity}"
    
    def F(self, dt: float):
        return np.array([[1, 0, 0, dt, 0, 0],
                         [0, 1, 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

    def Q(self, dt: float):
        # "Estimation with Applications to Tracking and Navigation", p. 270
        # process noise for the continuous white noise acceleration (CWNA)
        # model; velocity changes at discrete time intervals by white-noise
        # derived values
        dt3 = dt**3 / 3
        dt2 = dt**2 / 2
        return np.array([[dt3, 0, 0, dt2, 0 ,0],
                         [0, dt3, 0, 0, dt2, 0],
                         [0, 0, dt3, 0, 0, dt2],
                         [dt2, 0, 0, dt, 0, 0],
                         [0, dt2, 0, 0, dt, 0],
                         [0, 0, dt2, 0, 0, dt]]) * self.noise_intensity**2


def constant_velocity_models(*args) -> List[ConstantVelocityModel]:
    return [ConstantVelocityModel(sigma) for sigma in args]
