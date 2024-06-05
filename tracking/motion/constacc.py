import numpy as np
from typing import List


class ConstantAccelerationModel:
    def __init__(self, noise_intensity: float = 1) -> None:
        self.spatial_dim = 3    # number of spatial dimensions
        self.state_dim = 9      # size of the state
        self.noise_intensity = noise_intensity

    @property
    def name(self):
        """Motion model name"""
        return f"ca_{self.noise_intensity}"
    
    def F(self, dt):
        dt2 = dt**2
        return np.array([[1, 0, 0, dt, 0, 0, dt2, 0, 0],
                         [0, 1, 0, 0, dt, 0, 0, dt2, 0],
                         [0, 0, 1, 0, 0, dt, 0, 0, dt2],
                         [0, 0, 0, 1, 0, 0, dt, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, dt, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def Q(self, dt: float = None):
        return np.eye(self.state_dim) * self.noise_intensity**2

def constant_acceleration_models(*args) -> List[ConstantAccelerationModel]:
    return [ConstantAccelerationModel(noise_intensity) for noise_intensity in args]
