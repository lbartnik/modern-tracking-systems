import numpy as np
from typing import List


class ConstantVelocityModel:
    def __init__(self, sigma: float = 1) -> None:
        self.spatial_dim = 3    # number of spatial dimensions
        self.state_dim = 6      # size of the state
        self.sigma = sigma

    @property
    def name(self):
        """Motion model name"""
        return f"cv_{self.sigma}"
    
    def F(self, dt):
        return np.array([[1, 0, 0, dt, 0, 0],
                         [0, 1, 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

    def Q(self, dt: float = None):
        return np.eye(self.state_dim) * self.sigma


def constant_velocity_models(*args) -> List[ConstantVelocityModel]:
    return [ConstantVelocityModel(sigma) for sigma in args]
