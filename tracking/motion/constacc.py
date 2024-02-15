import numpy as np


class ConstantAccelerationModel:
    def __init__(self, sigma: float = 1) -> None:
        self.spatial_dim = 3    # number of spatial dimensions
        self.state_dim = 9      # size of the state
        self.sigma = sigma

    @property
    def name(self):
        """Motion model name"""
        return f"ca_{self.sigma}"
    
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
        return np.eye(self.state_dim) * self.sigma
