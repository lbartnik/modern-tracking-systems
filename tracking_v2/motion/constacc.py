import numpy as np
from .model import MotionModel


__all__ = ['ConstantAccelerationModel']


class ConstantAccelerationModel(MotionModel):
    def __init__(self, noise_intensity: float = 1) -> None:
        self.state_dim = 9
        self.noise_intensity = noise_intensity

    @property
    def name(self):
        """Motion model name"""
        return f"ca_{self.noise_intensity}"
    
    def F(self, dt: float):
        dt2 = dt**2 / 2
        return np.array([[1, 0, 0, dt, 0, 0, dt2, 0, 0],
                         [0, 1, 0, 0, dt, 0, 0, dt2, 0],
                         [0, 0, 1, 0, 0, dt, 0, 0, dt2],
                         [0, 0, 0, 1, 0, 0, dt, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, dt, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def Q(self, dt: float):
        # TODO verify if this is how the model is defined in "Estimation with Applications to Tracking and Navigation"
        # TODO or use process noise definitions from "Modern Tracking Systems", p. 204
        return np.eye(self.state_dim) * self.noise_intensity**2
