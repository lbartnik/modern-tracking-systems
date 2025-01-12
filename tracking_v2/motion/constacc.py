import numpy as np
from .model import MotionModel


__all__ = ['ConstantAccelerationModel']


class ConstantAccelerationModel(MotionModel):
    """Continuous Wiener Process Acceleration (aka. Near-Constant-Acceleration) motion model.

    Approximates a continuous-time random process where jerk is modeled as
    (continuous-time) white noise, which means that acceleration is a (continuous-time)
    Wiener process.
    
    Described in "Estimation with Applications to Tracking and Navigation", pp. 270-272."""

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
        T  = dt
        T2 = dt*dt / 2
        T3 = dt*dt*dt / 6
        T4 = dt*dt*dt*dt / 8
        T5 = dt*dt*dt*dt*dt / 20
        
        return np.array([[T5, 0,  0,  T4, 0,  0,  T3, 0,  0 ],
                         [0,  T5, 0,  0,  T4, 0,  0,  T3, 0 ],
                         [0,  0,  T5, 0,  0,  T4, 0,  0,  T3],
                         [T4, 0,  0,  T3, 0,  0,  T2, 0,  0 ],
                         [0,  T4, 0,  0,  T3, 0,  0,  T2, 0 ],
                         [0,  0,  T4, 0,  0,  T3, 0,  0,  T2],
                         [T3, 0,  0,  T2, 0,  0,  T,  0,  0 ],
                         [0,  T3, 0,  0,  T2, 0,  0,  T,  0 ],
                         [0,  0,  T3, 0,  0,  T2, 0,  0,  T ]]) * self.noise_intensity
