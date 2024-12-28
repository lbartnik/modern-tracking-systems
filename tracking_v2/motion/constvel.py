import numpy as np
from .model import MotionModel


__all__ = ['ConstantVelocityModel']


class ConstantVelocityModel(MotionModel):
    """Near-Constant-Velocity motion model.

    Models a target moving with nearly-constant velocity and whose acceleration is
    modeled as a continuous-time, white-noise random process.
     
    Defined in "Estimation with Applications to Tracking and Navigation", p. 270
    """

    def __init__(self, noise_intensity: float = 1):
        """Initialize the CV motion model.

        Args:
            noise_intensity (float, optional): Multiplier for the process noise
                covariance matrix. Its physical dimension is [length]^2 / [time]^3.
        """
        self.state_dim = 6
        self.noise_intensity = noise_intensity

    @property
    def name(self):
        """Returns the name of the motion model"""
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
                         [0, 0, dt2, 0, 0, dt]]) * self.noise_intensity
