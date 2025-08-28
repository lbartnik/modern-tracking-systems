import numpy as np
from math import log, pi

from .interface import Tracker, Track
from ..kalman import linear_ncv


class SingleTargetTracker(Tracker):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.time = None
        self.first_meas = None
        self.kf = None
        self.llr = None

        # LLR parameters
        self.P_d = 1  # probability of detection
        self.B_NT = 1 # probability of new target
        self.B_FT = 1 # probability of false target
    
    def add_measurements(self, ms):
        assert len(ms) == 1, "more than 1 measurement not supported"
        dim = len(ms[0].z.squeeze())
        
        assert dim == 3, "only 3D spaces are supported"
        m = ms[0]

        if self.time is None:
            self.time = m.time

        assert m.time >= self.time, "measurement in the past"

        if self.first_meas is None:
            self.first_meas = m
        
        elif self.track is None:
            x, P = _initialize_velocity(self.first_meas, m)
            self.kf = linear_ncv(noise_intensity=1)
            self.kf.initialize(x, P)
            
            # initial LLR: new target appears
            self.llr = log(self.P_d * self.B_NT / self.B_FT)

        else:
            self.kf.predict(m.time - self.time)
            self.kf.prepare_update(m.z, m.R)
            self.kf.update()

            # hit: measurement associated
            self.llr += log(self.P_d / self.B_FT) - dim/2*log(2*pi) - .5 * np.linalg.det(self.kf.S) \
                        - .5 * self.kf.innovation.T @ self.kf.S @ self.kf.innovation
    
            # not handling LLR for miss (measurement not associated)

    def estimate_tracks(self, t: float):
        if self.kf is None:
            return []
    
        assert t == self.time, "cannot estimate tracks for arbitrary t"
        return [Track(0, self.kf.x_hat, self.kf.P_hat)]



def _initialize_velocity(m0, m1):
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