import numpy as np
from numpy.typing import ArrayLike
from ..motion import ConstantAccelerationModel


__all__ = ['KalmanFilter9D']


class KalmanFilter9D:
    def __init__(self, R=np.eye(3), motion_model=None):
        self.spatial_dim = 3                                          # number of spatial dimensions
        self.state_dim = 9                                            # dimensionality of the state
        self.x = np.zeros(9)                                          # state (mean)
        self.P = np.eye(9)                                            # state (covariance)
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],               # measurement matrix
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        self.x_hat = self.x                                           # most recent predicted state, used to evaluate prediction error
        
        # measurement noise
        if np.isscalar(R):
            self.R = np.eye(self.spatial_dim) * R
        else:
            self.R = np.array(R); self.R.shape = (self.spatial_dim, self.spatial_dim)

        self.motion_model = ConstantAccelerationModel() if motion_model is None else motion_model
        
        assert self.motion_model.spatial_dim == self.spatial_dim
        assert self.motion_model.state_dim == self.state_dim

    def initialize(self, x, P):
        self.x[:len(x)] = x
        r, c = P.shape
        self.P[:r,:c] = P
    
    # extrapolate state and uncertainty
    def predict(self, dt):
        F = self.motion_model.F(dt)
        Q = self.motion_model.Q(dt)
        
        self.x = F @ self.x
        self.x_hat = np.copy(self.x)
        
        self.P = F @ self.P @ F.T + Q
        self.P_hat = np.copy(self.P)

    # update state with a measurement
    def update(self, z):
        z = np.array(z); z.shape = self.spatial_dim # measurement

        # Kalman gain
        # K = P*H (H*P*H + R)
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        # filtered state (mean)
        # X = X + K(z - H*X)
        self.x = self.x + K @ (z - self.H @ self.x)

        I = np.eye(self.state_dim)
        
        # filtered state (covariance)
        # P = (I - K*H) * P * (I - K*H) + K*R*K
        self.P = (I - K@self.H) @ self.P @ (I - K@self.H).T + K @ self.R @ K.T
