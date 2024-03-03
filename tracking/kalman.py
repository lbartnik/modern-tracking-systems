import numpy as np
from numpy.typing import ArrayLike
from .motion import ConstantAccelerationModel


__all__ = ['KalmanFilter', 'kalman_pv', 'kalman_pva']


class KalmanFilter:
    def __init__(self, state_dim: int):
        self.spatial_dim = None      # number of spatial dimensions
        self.state_dim = state_dim   # dimensionality of the state
        self.x = np.zeros(state_dim) # state (mean)
        self.P = np.eye(state_dim)   # state (covariance)
        self.H = None                # measurement matrix
        self.R = None                # measurement noise
        self.K = None                # Kalman gain
        self.motion_model = None     # motion model        

        self.x_hat = self.x          # most recent predicted state, used to evaluate prediction error
        self.P_hat = self.P          # most recent predicted state covariance

    def initialize(self, x, P):
        self.x[:len(x)] = x
        r, c = P.shape
        self.P[:r,:c] = P
    
    # extrapolate state and uncertainty
    def predict(self, dt):
        F = self.motion_model.F(dt)
        Q = self.motion_model.Q(dt)
        
        self.x_hat = F @ self.x
        self.x = np.copy(self.x_hat)
        
        self.P_hat = F @ self.P @ F.T + Q
        self.P = np.copy(self.P_hat)


    # update state with a measurement
    def update(self, z):
        z = np.array(z); z.shape = self.spatial_dim # measurement

        # innovation covariance
        # S = H*P*H + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        # K = P*H (H*P*H + R)^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # filtered state (mean)
        # X = X + K(z - H*X)
        x = self.x + K @ (z - self.H @ self.x)
        
        # filtered state (covariance)
        # P = P - K*S*K
        P = self.P - K @ S @ K.T

        self.x = x
        self.P = P
        self.K = K

# Kalman Filter with 6D state (position + velocity)
def kalman_pv(motion_model, R):

    kf = KalmanFilter(state_dim=6)

    kf.spatial_dim = 3
    kf.motion_model = motion_model

    assert motion_model.spatial_dim == kf.spatial_dim
    assert motion_model.state_dim == kf.state_dim

    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])

    if np.isscalar(R):
        kf.R = np.eye(kf.spatial_dim) * R
    else:
        kf.R = np.array(R)
        kf.R.shape = (kf.spatial_dim, kf.spatial_dim)

    return kf


# Kalman Filter with 9D state (position + velocity + acceleration)
def kalman_pva(motion_model, R):

    kf = KalmanFilter(state_dim=9)

    kf.spatial_dim = 3
    kf.motion_model = motion_model

    assert motion_model.spatial_dim == kf.spatial_dim
    assert motion_model.state_dim == kf.state_dim

    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0]])

    if np.isscalar(R):
        kf.R = np.eye(kf.spatial_dim) * R
    else:
        kf.R = np.array(R)
        kf.R.shape = (kf.spatial_dim, kf.spatial_dim)

    return kf

