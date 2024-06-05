import numpy as np
from typing import Callable


class ExtendedKalmanFilter(object):
    def __init__(self, coords: str, h: Callable) -> None:
        self.coords = coords
        self.h = h
        self.x = None
        self.P = None
        self.motion_model = None
    
    def initialize(self, x, P):
        self.x[:len(x),0] = x
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
        z = np.array(z); z.shape = (self.spatial_dim, 1) # measurement, column vector

        H = self.h(z, self.x)

        # innovation covariance
        # S = H*P*H + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        # K = P*H (H*P*H + R)^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # filtered state (mean)
        # X = X + K(z - H*X)
        x = self.x + K @ (z - H @ self.x)
        
        # filtered state (covariance)
        # P = P - K*S*K
        P = self.P - K @ S @ K.T

        self.x = x
        self.P = P
        self.K = K
