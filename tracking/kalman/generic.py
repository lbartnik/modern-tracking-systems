import numpy as np
from numpy.typing import ArrayLike


__all__ = ['ExtendedKalmanFilter']



class CallbackInvoker(object):
    def __init__(self, callback):
        self.callback = callback
        self.t = 0
    
    def call_on_predict(self, x: ArrayLike, P: ArrayLike):
        if hasattr(self.callback, 'on_predict'):
            self.callback.on_predict(self.t, np.copy(x), np.copy(P))

    def call_on_update(self, x: ArrayLike, P: ArrayLike, K: ArrayLike, innovation: ArrayLike):
        self.t += 1

        if hasattr(self.callback, 'on_update'):
            self.callback.on_update(self.t, np.copy(x), np.copy(P), np.copy(K), np.copy(innovation))


class GenericKalmanFilter(object):
    """Base class for Kalman Filter implementations. Can be used to implement
    the regular (linear) Kalman Filter as well as the Extended Kalman Filter.
    
    Children classes need to implement the following methods:

      - h: measurement function
      - h_inv: the inverse of measurement function
      - H: measurement matrix
    """

    def __init__(self, state_space, measurement_space, motion_model = None, callback = None):
        if motion_model is not None:
            assert state_space.name == motion_model.space_name
            assert state_space.spatial_dim == motion_model.spatial_dim
            assert state_space.state_dim == motion_model.state_dim

        self.state_space = state_space
        self.motion_model = motion_model
        self.measurement_space = measurement_space
        self.callback = CallbackInvoker(callback) if callback is not None else None

        self.x = np.zeros((state_space.state_dim, 1)) # column vector
        self.P = np.eye(state_space.state_dim)

    def initialize(self, z: ArrayLike, R: ArrayLike):
        z, R = np.array(z).squeeze(), np.array(R).squeeze()
        r, c = R.shape

        assert r == c
        assert r == len(z)
        
        # passes a column vector to h_inv()
        self.x[:3,0] = self.h_inv(np.atleast_2d(z).T)

        # important: here we take the inverse of H
        H = self.H()[:r, :c]
        H_inv = np.linalg.inv(H)
        
        self.P[:r,:c] = H_inv @ R @ H_inv.T
    
    def predict(self, dt):
        F = self.motion_model.F(dt)
        Q = self.motion_model.Q(dt)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self.callback.call_on_predict(self.x, self.P)


    # measurement matrix
    def H(self):
        # passes a column vector to jacobian()
        J = self.measurement_space.jacobian(self.x[:3, :])
        H = np.zeros((3, self.state_space.state_dim))
        H[:3, :3] = J
        return H


    def update(self, z, R):
        z = np.array(z); z.shape = (self.measurement_space.spatial_dim, 1) # measurement, column vector
        
        # passes a column vector to h()
        z_hat = self.h(self.x[:3, :])
        z_hat.shape = (self.measurement_space.spatial_dim, 1)

        H = self.H()

        # innovation covariance
        # S = H*P*H + R
        S = H @ self.P @ H.T + R

        # Kalman gain
        # K = P*H (H*P*H + R)^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        innovation = z - z_hat
        
        # filtered state (mean)
        # X = X + K(z - H*X)
        x = self.x + K @ innovation
        
        # filtered state (covariance)
        # P = P - K*S*K
        P = self.P - K @ S @ K.T

        self.x = x
        self.P = P

        self.callback.call_on_update(self.x, self.P, K, innovation)
