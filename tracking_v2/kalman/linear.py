import numpy as np
from numpy.typing import ArrayLike

from .interface import KalmanFilter
from ..motion import MotionModel
from ..np import as_column


__all__ = ['LinearKalmanFilter']


class LinearKalmanFilter(KalmanFilter):
    def __init__(self, motion_model: MotionModel, H: ArrayLike):
        """Initialize a linear Kalman Filter. State dimensionality is derived from the motion
        model's `state_dim`. State mean is initialized as a vector of zeros while covariance
        is initialized as an identity matrix.

        Args:
            motion_model (MotionModel): Motion model, used to derive state transition matrix
                                        F and motion noise matrix Q.
            H (ArrayLike): Measurement matrix.
        """
        self.motion_model = motion_model
        self.H = np.atleast_2d(H)
        
        assert self.H.shape[1] == self.motion_model.state_dim

        self.x_hat = as_column(np.zeros(self.motion_model.state_dim))
        self.P_hat = np.eye(self.motion_model.state_dim)

        self.innovation = None
        self.S = None

    def initialize(self, x: ArrayLike, P: ArrayLike):
        """Initialize the Kalman Filter state. If `x` has N elements, the shape of `P` must
        be (N, N). N must be equal or less to the number of state dimensions. If `N` is less
        than the number of state dimensions, only the N first values of the mean and the
        top-left NxN corner of the covariance matrix are replaced.

        Args:
            x (ArrayLike): N new state mean values, where N is less of equal to the number
                           of state dimensions.
            P (ArrayLike): NxN new state covariance values. N must match the number of elements
                           in `x`.
        """
        x, P = np.array(x).squeeze(), np.array(P)
        r, c = P.shape

        assert len(x) <= len(self.x_hat)
        assert r <= self.P_hat.shape[0]
        assert c <= self.P_hat.shape[1]

        self.x_hat[:len(x), 0] = x
        self.P_hat[:r, :c] = P
    
    # extrapolate state and uncertainty
    def predict(self, dt: float):
        """Forward-predict the mean and covariance given the time delta `dt` and the
        configured motion model.

        Args:
            dt (float): Time delta, greater than zero.
        """
        assert dt > 0

        F = self.motion_model.F(dt)
        Q = self.motion_model.Q(dt)
        
        self.x_hat = F @ self.x_hat
        self.P_hat = F @ self.P_hat @ F.T + Q

    # update state with a measurement
    def update(self, z: ArrayLike, R: ArrayLike):
        """Update the Kalman Filter state with a measurement.

        Args:
            z (ArrayLike): Measurement of size M. M must match the dimensionality of the
                           measurement matrix.
            R (ArrayLike): Measurement error covariance matrix. Must be of shape (M, M),
                           where M is the size of `z`.
        """
        z, R = np.array(z).squeeze(), np.array(R).squeeze()

        assert len(z) == self.H.shape[0]
        assert R.shape == (len(z), len(z))

        # innovation covariance
        # S = H*P*H + R
        S = self.H @ self.P_hat @ self.H.T + R

        # Kalman gain
        # K = P*H (H*P*H + R)^-1
        K = self.P_hat @ self.H.T @ np.linalg.inv(S)

        innovation = as_column(z) - self.H @ self.x_hat

        # filtered state (mean)
        # X = X + K(z - H*X)
        x = self.x_hat + K @ innovation
        
        # filtered state (covariance)
        # P = P - K*S*K
        P = self.P_hat - K @ S @ K.T

        self.x_hat = x
        self.P_hat = P
        self.innovation = innovation
        self.S = S
