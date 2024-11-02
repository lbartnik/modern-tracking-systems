import numpy as np
from numpy.typing import ArrayLike

from .interface import KalmanFilter
from ..np import as_column


__all__ = ['CoordinatedTurn']


# nearly coordinated turn; "Estimation with Applications to Tracking
# and Navigation", pp. 467-470
class CoordinatedTurn(KalmanFilter):
    def __init__(self, H: ArrayLike, Q_sigmas: ArrayLike):
        """Initialize a Coordinated-Turn Kalman Filter.

        Args:
            H (ArrayLike): Measurement matrix.
            Q_sigmas (ArrayLike): Two-element vector of standard deviations for the process noise
                                  matrix. The first element is the spatial standard deviation, the
                                  second element is the standard deviation for turn rate.
        """
        Q_sigmas = np.array(Q_sigmas).squeeze()
        assert len(Q_sigmas) == 2

        self.x_hat = np.zeros((5, 1))
        self.P_hat = np.eye(5)
        self.H = np.atleast_2d(H)
        self.epsilon = 1e-6
        self.Q_sigmas = Q_sigmas

        self.K = None

    def f(self, dt: float) -> np.ndarray:
        # state prediction matrix, eq. 11.7.1-4, p. 468
        #
        # except: state is defined as [x, y, x_dot, y_dot, omega] rather than
        # [x, x_dot, y, y_dot, omega]
        Omega = self.x_hat[4, 0]

        if np.abs(Omega) < self.epsilon:
            return np.array([
                [1, dt, 0, 0,  0],
                [0, 1,  0, 0,  0],
                [0, 0,  1, dt, 0],
                [0, 0,  0, 1,  0],
                [0, 0,  0, 0,  1]
            ])
        else:
            ST = np.sin(Omega * dt)
            CT = np.cos(Omega * dt)

            return np.array([
                [1, ST/Omega,     0, -(1-CT)/Omega, 0],
                [0, CT,           0, -ST,           0],
                [0, (1-CT)/Omega, 1, ST/Omega,      0],
                [0, ST,           0, CT,            0],
                [0, 0,            0, 0,             1]
            ])
    

    def f_x(self, dt: float) -> np.ndarray:
        # state prediction covariance matrix
        #
        # except: state is defined as [x, y, x_dot, y_dot, omega] rather than
        # [x, x_dot, y, y_dot, omega]
        _, x_dot, _, y_dot, Omega = self.x_hat[:, 0]

        # for |Omega| < epsilon, eq. (11.7.2-7), p. 470
        if np.abs(Omega) < self.epsilon:
            T  = dt
            T2 = dt*dt
            return np.array([
                [1, T, 0, 0, -0.5 * T2 * y_dot],
                [0, 1, 0, 0, -T * y_dot],
                [0, 0, 1, T,  0.5 * T2 * x_dot],
                [0, 0, 0, 1,  T * x_dot],
                [0, 0, 0, 0,  1]
            ])

        # |Omega| > epsilon, eq. (11.7.2-3) and (11.7.2-4), p. 469
        else:
            ST = np.sin(Omega * dt)
            CT = np.cos(Omega * dt)
            XT = x_dot * dt
            YT = y_dot * dt
            O2 = Omega*Omega

            f_Omega_1 = (CT * XT)/Omega - \
                        (ST * x_dot)/O2 - \
                        (ST * YT)/Omega - \
                        ((-1 + CT)*y_dot)/O2
            
            f_Omega_2 = -ST*XT - CT*YT

            f_Omega_3 = (ST * XT)/Omega - \
                        ((1-CT)*x_dot)/O2 + \
                        (CT*YT)/Omega - \
                        (ST*y_dot)/O2
            
            f_Omega_4 = CT*XT - ST*YT

            return np.array([
                [1, ST/Omega,     0, -(1-CT)/Omega, f_Omega_1],
                [0, CT,           0, -ST,           f_Omega_2],
                [0, (1-CT)/Omega, 1, ST/Omega,      f_Omega_3],
                [0, ST,           0, CT,            f_Omega_4],
                [0, 0,            0, 0,             1]
            ])


    def Q(self, dt: float) -> np.ndarray:
        # nearly coordinated turn; "Estimation with Applications to Tracking
        # and Navigation", eq. (11.7.1-4), pp. 467-477
        #
        # except: state is defined as [x, y, x_dot, y_dot, omega] rather than
        # [x, x_dot, y, y_dot, omega]
        var_spatial = self.Q_sigmas[0] ** 2
        var_turn    =  self.Q_sigmas[1] ** 2

        dt2 = dt*dt / 2
        return np.array([
            [dt2*var_spatial, 0,   0,   0,  0],
            [0,   dt*var_spatial,  0,   0,  0],
            [0,   0,   dt2*var_spatial, 0,  0],
            [0,   0,   0,   dt*var_spatial, 0],
            [0,   0,   0,   0,    dt*var_turn]
        ])

    def initialize(self, x: ArrayLike, P: ArrayLike):
        x, P = np.array(x).squeeze(), np.array(P)
        r, c = P.shape

        assert len(x) <= len(self.x_hat)
        assert r <= self.P_hat.shape[0]
        assert c <= self.P_hat.shape[1]

        self.x_hat[:len(x), 0] = x
        self.P_hat[:r, :c] = P
    
    def predict(self, dt: float):
        assert dt > 0

        f   = self.f(dt)
        f_x = self.f_x(dt)
        Q   = self.Q(dt)
        
        self.x_hat = f @ self.x_hat
        self.P_hat = f_x @ self.P_hat @ f_x.T + Q

    def update(self, z: ArrayLike, R: ArrayLike):
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
        self.K = K
