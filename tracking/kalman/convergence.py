import numpy as np
from numpy.typing import ArrayLike
from typing import Union

import sympy.core.numbers as scn
from sympy.solvers import solve, solveset, nsolve, nonlinsolve
from sympy import Symbol, symbols, N
from sympy.matrices import Matrix

from typing import Tuple


def kalman_gain_cv(R: Union[float, ArrayLike], Q: Union[float, ArrayLike], t: float = 1, numeric: bool = False) -> np.ndarray:
    """Calculate limit values for Kalman gain using a Constant Velocity motion
    model with the given measurement error variance and process noise intensity.

    Args:
        R (Union[float, ArrayLike]): measurement noise variance
        Q (Union[float, ArrayLike]): scalar process noise intensity or process noise covariance matrix.
    
    Returns:
        np.ndarray: Kalman gain array.
    """
    # measurement noise covariance, diagonal with equal variances or as provided
    R = _as_array(R, np.eye(1))

    # process noise covariance, matching continuous white noise acceleration (CWNA)
    # model or as provided
    Q = _as_array(Q, np.array([[t**3/3, t**2/2], [t**2/2, t]]))

    K_num = kalman_gain_cv_numeric(R, Q, t)
    if bool(numeric):
        return K_num
    
    # derive algebraically and make sure that it is close to values derived numerically
    K_alg = kalman_gain_cv_algebraic(R, Q, t)
    assert np.allclose(K_num, K_alg, rtol=1e-3, atol=1e-6), K_num-K_alg
    return K_alg


def _as_array(x: Union[float, ArrayLike], a: ArrayLike) -> np.ndarray:
    """Assure that x is a numpy array. If x is a scalar, return a*x. Otherwise,
    make sure that x and a are of the same shape and return x.

    Args:
        x (Union[float, ArrayLike]): Scalar or numpy array.
        a (ArrayLike): Numpy array.

    Returns:
        np.ndarray: Array of the same shape as a, derived as described above.
    """
    a = np.array(a)
    if isinstance(x, (int, float)):
        return a * float(x)
    else:
        x = np.array(x)
        assert x.shape == a.shape
        return x


def kalman_gain_cv_numeric(R: ArrayLike, Q: ArrayLike, t: float = 1, n: int = 300) -> np.ndarray:
    """Derive the limit Kalman gain numerically.

    Args:
        R (ArrayLike): Measurement noise covariance.
        Q (ArrayLike): Process noise covariance.
        t (float, optional): Sampling rate. Defaults to 1.
        n (int, optional): Number of iterations. Defaults to 100.

    Returns:
        ArrayLike: Limit Kalman gain matrix.
    """
    R = np.array(R); R.shape = (1, 1)
    Q = np.array(Q); Q.shape = (2, 2)

    H = np.array([[1, 0]])
    F = np.array([[1, t], [0, 1]])

    P = Q
    for _ in range(n):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
    
        P = P - K @ S @ K.T
    
    return K


def kalman_gain_cv_algebraic(R: ArrayLike, Q: ArrayLike, t: float = 1, n: int = 100) -> np.ndarray:
    Px, Pv, Pxv = symbols("P_x, P_v, P_xv", positive=True, real=True)
    
    # define constants
    F = Matrix([[1, t], [0, 1]])
    H = Matrix([[1, 0]])
    Q = Matrix(Q)
    R = Matrix(R)

    # define equations
    P = Matrix([[Px, Pxv], [Pxv, Pv]])
    Pn = F*P*F.T + Q
    S = H*Pn*H.T + R
    K = Pn*H.T*S.inv()

    # solve for variance
    X = P - Pn + K*S*K.T
    res = nonlinsolve([X[0,0], X[0, 1], X[1, 1]], [Px, Pv, Pxv])

    # find real-numbered solution consisting of positive numbers and return it as floats
    for solution in res:
        if real_and_positive(solution):
            a, b, c = [s.evalf() for s in solution]
            subs = {Px: a, Pv: b, Pxv: c}
            return np.asarray(K.evalf(subs=subs), dtype=float)
    
    raise Exception("No real-numbered positive solution found")

def _is_imaginary(x):
    if isinstance(x, scn.ImaginaryUnit):
        return True
    if isinstance(x, scn.Float):
        return x.is_imaginary
    return any([_is_imaginary(y) for y in x.args])


def real_and_positive(solution):
    if any([_is_imaginary(s) for s in solution]):
        return False
    if any([s.evalf() < 0 for s in solution]):
        return False
    return True


def cv_estimator_variances(Kx: float, Kv: float, R: float=1, n: int=100, t: float=1) -> Tuple[float, float, float]:
    """Calculate the variance of position and velocity estimators in the steady-state,
    given the limit values of Kalman gain (also calculated for the steady-state).

    Args:
        Kx (float): Kalman gain value for the position estimator.
        Kv (float): Kalman gain value for the velocity estimator.
        R (float): Measurement noise variance.
        n (int, optional): Number of iterations. Defaults to 100.
        t (float, optional): Sampling rate.

    Returns:
        Tuple: Limit values for estimator variances and their covariance.
    """
    Vx, Vv, Cov = Kx**2 * R, Kv**2 * R, Kx*Kv*R
    
    for _ in range(n):
        nVx = (1-Kx)**2 * (Vx + t**2 * Vv + 2*t*Cov) + Kx**2 * R
        
        nVv = (1-t*Kv)**2 * Vv + \
              Kv**2 * Vx \
              - 2*(1-t*Kv)*Kv*Cov + \
              Kv**2 * R
        
        Cov = Cov*(1-Kx)*(1-2*Kv*t) + \
              t*(1-Kx)*(1-Kv*t)*Vv \
              - (1-Kx)*Kv*Vx + \
              Kx*Kv*R
                
        Vx = nVx
        Vv = nVv
    return Vx, Vv, Cov

def cv_error_variances(Kx: float, Kv: float, R: float) -> Tuple[float, float]:
    """Calculate the limit values of variances of position and velocity errors.
    Those differ from estimator variances because we calculate prediction errors
    after forward-prediction (after applying the state transition matrix).

    Args:
        Kx (float): Kalman gain value for the position estimator.
        Kv (float): Kalman gain value for the velocity estimator.
        R (float): Measurement noise variance.
    
    Returns:
        Tuple: predicted, steady-state variances for position and velocity errors.
    """
    Vx, Vv, Cov = cv_estimator_variances(Kx, Kv, R)
    
    # position error is derived from:
    #    x_hat_{i+1,i} = x_hat_{i,i} + v_hat_{i,i}*t
    #    v_hat_{i+1,i} = v_hat_{i,i}
    return Vx+Vv+2*Cov, Vv

