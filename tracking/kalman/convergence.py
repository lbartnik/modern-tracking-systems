import numpy as np
from numpy.typing import ArrayLike
from typing import Union

import sympy.core.numbers as scn
from sympy.solvers import solve, solveset, nsolve, nonlinsolve
from sympy import Symbol, symbols, N
from sympy.matrices import Matrix


def kalman_gain_pv(R: Union[float, ArrayLike], Q: Union[float, ArrayLike], t: float = 1, numeric: bool = False) -> np.ndarray:
    # measurement noise covariance, diagonal with equal variances or as provided
    R = _as_array(R, np.eye(1))

    # process noise covariance, matching continuous white noise acceleration (CWNA)
    # model or as provided
    Q = _as_array(Q, np.array([[t**3/3, t**2/2], [t**2/2, t]]))

    K_num = kalman_gain_pv_numeric(R, Q, t)
    if bool(numeric):
        return K_num
    
    # derive algebraically and make sure that it is close to values derived numerically
    K_alg = kalman_gain_pv_algebraic(R, Q, t)
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


def kalman_gain_pv_numeric(R: ArrayLike, Q: ArrayLike, t: float = 1, n: int = 300) -> np.ndarray:
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


def kalman_gain_pv_algebraic(R: ArrayLike, Q: ArrayLike, t: float = 1, n: int = 100) -> np.ndarray:
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
