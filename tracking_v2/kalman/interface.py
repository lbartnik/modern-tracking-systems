import numpy as np
from numpy.typing import ArrayLike


class KalmanFilter(object):
    x_hat: np.ndarray
    P_hat: np.ndarray
    innovation: np.ndarray
    S: np.ndarray
    state_dim: int
    space_dim: int

    def initialize(self, x: ArrayLike, P: ArrayLike):
        raise Exception(f"initialize() method not implemented in {self.__class__.__name__}")

    def predict(self, dt: float):
        raise Exception(f"predict() method not implemented in {self.__class__.__name__}")
    
    def update(self, z: ArrayLike, R: ArrayLike):
        raise Exception(f"update() method not implemented in {self.__class__.__name__}")
