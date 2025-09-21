import numpy as np
from numpy.typing import ArrayLike


__all__ = ['KalmanFilter']


class KalmanFilter(object):
    x_hat: np.ndarray
    P_hat: np.ndarray
    innovation: np.ndarray
    S: np.ndarray
    state_dim: int
    space_dim: int
    update_prepared: bool

    def reset(self):
        raise Exception(f"reset() method not implemented in {self.__class__.__name__}")

    def initialize(self, x: ArrayLike, P: ArrayLike):
        raise Exception(f"initialize() method not implemented in {self.__class__.__name__}")

    def predict(self, dt: float):
        raise Exception(f"predict() method not implemented in {self.__class__.__name__}")
    
    def calculate_innvovation(self, z: ArrayLike, R: ArrayLike):
        raise Exception(f"calculate_innvovation() method not implemented in {self.__class__.__name__}")

    def update(self):
        raise Exception(f"update() method not implemented in {self.__class__.__name__}")

    def nis(self):
        return (self.innovation.T @ np.linalg.inv(self.S) @ self.innovation).squeeze()

    def mahalanobis_distance(self):
        return np.sqrt(self.nis())
