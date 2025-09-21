import numpy as np
from numpy.typing import ArrayLike

from .interface import KalmanFilter


__all__ = ['DelegatingFilter', 'GatedFilter']


class DelegatingFilter(KalmanFilter):
    """Filter """
    def __init__(self, delegate: KalmanFilter):
        self.delegate = delegate
    
    @property
    def x_hat(self) -> np.ndarray:
        return self.delegate.x_hat
    
    @property
    def P_hat(self) -> np.ndarray:
        return self.delegate.P_hat
    
    @property
    def innovation(self) -> np.ndarray:
        return self.delegate.innovation
    
    @property
    def S(self) -> np.ndarray:
        return self.delegate.S
    
    @property
    def state_dim(self) -> np.ndarray:
        return self.delegate.state_dim
    
    @property
    def space_dim(self) -> np.ndarray:
        return self.delegate.space_dim
    
    @property
    def update_prepared(self) -> np.ndarray:
        return self.delegate.update_prepared
    
    def reset(self):
        return self.delegate.reset()

    def initialize(self, x: ArrayLike, P: ArrayLike):
        return self.delegate.initialize(x, P)

    def predict(self, dt: float):
        return self.delegate.predict(dt)
    
    def calculate_innvovation(self, z: ArrayLike, R: ArrayLike):
        return self.delegate.calculate_innvovation(z, R)

    def update(self):
        return self.delegate.update()

    def nees(self):
        return self.delegate.nees()

    def mahalanobis_distance(self):
        return self.delegate.reset()



class GatedFilter(DelegatingFilter):
    def __init__(self, delegate: KalmanFilter, max_chi_sq: float, gate_delay: int = 50):
        self.delegate = delegate
        self.max_chi_sq = max_chi_sq
        self.gate_delay = gate_delay
        self.reset()

    def reset(self):
        self.i = 0
        self.gates = []
        self.delegate.reset()

    def update(self):
        assert self.update_prepared, "calculate_innvovation() not called"
        self.i += 1

        nis = self.nis()
        update = bool(self.i <= self.gate_delay or nis <= self.max_chi_sq)

        self.gates.append((nis, update))

        if update:
            return self.delegate.update()
