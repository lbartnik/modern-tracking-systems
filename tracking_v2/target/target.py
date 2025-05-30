import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union


__all__ = ['Target']


class Target(object):
    name: str
    spatial_dim: int
    seed: int
    rng: np.random.Generator
    T: float
    n: int
    cached: np.ndarray

    def cache(self, T: float = 1, n: int = 400):
        self.T = T
        self.n = n
        self.cached = self.true_states(T, n)

    def reset_seed(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def reset_rng(self, rng: np.random.Generator = None):
        self.seed = None
        self.rng = rng

    def __repr__(self):
        if self.seed is None:
            return self.name
        else:
            return f"{self.name}_{self.seed}"

    def true_state(self, t: float) -> np.ndarray:
        if self.cached is not None:
            return self.cached[t, :]
        else:
            return self.true_states()[t, :]

    def true_states(self, T: float = 1, n: int = 400) -> np.ndarray:
        """Generate target states.

        Args:
            T (Union[float, ArrayLike]): Sampling interval or array of specific timestamps.
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Raises:
            Exception: If not implemented by the subclass.

        Returns:
            np.ndarray: (n, 6) array of states.
        """
        raise Exception(f"Target {self.__class__.__name__} does not implement true_states()")
