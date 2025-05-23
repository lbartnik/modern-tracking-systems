import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union


__all__ = ['Target']


class Target(object):
    name: str
    spatial_dim: int
    seed: int

    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None) -> np.ndarray:
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

    def true_state(self, t: float) -> np.ndarray:
        return self.true_states()[t, :]

    def reset_seed(self, seed: int = 0):
        raise Exception(f"Target {self.__class__.__name__} does not implement reset_seed()")
    
    def reset_rng(self, rng: np.random.Generator = None):
        raise Exception(f"Target {self.__class__.__name__} does not implement reset_rng()")

    def __repr__(self):
        return self.name
