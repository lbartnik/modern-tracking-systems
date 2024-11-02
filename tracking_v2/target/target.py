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

    def position_at_time(self, t: float) -> np.ndarray:
        return self.true_states([t], 1)[0, :]
