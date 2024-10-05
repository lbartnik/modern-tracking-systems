import numpy as np
from numpy.typing import ArrayLike


def as_column(x: ArrayLike) -> np.ndarray:
    x = np.array(x)
    if len(x.shape) == 1:
        # atleast_2d turns x into a row vector
        return np.atleast_2d(x).T
    if len(x.shape) == 2:
        if x.shape[0] == 1:
            return x.T
        elif x.shape[1] == 1:
            return x
    raise Exception(f'Input has shape {x.shape} and is not a vector')
