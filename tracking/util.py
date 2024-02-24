import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def to_df(array: ArrayLike, col_prefix='x', columns=None):
    array = np.array(array)

    if len(array.shape) == 1:
        array = np.reshape(array, (len(array), 1))

    if columns is not None:
        if len(columns) != array.shape[-1]:
            raise ValueError(f"Array has {array.shape[-1]} columns but got {len(columns)} column names")
    else:
        columns = [col_prefix + str(i) for i in range(array.shape[-1])]

    d = {}
    for column, name in zip(np.array(array).T, columns):
        d[name] = column
    
    return pd.DataFrame(d)
