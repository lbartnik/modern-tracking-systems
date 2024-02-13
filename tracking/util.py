import numpy as np
import pandas as pd


def to_df(np_array, col_prefix='x', columns=None):
    if columns is not None:
        if len(columns) != np_array.shape[1]:
            raise ValueError(f"Array has {np_array.shape[1]} columns but got {len(columns)} column names")
    else:
        columns = [col_prefix + str(i) for i in range(np_array.shape[1])]

    d = {}
    for column, name in zip(np.array(np_array).T, columns):
        d[name] = column
    
    return pd.DataFrame(d)
