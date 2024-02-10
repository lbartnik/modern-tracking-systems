import numpy as np
import pandas as pd


def to_df(np_array, col_prefix='x'):
    d = {}
    for i, column in enumerate(np.array(np_array).T):
        d[col_prefix + str(i)] = column
    return pd.DataFrame(d)
