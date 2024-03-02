import numpy as np
import pandas as pd
import plotly.colors
from numpy.typing import ArrayLike
from typing import Any, Dict, List, Iterable, Union


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

def colorscale(x: Iterable[Any] = None, n: int = None, alpha: float = None) -> Union[List[str], Dict[Any, str]]:
    if x is not None and n is not None:
        raise Exception("Specify only one, x or n")
    
    if x is not None:
        n = len(x)
    
    n = int(n)
    # there need to be at least two colors to generate a color scale
    if n == 1:
        n = 2
    rgb = plotly.colors.sample_colorscale(plotly.colors.sequential.Jet, n)
    rgb = [name for _, name in plotly.colors.make_colorscale(rgb)]

    if alpha is not None:
        rgb = [f"rgba{name[3:-1]}, {alpha})" for name in rgb]

    return rgb if x is None else dict(zip(x, rgb))
