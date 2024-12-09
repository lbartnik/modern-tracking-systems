import plotly.graph_objects as go
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from typing import Dict


class SubFigure:
    def __init__(self, fig: go.Figure, row: int, col: int):
        self.fig = fig
        self.row = row
        self.col = col
    
    def add_trace(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_trace(*args, **kwargs)
    
    def add_hline(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_hline(*args, **kwargs)
    
    def add_vline(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_vline(*args, **kwargs)

    def add_annotation(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_annotation(*args, **kwargs)



def to_df(array: ArrayLike, col_prefix='x', columns=None, additional_columns: Dict = None):
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
    
    if additional_columns is not None:
        for name, value in additional_columns.items():
            d[name] = value

    return pd.DataFrame(d)
