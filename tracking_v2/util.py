import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly
import os

import IPython.display as ipython
import ipywidgets.widgets as widgets

from numpy.typing import ArrayLike
from typing import Any, Dict, Iterable, List, Union


__all__ = ['SubFigure', 'to_df', 'colorscale', 'display']


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



def to_df(array: ArrayLike, col_prefix='x', columns=None, additional_columns: Dict = None, add_time: bool = False):
    array = np.asarray(array)

    if len(array.shape) == 1:
        array = np.reshape(array, (len(array), 1))

    if columns is not None:
        if len(columns) != array.shape[-1]:
            raise ValueError(f"Array has {array.shape[-1]} columns but got {len(columns)} column names")
    else:
        columns = [col_prefix + str(i) for i in range(array.shape[-1])]

    d = {}
    if add_time:
        d['t'] = np.arange(0, array.shape[0])

    for column, name in zip(array.T, columns):
        d[name] = column
    
    if additional_columns is not None:
        for name, value in additional_columns.items():
            d[name] = value

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



class DisplayProcessor:
    def __init__(self):
        self.as_png   = True
        self._count   = 0
        self._pattern = 'images/fig{count:02d}.png'

    def __call__(self, fig, as_png: bool = None):
        if (as_png is not None and not as_png) or not self.as_png:
            ipython.display(fig)
            return
        
        path = self._pattern.format(count = self._count)
        self._count += 1

        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        fig.write_image(path)
        ipython.display(ipython.Image(filename=path))


display = DisplayProcessor()


def columns(*figs):
    width = int(100 / len(figs))
    
    outputs = []
    for fig in figs:
        output = widgets.Output(layout={'width': f'{width}%'})
        with output:
            display(fig)
        outputs.append(output)
  
    column_layout = widgets.HBox(outputs)
    ipython.display(column_layout)
