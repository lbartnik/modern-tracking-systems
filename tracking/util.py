import numpy as np
import os
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from typing import Any, Dict, List, Iterable, Union
from IPython.display import Image


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

def as_png(fig: go.Figure, width: int = None, height: int = None, images_path: str = 'images') -> Image:
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    
    if fig.layout.title.text is not None and len(str(fig.layout.title.text)) > 0:
        file_name = str(fig.layout.title.text) + '.png'
    else:
        file_name = str(abs(hash(str(fig)))) + '.png'

    if fig.layout.width is None:
        if width is not None:
            fig.update_layout(width=width)
        else:
            # default width; without it, PNGs are by default too narrow
            fig.update_layout(width=1100)
    
    if fig.layout.height is None and height is not None:
        fig.update_layout(height=height)    

    file_path = os.path.join(images_path, file_name)
    fig.write_image(file_path)
    return Image(filename=file_path)
