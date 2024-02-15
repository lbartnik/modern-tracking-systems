import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go
from typing import List

from .evaluation import EvaluationResult, rmse


def boxplot_rmse(results: List[EvaluationResult]) -> go.Figure:
    data = rmse(results)
    fig=ex.box(data, y='rmse', x='z_sigma', color='motion', log_y=True)
    fig.update_xaxes(type='category')
    return fig


def task_names(results: List[EvaluationResult]) -> List[str]:
    pass
