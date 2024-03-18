import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import UserList
from typing import List, Tuple

from .evaluation import EvaluationResult, EvaluationResultList, GroupedEvaluationResultList, rmse, state_residuals
from .util import to_df, colorscale


class Figures(UserList):
    def __repr__(self):
        for f in self.data:
            f.show()
        return ''

    def update_layout(self, **kwargs):
        for f in self.data:
            f.update_layout(**kwargs)
        return self


def boxplot_rmse(results: List[EvaluationResult]) -> go.Figure:
    data = rmse(results)
    fig=ex.box(data, y='rmse', x='z_sigma', color='motion', log_y=True)
    fig.update_xaxes(type='category')
    return fig


def task_names(results: List[EvaluationResult]) -> List[str]:
    pass


def plot_position_errors(result: EvaluationResult) -> Figures:
    data = state_residuals(result)
    return Figures([ex.scatter(data, x='time', y='diff', facet_row='dim'),
                    ex.histogram(data, x="diff", facet_row='dim')])


def aggregated_errors(results: EvaluationResultList):
    # aggregate across seeds
    parameters = set(dict(results[0]).keys())
    parameters.discard('seed')

    partial_results = []

    for group in results.group(parameters):
        errs = []
        for result in group:
            err = result.truth[:,:3] - result.x_hat[:,:3]
            err = np.sqrt(np.power(err, 2).sum(axis=1))
            errs.append(err)
        
        error_data_frame = to_df(np.array(errs).T.mean(axis=1), columns=['error'])
        error_data_frame['time'] = np.arange(len(error_data_frame))

        # fill out the rest of columns with values of task parameters
        for parameter in parameters:
            error_data_frame[parameter] = getattr(group[0], parameter)
        
        partial_results.append(error_data_frame)

    return pd.concat(partial_results, ignore_index=True)


def plot_aggregated_errors(results: EvaluationResultList):
    errors = aggregated_errors(results)
    
    in_time = ex.line(errors, x='time', y='error', color='motion', facet_col='z_sigma', facet_row='target')
    in_time.update_yaxes(matches=None, showticklabels=True)
    in_time.update_layout(width=1100, height=900)

    errors['error'] = np.log10(errors['error'])

    hist = ex.histogram(errors, x='error', color='motion', facet_col='z_sigma', facet_row='target', nbins=50)
    hist.update_xaxes(matches=None, showticklabels=True)
    hist.update_layout(width=1100, height=900)
    hist.update_traces(bingroup=None)
    hist.update_traces(opacity=.4)

    return Figures([in_time, hist])



def plot_error_band(grouped: GroupedEvaluationResultList, abs: bool = False) -> go.Figure:
    assert type(grouped) == GroupedEvaluationResultList

    subplot_titles = []
    for group in grouped:
        subplot_titles.append(' '.join([str(x) for x in dict(group[0]).values()]))

    fig = make_subplots(len(grouped), 2, shared_yaxes=False,
                        subplot_titles=[x for t in subplot_titles for x in [t, '']])

    col_line = colorscale(n=len(grouped))
    col_fill = colorscale(n=len(grouped), alpha=.3)

    for i, (title, group) in enumerate(zip(subplot_titles, grouped)):
        pos, vel = [], []
        for result in group:
            pos.append(result.x_hat[:,0] - result.truth[:,0])
            vel.append(result.x_hat[:,3] - result.truth[:,3])

        pos = np.array(pos).T
        vel = np.array(vel).T
        time = np.arange(pos.shape[0])

        # this results in a plot resembling plots depicting Euclidean distance in 3D
        if abs:
            pos = np.abs(pos)
            vel = np.abs(vel)
        
        fig.add_trace(go.Scatter(name=title, legendgroup=title, x=time, y=pos.mean(axis=1), mode='lines', line=dict(color=col_line[i]),
                                 showlegend=False), row=i+1, col=1)
        fig.add_trace(go.Scatter(legendgroup=title, x=time, y=pos.max(axis=1), mode='lines', line=dict(width=0), showlegend=False),
                      row=i+1, col=1)
        fig.add_trace(go.Scatter(legendgroup=title, x=time, y=pos.min(axis=1), mode='lines', line=dict(width=0), showlegend=False,
                                     fillcolor=col_fill[i], fill='tonexty'), row=i+1, col=1)
    
        fig.add_trace(go.Scatter(name=title, legendgroup=title, x=time, y=vel.mean(axis=1), mode='lines', line=dict(color=col_line[i]),
                                 showlegend=False), row=i+1, col=2)
        fig.add_trace(go.Scatter(legendgroup=title, x=time, y=vel.max(axis=1), mode='lines', line=dict(width=0), showlegend=False),
                      row=i+1, col=2)
        fig.add_trace(go.Scatter(legendgroup=title, x=time, y=vel.min(axis=1), mode='lines', line=dict(width=0), showlegend=False,
                                     fillcolor=col_fill[i], fill='tonexty'), row=i+1, col=2)
    
    return fig
