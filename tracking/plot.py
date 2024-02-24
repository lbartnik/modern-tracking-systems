import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go
from collections import UserList
from typing import List, Tuple

from .evaluation import EvaluationResult, EvaluationResultList, rmse
from .util import to_df


class Figures(UserList):
    def __repr__(self):
        for f in self.data:
            f.show()
        return ''


def boxplot_rmse(results: List[EvaluationResult]) -> go.Figure:
    data = rmse(results)
    fig=ex.box(data, y='rmse', x='z_sigma', color='motion', log_y=True)
    fig.update_xaxes(type='category')
    return fig


def task_names(results: List[EvaluationResult]) -> List[str]:
    pass


def position_errors(result: EvaluationResult) -> Figures:
    col_names = ['x', 'y', 'z']

    data = to_df(result.truth[:,:3] - result.x_hat[:,:3], columns=col_names)
    data['time'] = np.arange(len(data))
    data = data.melt(id_vars=['time'], value_vars=col_names, var_name='dim', value_name='diff')

    return data


def plot_position_errors(result: EvaluationResult) -> Figures:
    data = position_errors(result)
    return Figures([ex.scatter(data, x='time', y='diff', facet_row='dim'),
                    ex.histogram(data, x="diff", facet_row='dim')])


def aggregated_errors(results: EvaluationResultList):
    # aggregate across seeds
    parameters = set(dict(results[0].task).keys())
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
            error_data_frame[parameter] = getattr(group[0].task, parameter)
        
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

