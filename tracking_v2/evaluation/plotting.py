import numpy as np
import scipy as sp
from typing import List
from numpy.typing import ArrayLike
import pandas as pd

import datetime

import plotly.express as ex
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .runner import NScoreEvaluationResult, Runner, evaluate_runner
from ..util import SubFigure, to_df, colorscale


__all__ = ['plot_nscore', 'plot_error', 'plot_2d', 'plot_3d', 'plot_runs', 'plot_track']


def plot_error(runner, skip=0, run=0):
    assert runner.m == 1

    tm  = np.arange(runner.n-skip).reshape((-1, 1))
    err = runner.many_x_hat[run, skip:,:3,0] - runner.truth[(skip+1):,:3]

    df = to_df(np.concatenate((tm, err), axis=1), columns=['time', 'x', 'y', 'z'])
    df = df.melt(['time'], ['x', 'y', 'z'], 'dim', 'error')

    fig = ex.line(df, x='time', y='error', facet_row='dim')
    fig.update_yaxes(matches=None)

    return fig




def plot_2d(runner, skip=0, run=0, residual=False):
    tm = np.arange(runner.truth.shape[0])
    cl = colorscale(n=2)

    fig = make_subplots(rows=3, cols=1,)
    
    if residual:
        data = runner.many_x_hat[run, skip:,:3,0] - runner.many_truth[run, (skip+1):,:3]
        name = 'error'
    else:
        data = runner.many_x_hat[run, skip:,:,0]
        name = 'x_hat'

    fig.add_trace(go.Scatter(x=tm, y=data[:,0], name=name, legendgroup='x_hat', marker=dict(size=.1, color=cl[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=data[:,1], name=name, legendgroup='x_hat', showlegend=False, marker=dict(size=.1, color=cl[0])), row=2, col=1)
    fig.add_trace(go.Scatter(x=tm, y=data[:,2], name=name, legendgroup='x_hat', showlegend=False, marker=dict(size=.1, color=cl[0])), row=3, col=1)
    
    if not residual:
        fig.add_trace(go.Scatter(x=tm, y=runner.many_truth[run, (skip+1):,0], name='truth', legendgroup='truth', marker=dict(size=.1, color=cl[1])), row=1, col=1)
        fig.add_trace(go.Scatter(x=tm, y=runner.many_truth[run, (skip+1):,1], name='truth', legendgroup='truth', showlegend=False, marker=dict(size=.1, color=cl[1])), row=2, col=1)
        fig.add_trace(go.Scatter(x=tm, y=runner.many_truth[run, (skip+1):,2], name='truth', legendgroup='truth', showlegend=False, marker=dict(size=.1, color=cl[1])), row=3, col=1)

    fig.update_layout(
        yaxis=dict(title=dict(text='x')),
        yaxis2=dict(title=dict(text='y')),
        yaxis3=dict(title=dict(text='z')),
    )

    return fig





def plot_3d(runner, skip=0, run=0):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=runner.many_x_hat[run, skip:,0,0],
                               y=runner.many_x_hat[run, skip:,1,0],
                               z=runner.many_x_hat[run, skip:,2,0],
                               name='x_hat',
                               marker=dict(
                                   size=.1
                               )))

    fig.add_trace(go.Scatter3d(x=runner.truth[(skip+1):,0],
                               y=runner.truth[(skip+1):,1],
                               z=runner.truth[(skip+1):,2],
                               name='truth',
                               marker=dict(
                                   size=.1
                               )))

    return fig



def plot_nscore(score: NScoreEvaluationResult, skip: int=25) -> go.Figure:
    scores = score.scores[:, skip:]
    dim    = score.dim
    type   = score.type
    
    # confidence interval for the mean
    run_count = scores.shape[0]
    ci_mean = sp.stats.chi2.ppf([0.025, 0.975], run_count * dim) / run_count

    mean_score = np.mean(scores, axis=0)


    # confidence interval for individual runs
    ci_qs = sp.stats.chi2.ppf([0.025, 0.975], dim)

    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                        subplot_titles=[f'{type} vs. time', f'Mean {type}', f'All {type}'])
    tm = SubFigure(fig, 1, 1)
    h1 = SubFigure(fig, 2, 1)
    h2 = SubFigure(fig, 2, 2)


    # -- time plot
    x = np.arange(scores.shape[1])
    
    #tm.add_trace(go.Scatter(x=x, y=np.median(scores, axis=0), mode='lines', marker_color='blue', name='median'))
    tm.add_trace(go.Scatter(x=x, y=mean_score, mode='lines', marker_color='red', name='mean'))
    
    # confidence interval for the mean
    tm.add_hline(y=ci_mean[0], line_width=.5, line_dash="dash", line_color="red")
    tm.add_hline(y=ci_mean[1], line_width=.5, line_dash="dash", line_color="red")
    tm.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'), name='95% all runs', showlegend=True))

    tm.add_hline(y=ci_qs[0], line_width=.5, line_dash="dash", line_color="green")
    tm.add_hline(y=ci_qs[1], line_width=.5, line_dash="dash", line_color="green")
    tm.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'), name='95% one run', showlegend=True))

    # observed quantiles
    q025 = np.quantile(scores, .025, axis=0)
    q975 = np.quantile(scores, .975, axis=0)

    q_color = 'rgba(143,188,143, 0.5)'
    tm.add_trace(go.Scatter(x=x, y=q025, fill=None, mode='lines', marker_color=q_color, showlegend=False, legendgroup='conf_int'))
    tm.add_trace(go.Scatter(x=x, y=q975, fill='tonexty', mode='lines', fillcolor=q_color, line_color=q_color, legendgroup='conf_int', name='95% data'))
        
    
    # -- histogram of mean
    h1.add_trace(go.Histogram(x=mean_score, nbinsx=40, name='mean'))
    h1.add_vline(x=ci_mean[0], line_width=.5, line_dash="dash", line_color="red")
    h1.add_vline(x=ci_mean[1], line_width=.5, line_dash="dash", line_color="red")

    lower = np.mean(mean_score < ci_mean[0])
    upper = np.mean(ci_mean[1] < mean_score)
    center = 1 - lower - upper
    h1.add_annotation(text=f"{round(lower*100, 2)}%", xref="x domain", yref="y domain", x=0, y=1, showarrow=False)
    h1.add_annotation(text=f"{round(center*100, 2)}%", xref="x domain", yref="y domain", x=.5, y=1, showarrow=False)
    h1.add_annotation(text=f"{round(upper*100, 2)}%", xref="x domain", yref="y domain", x=1, y=1, showarrow=False)


    # -- histogram of all data
    h2.add_trace(go.Histogram(x=scores.reshape(-1), nbinsx=40, name='data'))
    h2.add_vline(x=ci_qs[0], line_width=.5, line_dash="dash", line_color="red")
    h2.add_vline(x=ci_qs[1], line_width=.5, line_dash="dash", line_color="red")

    lower = np.mean(scores.reshape(-1) < ci_qs[0])
    upper = np.mean(ci_qs[1] < scores.reshape(-1))
    center = 1 - lower - upper
    h2.add_annotation(text=f"{round(lower*100, 2)}%", xref="x domain", yref="y domain", x=-.05, y=1, showarrow=False)
    h2.add_annotation(text=f"{round(center*100, 2)}%", xref="x domain", yref="y domain", x=.2, y=1, showarrow=False)
    h2.add_annotation(text=f"{round(upper*100, 2)}%", xref="x domain", yref="y domain", x=1, y=1, showarrow=False)

    fig.update_layout(height=700)    
    return fig


def plot_runs(nees: NScoreEvaluationResult, n: int=None, skip: int=25) -> go.Figure:
    # runs x time
    assert len(nees.scores.shape) == 2
    scores = nees.scores[:, skip:]

    fig = go.Figure()

    ci_mean = sp.stats.chi2.ppf([0.025, 0.975], nees.dim)
    fig.add_hline(y=ci_mean[0], line_width=.5, line_dash="dash", line_color="red")
    fig.add_hline(y=ci_mean[1], line_width=.5, line_dash="dash", line_color="red")

    x = np.arange(scores.shape[1])
    if n is None:
        n = scores.shape[0]

    for i in range(n):
        #fig.add_trace(go.Scatter(x=x, y=scores[i,:],  name=f'run {i}', mode='markers', marker=dict(size=0.5)))
        fig.add_trace(go.Scatter(x=x, y=scores[i,:],  name=f'run {i}', mode='lines', line=dict(width=.4), opacity=0.5))

    return fig


class Tagged:
    def __init__(self, runner: Runner, tags: dict):
        self.runner = runner
        self.tags = tags


def tag(runner: Runner, **kwargs):
    return Tagged(runner, kwargs)


def plot_error_vs_nees(*tagged: List[Tagged], x: str = None, facet_row: str = None):

    for t in tagged:
        assert isinstance(t, Tagged), "Each input must be a tagged Runner"
        assert t.tags.keys() == tagged[0].tags.keys(), "All runners must have the same set of tags"
    
    parts = []
    for t in tagged:
        e = evaluate_runner(t.runner)
        
        nees = e.position_nees.scores.mean(axis=0)
        err = e.position_error.mean(axis=0)

        part = np.asarray((nees, err)).T
        part = to_df(part, columns=['nees', 'err'])

        for name, value in t.tags.items():
            part[name] = value
        
        parts.append(part)

    id_vars = [x]
    if facet_row is not None:
        id_vars.append(facet_row)

    data = pd.concat(parts)
    data = data.melt(id_vars, ['nees', 'err'], 'metric', 'value')

    fig = ex.box(data, x=x, y='value', color='metric', facet_row=facet_row)
    return fig


class StoneSoupState:
    def __init__(self, timestamp: int, mean: ArrayLike, cov: ArrayLike):
        self.timestamp = timestamp
        self.mean = mean
        self.state_vector = mean
        self.ndim = len(mean)
        self.covar = cov

class StoneSoupTrack:
    def __init__(self, id: int, states: List):
        self.id = id
        self.states = states
    
    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)
        



def plot_track(runner: Runner, m: int = 0, n: ArrayLike = None, gate: float = None) -> go.Figure:
    start = datetime.datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    if n is None:
        n = np.arange(runner.n)
        timesteps = [start + datetime.timedelta(seconds=i) for i in range(runner.n)]
    elif isinstance(n, slice):
        n = np.arange(start=n.start, stop=n.stop, step=n.step)
        timesteps = [start + datetime.timedelta(seconds=int(i)) for i in n]
    else:
        n = np.asarray(n)
        timesteps = [start + datetime.timedelta(seconds=i) for i in n]

    if gate is None:
        gate_multiplier = 1
    else:
        gate_multiplier = float(gate) * float(gate)


    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.types.detection import Detection
    from stonesoup.plotter import AnimatedPlotterly

    plotter = AnimatedPlotterly(timesteps=timesteps, tail_length=5/len(n))

    measurement_model = LinearGaussian(
        ndim_state=2,
        mapping=(0, 1),
        noise_covar=np.array([[5, 0],
                            [0, 5]])
    )

    truth = []
    measurements = []
    states = []

    for i, t in zip(n, timesteps):
        tr = StoneSoupState(timestamp=t, mean=runner.many_truth[m, i+1, :2], cov=None)
        truth.append(tr)

        d = Detection(runner.many_z[m, i+1, :2, :], timestamp=t,
                      measurement_model=measurement_model)
        measurements.append(d)

        mean = runner.many_x_hat[m, i, :2, :]
        cov = runner.many_P_hat[m, i, :2, :2] * gate_multiplier
        s = StoneSoupState(t, mean, cov)
        states.append(s)

    plotter.plot_ground_truths([truth], [0, 1])

    plotter.plot_measurements(measurements, [0, 1])

    track = StoneSoupTrack(m, states)
    plotter.plot_tracks([track], [0, 1], uncertainty=True)

    return plotter.fig
