import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from typing import List
import pandas as pd
import inspect

import plotly.express as ex
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .util import SubFigure, to_df, colorscale


__all__ = ['Runner', 'run_one', 'run_many', 'before_one', 'after_one', 'before_many', 'after_many',
           'after_update', 'evaluate_nees', 'evaluate_many', 'plot_nscore', 'plot_error', 'plot_2d', 'plot_3d']



class Runner:
    def __init__(self, target, sensor, kf):
        self.target = target
        self.sensor = sensor
        self.kf = kf

        self.truth = None
        self.n = None
        self.m = None
        self.seeds = None

        self.one_x_hat, self.one_P_hat = [], []
        self.many_x_hat, self.many_P_hat = [], []
        self.one_truth, self.many_truth = [], []
        # innovation and its covariance
        self.one_v, self.one_S = [], []
        self.many_v, self.many_S = [], []

        self.dim = 3

    def after_predict(self):
        self.one_x_hat.append(np.copy(self.kf.x_hat))
        self.one_P_hat.append(np.copy(self.kf.P_hat))
    
    def after_initialize(self):
        pass

    def after_update(self, m):
        self.one_v.append(np.copy(self.kf.innovation))
        self.one_S.append(np.copy(self.kf.S))

        self.__execute_user_callbacks('after_update', m)

    def before_one(self):
        self.one_x_hat = []
        self.one_P_hat = []
        self.one_v = []
        self.one_S = []

        self.__execute_user_callbacks('before_one')

    def after_one(self):
        self.one_x_hat = np.array(self.one_x_hat)
        self.one_P_hat = np.array(self.one_P_hat)

        assert self.one_x_hat.shape[0] == self.n, f"{self.one_x_hat.shape[0]} != {self.n}"
        assert self.one_P_hat.shape[0] == self.n, f"{self.one_P_hat.shape[0]} != {self.n}"

        self.many_x_hat.append(self.one_x_hat)
        self.many_P_hat.append(self.one_P_hat)

        self.one_v = np.array(self.one_v)
        self.one_S = np.array(self.one_S)

        assert self.one_v.shape[0] == self.n, f"{self.one_v.shape[0]} != {self.n}"
        assert self.one_S.shape[0] == self.n, f"{self.one_S.shape[0]} != {self.n}"

        self.many_v.append(self.one_v)
        self.many_S.append(self.one_S)

        self.one_truth = np.copy(self.truth)
        self.many_truth.append(self.one_truth)

        self.__execute_user_callbacks('after_one')

    def before_many(self):
        self.many_x_hat = []
        self.many_P_hat = []
        self.many_truth = []
        self.many_v = []
        self.many_S = []

        self.__execute_user_callbacks('before_many')

    def after_many(self):
        self.many_x_hat = np.array(self.many_x_hat)
        self.many_P_hat = np.array(self.many_P_hat)
        self.many_truth = np.array(self.many_truth)
        self.many_v = np.array(self.many_v)
        self.many_S = np.array(self.many_S)

        assert self.many_x_hat.shape[0] == self.m
        assert self.many_P_hat.shape[0] == self.m
        assert self.many_truth.shape[0] == self.m
        assert self.many_v.shape[0] == self.m
        assert self.many_S.shape[0] == self.m

        self.__execute_user_callbacks('after_many')

    def __execute_user_callbacks(self, stage, *args):
        for name, member in inspect.getmembers(self, inspect.ismethod):
            if hasattr(member, 'runner_callback') and member.runner_callback == stage:
                member(*args)


    def run_one(self, n: int, T: float = 1):
        t = 0
        self.n = n
        
        self.before_one()
        self.truth = self.target.true_states(T, n+1)
        
        m = self.sensor.generate_measurement(t, self.truth[0, :self.dim])

        self.kf.reset()
        self.kf.initialize(m.z, m.R)

        self.after_initialize()

        for position in self.truth[1:, :self.dim]:
            t += T

            self.kf.predict(T)

            self.after_predict()

            m = self.sensor.generate_measurement(t, position)
            self.kf.prepare_update(m.z, m.R)
            self.kf.update()

            self.after_update(m)

        self.after_one()



    def run_many(self, m: int, n: int, T: float = 1, seeds=None):
        if seeds is None:
            seeds = np.arange(m)
        assert m == len(seeds)
        
        self.n = n
        self.m = m
        self.seeds = seeds

        self.before_many()
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            self.sensor.reset_rng(rng)
            self.target.reset_rng(rng)

            self.run_one(n, T)
        
        self.after_many()


def run_one(n, target, sensor, kf):
    runner = Runner(target, sensor, kf)
    runner.run_one(n)
    return runner.one_x_hat, runner.one_P_hat, runner.truth[1:,:]


def run_many(m, n, target, sensor, kf, seeds=None):
    runner = Runner(target, sensor, kf)
    runner.run_many(m, n, seeds)
    return runner.many_x_hat, runner.many_P_hat, runner.truth[1:,:]


def before_one(method):
    method.runner_callback = 'before_one'
    return method


def after_one(method):
    method.runner_callback = 'after_one'
    return method


def before_many(method):
    method.runner_callback = 'before_many'
    return method


def after_many(method):
    method.runner_callback = 'after_many'
    return method


def after_update(method):
    method.runner_callback = 'after_update'
    return method



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



















class NScoreEvaluationResult:
    def __init__(self, scores: ArrayLike, dim: int, type: str):
        self.scores = scores
        self.dim = dim
        self.type = type


def _resolve_truth_shape(truth, x_hat):
    assert len(truth.shape) in [2, 3, 4]
        
    if len(truth.shape) == 2:
        assert truth.shape[0] == x_hat.shape[1]
        assert truth.shape[1] == x_hat.shape[2]
        truth = truth.reshape((1, truth.shape[0], truth.shape[1], 1))

    elif len(truth.shape) == 3:
        if truth.shape[-1] == 1:
            assert truth.shape[0] == x_hat.shape[1]
            assert truth.shape[1] == x_hat.shape[2]
            assert truth.shape[2] == x_hat.shape[3]
            truth = truth.reshape((1, truth.shape[0], truth.shape[1], truth.shape[2]))

        else:
            assert truth.shape[0] == x_hat.shape[0] or truth.shape[0] == 1 
            assert truth.shape[1] == x_hat.shape[1]
            assert truth.shape[2] == x_hat.shape[2]
            truth = truth.reshape((truth.shape[0], truth.shape[1], truth.shape[2], 1))

    return truth

def evaluate_nees(x_hat, P_hat, truth):
    # if data comes from single run, add a dimension in the front
    if len(x_hat.shape) == 3 and len(P_hat.shape) == 3:
        x_hat = np.expand_dims(x_hat, 0)
        P_hat = np.expand_dims(P_hat, 0)

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(x_hat.shape) == 4
    assert len(P_hat.shape) == 4

    truth = _resolve_truth_shape(truth, x_hat)
    dim   = truth.shape[-2]
    
    diff  = x_hat - truth
    P_inv = np.linalg.inv(P_hat)
    
    scores = np.matmul(np.matmul(diff.transpose(0, 1, 3, 2), P_inv), diff).squeeze()
    scores = np.atleast_2d(scores)
    
    return NScoreEvaluationResult(scores, dim, 'NEES')


def evaluate_error(x_hat: np.ndarray, truth: np.ndarray) -> np.ndarray:
    # if data comes from single run, add a dimension in the front
    if len(x_hat.shape) == 3:
        x_hat = np.expand_dims(x_hat, 0)

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(x_hat.shape) == 4

    truth = _resolve_truth_shape(truth, x_hat)
    
    diff = x_hat - truth
    err_sq = np.matmul(diff.transpose(0, 1, 3, 2), diff)
    err_sq = np.atleast_2d(err_sq.squeeze())
    return np.sqrt(err_sq)



def evaluate_nis(v: ArrayLike, S: ArrayLike) -> NScoreEvaluationResult:    
    # if data comes from single run, add a dimension in the front
    if len(v.shape) == 3 and len(S.shape) == 3:
        v = np.expand_dims(v, 0)
        S = np.expand_dims(S, 0)
    
    dim = v.shape[-2]

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(v.shape) == 4
    assert len(S.shape) == 4
    
    S_inv = np.linalg.inv(S)
    
    scores = np.matmul(np.matmul(v.transpose(0, 1, 3, 2), S_inv), v).squeeze()
    scores = np.atleast_2d(scores)
    return NScoreEvaluationResult(scores, dim, 'NIS')



class EvaluationResult:
    def __init__(self, position_nees: NScoreEvaluationResult, velocity_nees: NScoreEvaluationResult,
                 position_nis: NScoreEvaluationResult, velocity_nis: NScoreEvaluationResult,
                 position_error: np.ndarray):
        self.position_nees = position_nees
        self.velocity_nees = velocity_nees
        self.position_nis = position_nis
        self.velocity_nis = velocity_nis
        self.position_error = position_error



def evaluate_runner(runner):
    x_hat, P_hat, truth, v, S = runner.many_x_hat[:, :, :6, :], runner.many_P_hat[:, :, :6, :6], \
                                runner.many_truth[:, 1:, :6], runner.many_v[:, :, :6, :], \
                                runner.many_S[:, :, :6, :6]

    assert len(x_hat.shape) == 4
    assert len(P_hat.shape) == 4
    assert len(truth.shape) in [2, 3]
    assert len(v.shape) == 4
    assert len(S.shape) == 4
    
    assert x_hat.shape[0] == P_hat.shape[0] # number of independent runs
    assert x_hat.shape[1] == P_hat.shape[1] # length of a single run
    assert x_hat.shape[2] == P_hat.shape[2] # number of state dimensions
    assert P_hat.shape[2] == P_hat.shape[3] # P_hat is a square matrix
    assert x_hat.shape[3] == 1              # x_hat is a column vector
    
    assert v.shape[3] == 1
    assert S.shape[3] == v.shape[2]
    assert S.shape[2] == S.shape[3]
    
    if len(truth.shape) == 2:
        truth = np.expand_dims(truth, 0)

    return EvaluationResult(
        evaluate_nees(x_hat[:,:,:3,:], P_hat[:,:,:3,:3], truth[:, :,:3]),
        evaluate_nees(x_hat[:,:,3:,:], P_hat[:,:,3:,3:], truth[:, :,3:]),
        evaluate_nis(v[:,:,:3,:], S[:,:,:3,:3]),
        evaluate_nis(v[:,:,3:,:], S[:,:,3:,3:]),
        evaluate_error(x_hat[:,:,:3,:], truth[:,:,:3])
    )


def nees_ci(runner):
    return sp.stats.chi2.ppf([0.025, 0.975], runner.m * runner.dim) / runner.m


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
