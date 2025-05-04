import numpy as np
import scipy as sp

import plotly.express as ex
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .util import SubFigure, to_df, colorscale


__all__ = ['Runner', 'run_one', 'run_many', 'evaluate_nees', 'evaluate_many', 'plot_nees', 'plot_error', 'plot_2d', 'plot_3d']



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

        self.dim = 3

    def after_predict(self):
        self.one_x_hat.append(np.copy(self.kf.x_hat))
        self.one_P_hat.append(np.copy(self.kf.P_hat))
    
    def after_initialize(self):
        pass

    def after_update(self, m):
        pass

    def before_one(self):
        self.one_x_hat = []
        self.one_P_hat = []

    def after_one(self):
        self.one_x_hat = np.array(self.one_x_hat)
        self.one_P_hat = np.array(self.one_P_hat)

        assert self.one_x_hat.shape[0] == self.n, f"{self.one_x_hat.shape[0]} != {self.n}"
        assert self.one_P_hat.shape[0] == self.n, f"{self.one_P_hat.shape[0]} != {self.n}"

        self.many_x_hat.append(self.one_x_hat)
        self.many_P_hat.append(self.one_P_hat)
        
        self.one_truth = np.copy(self.truth)
        self.many_truth.append(self.one_truth)

    def before_many(self):
        self.many_x_hat = []
        self.many_P_hat = []
        self.many_truth = []

    def after_many(self):
        self.many_x_hat = np.array(self.many_x_hat)
        self.many_P_hat = np.array(self.many_P_hat)
        self.many_truth = np.array(self.many_truth)

        assert self.many_x_hat.shape[0] == self.m
        assert self.many_P_hat.shape[0] == self.m
        assert self.many_truth.shape[0] == self.m



    def run_one(self, n):
        T = 1
        t = 0
        self.n = n
        
        self.before_one()
        self.truth = self.target.true_states(T, n+1)
        
        m = self.sensor.generate_measurement(t, self.truth[0, :self.dim])
        self.kf.initialize(m.z, m.R)

        self.after_initialize()

        for position in self.truth[1:, :self.dim]:
            t += T

            self.kf.predict(T)

            self.after_predict()

            m = self.sensor.generate_measurement(t, position)
            self.kf.update(m.z, m.R)

            self.after_update(m)

        self.after_one()



    def run_many(self, m, n, seeds=None):
        if seeds is None:
            seeds = np.arange(m)
        assert m == len(seeds)
        
        self.n = n
        self.m = m
        self.seeds = seeds

        self.before_many()
        for seed in seeds:
            self.sensor.reset_seed(seed)
            self.kf.reset()

            self.run_one(n)
        
        self.kf.reset()
        self.after_many()


def run_one(n, target, sensor, kf):
    runner = Runner(target, sensor, kf)
    runner.run_one(n)
    return runner.one_x_hat, runner.one_P_hat, runner.truth[1:,:]


def run_many(m, n, target, sensor, kf, seeds=None):
    runner = Runner(target, sensor, kf)
    runner.run_many(m, n, seeds)
    return runner.many_x_hat, runner.many_P_hat, runner.truth[1:,:]






def plot_error(runner, skip=0, run=0):
    assert runner.m == 1

    tm  = np.arange(runner.n-skip).reshape((-1, 1))
    err = runner.many_x_hat[run, skip:,:3,0] - runner.truth[(skip+1):,:3]

    df = to_df(np.concatenate((tm, err), axis=1), columns=['time', 'x', 'y', 'z'])
    df = df.melt(['time'], ['x', 'y', 'z'], 'dim', 'error')

    fig = ex.line(df, x='time', y='error', facet_row='dim')
    fig.update_yaxes(matches=None)

    return fig




def plot_2d(runner, skip=0, run=0):
    tm = np.arange(runner.truth.shape[0])
    cl = colorscale(n=2)

    fig = make_subplots(rows=3, cols=1,)
    
    fig.add_trace(go.Scatter(x=tm, y=runner.many_x_hat[run, skip:,0,0], name='x_hat', legendgroup='x_hat', marker=dict(size=.1, color=cl[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=runner.many_x_hat[run, skip:,1,0], name='x_hat', legendgroup='x_hat', showlegend=False, marker=dict(size=.1, color=cl[0])), row=2, col=1)
    fig.add_trace(go.Scatter(x=tm, y=runner.many_x_hat[run, skip:,2,0], name='x_hat', legendgroup='x_hat', showlegend=False, marker=dict(size=.1, color=cl[0])), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=tm, y=runner.truth[(skip+1):,0], name='truth', legendgroup='truth', marker=dict(size=.1, color=cl[1])), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=runner.truth[(skip+1):,1], name='truth', legendgroup='truth', showlegend=False, marker=dict(size=.1, color=cl[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=tm, y=runner.truth[(skip+1):,2], name='truth', legendgroup='truth', showlegend=False, marker=dict(size=.1, color=cl[1])), row=3, col=1)

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



















class NeesEvaluationResult:
    def __init__(self, scores, dim):
        self.scores = scores
        self.dim = dim


def evaluate_nees(x_hat, P_hat, truth):
    # if data comes from single run, add a dimension in the front
    if len(x_hat.shape) == 3 and len(P_hat.shape) == 3:
        x_hat = np.expand_dims(x_hat, 0)
        P_hat = np.expand_dims(P_hat, 0)

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(x_hat.shape) == 4
    assert len(P_hat.shape) == 4
    assert len(truth.shape) in [2, 3, 4]
    
    dim   = truth.shape[-1]
    
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
    
    diff  = x_hat - truth
    P_inv = np.linalg.inv(P_hat)
    
    scores = np.matmul(np.matmul(diff.transpose(0, 1, 3, 2), P_inv), diff).squeeze()
    scores = np.atleast_2d(scores)
    return NeesEvaluationResult(scores, dim)


class EvaluationResult:
    def __init__(self, position_nees, velocity_nees):
        self.position_nees = position_nees
        self.velocity_nees = velocity_nees

    
def evaluate_many(x_hat, P_hat, truth):
    assert len(x_hat.shape) == 4
    assert len(P_hat.shape) == 4
    assert len(truth.shape) in [2, 3]
    
    assert x_hat.shape[0] == P_hat.shape[0] # number of independent runs
    assert x_hat.shape[1] == P_hat.shape[1] # length of a single run
    assert x_hat.shape[2] == P_hat.shape[2] # number of state dimensions
    assert P_hat.shape[2] == P_hat.shape[3] # P_hat is a square matrix
    assert x_hat.shape[3] == 1              # x_hat is a column vector
    
    if len(truth.shape) == 2:
        truth = np.expand_dims(truth, 0)

    return EvaluationResult(
        evaluate_nees(x_hat[:,:,:3,:], P_hat[:,:,:3,:3], truth[:, :,:3]),
        evaluate_nees(x_hat[:,:,3:,:], P_hat[:,:,3:,3:], truth[:, :,3:])
    )


def evaluate_runner(runner):
    return evaluate_many(runner.many_x_hat[:, :, :6, :], runner.many_P_hat[:, :, :6, :6], runner.many_truth[:, 1:, :6])


def nees_ci(runner):
    return sp.stats.chi2.ppf([0.025, 0.975], runner.m * runner.dim) / runner.m


def plot_nees(nees: NeesEvaluationResult, skip=25):
    scores = nees.scores[:, skip:]
    dim    = nees.dim
    
    # confidence interval for the mean
    run_count = scores.shape[0]
    ci_mean = sp.stats.chi2.ppf([0.025, 0.975], run_count * dim) / run_count

    mean_score = np.mean(scores, axis=0)


    # confidence interval for individual runs
    ci_qs = sp.stats.chi2.ppf([0.025, 0.975], dim)

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"colspan": 2}, None], [{}, {}]])
    tm = SubFigure(fig, 1, 1)
    h1 = SubFigure(fig, 2, 1)
    h2 = SubFigure(fig, 2, 2)


    # -- time plot
    x = np.arange(scores.shape[1])
    
    tm.add_trace(go.Scatter(x=x, y=np.median(scores, axis=0), mode='lines', marker_color='blue', name='median'))
    tm.add_trace(go.Scatter(x=x, y=mean_score, mode='lines', marker_color='red', name='mean'))
    
    # confidence interval for the mean
    tm.add_hline(y=ci_mean[0], line_width=.5, line_dash="dash", line_color="red")
    tm.add_hline(y=ci_mean[1], line_width=.5, line_dash="dash", line_color="red")

    tm.add_hline(y=ci_qs[0], line_width=.5, line_dash="dash", line_color="green")
    tm.add_hline(y=ci_qs[1], line_width=.5, line_dash="dash", line_color="green")

    # observed quantiles
    q025 = np.quantile(scores, .025, axis=0)
    q975 = np.quantile(scores, .975, axis=0)

    q_color = 'rgba(143,188,143, 0.5)'
    tm.add_trace(go.Scatter(x=x, y=q025, fill=None, mode='lines', marker_color=q_color, showlegend=False, legendgroup='conf_int'))
    tm.add_trace(go.Scatter(x=x, y=q975, fill='tonexty', mode='lines', fillcolor=q_color, line_color=q_color, legendgroup='conf_int', name='95% conf'))
        
    
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

    data_mass = np.mean(np.logical_and(ci_qs[0] <= scores.reshape(-1), scores.reshape(-1) <= ci_qs[1]))
    h2.add_annotation(text=f"Mass within bounds: {round(data_mass*100, 2)}%", xref="x domain", yref="y domain", x=0, y=1, showarrow=False)



    fig.update_layout(height=700)    
    return fig

