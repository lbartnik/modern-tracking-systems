import datetime
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from numpy.typing import ArrayLike
from typing import List

from .stonesoup import AnimatedPlotterly
from ..runner.filter import FilterRunner, evaluate_runner
from ...util import SubFigure

__all__ = ['plot_track']






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


def plot_track(runner: FilterRunner, m: int = 0, n: ArrayLike = None, gate: float = None) -> go.Figure:
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
        gate = 1

    tail_length = 5

    e = evaluate_runner(runner)
    position_nis = e.position_nis.scores[m, n]
    position_error = e.position_error[m, n]
    position_cov = np.sqrt(np.linalg.det(runner.many_P_hat[m, n, :3, :3]))

    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.types.detection import Detection


    fig = make_subplots(rows=3, cols=1, row_heights=[.5, .25, .25], specs=[[{}], [{"secondary_y": True}], [{}]])

    plotter = AnimatedPlotterly(timesteps=timesteps, tail_length=tail_length/len(n), figure=SubFigure(fig, 1, 1))

    measurement_model = LinearGaussian(
        ndim_state=2,
        mapping=(0, 1),
        noise_covar=np.eye(2)
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
        cov = runner.many_P_hat[m, i, :2, :2] * gate
        s = StoneSoupState(t, mean, cov)
        states.append(s)

    plotter.plot_ground_truths([truth], [0, 1])

    plotter.plot_measurements(measurements, [0, 1])

    track = StoneSoupTrack(m, states)
    plotter.plot_tracks([track], [0, 1], uncertainty=True)

    # plot errors
    trace_base = len(plotter.fig.data)  # number of traces currently in the animation

    # NIS
    error_kwargs = dict(x=[], y=[], mode="lines", hoverinfo='none', legendgroup='NIS',
                        line=dict(dash="dash", color=plotter.colorway[3]),
                        name='NIS', showlegend=True)
    fig.add_trace(go.Scatter(error_kwargs), row=2, col=1)
    fig.add_hline(y=gate, line_width=.5, line_dash="dash", line_color="red", annotation_text='NIS gate', annotation_position='bottom left')

    fig.update_xaxes(range=[n.min() - tail_length + 1, n.max() + tail_length - 1], row=2, col=1)
    fig.update_yaxes(range=[0, position_nis.max() + 1], row=2, col=1)

    # absolute error
    error_kwargs = dict(x=[], y=[], mode="lines", hoverinfo='none', legendgroup='Error',
                        line=dict(dash="dash", color=plotter.colorway[4]),
                        name='Error', showlegend=True)
    fig.add_trace(go.Scatter(error_kwargs), row=2, col=1, secondary_y=True)
    fig.update_yaxes(range=[0, position_error.max() * 1.02], row=2, col=1, secondary_y=True)

    # volume of covariance
    error_kwargs = dict(x=[], y=[], mode="lines", hoverinfo='none', legendgroup='cov',
                        line=dict(dash="dash", color=plotter.colorway[5]),
                        name='Det[Cov]', showlegend=True)
    fig.add_trace(go.Scatter(error_kwargs), row=3, col=1)
    fig.update_xaxes(range=[n.min() - tail_length + 1, n.max() + tail_length - 1], row=3, col=1)
    fig.update_yaxes(range=[0, np.log10(position_cov.max())], type='log', row=3, col=1)



    timesteps = np.asarray(timesteps)
    times = np.asarray([str(t) for t in timesteps])

    for frame in fig.frames:
        data_ = list(frame.data)
        traces_ = list(frame.traces)

        # NIS legend
        data_.append(go.Scatter(x=[0, 0], y=[0, 0]))
        traces_.append(trace_base)

        # error legend
        data_.append(go.Scatter(x=[0, 0], y=[0, 0]))
        traces_.append(trace_base)

        frame_time = datetime.datetime.fromisoformat(frame.name)
        cutoff_time = (frame_time - plotter.time_window)
        mask = np.logical_and(timesteps <= frame_time, timesteps >= cutoff_time)

        # NIS
        data_.append(go.Scatter(x=n[mask],
                                     y=position_nis[mask],
                                     meta=times[mask],
                                     hovertemplate='NIS' +
                                                '<br>(%{y})' +
                                                '<br>Time: %{meta}'))
        traces_.append(trace_base)

        # error
        data_.append(go.Scatter(x=n[mask],
                                     y=position_error[mask],
                                     meta=times[mask],
                                     hovertemplate='Error' +
                                                '<br>(%{y})' +
                                                '<br>Time: %{meta}'))

        traces_.append(trace_base + 1)

        # covariance
        data_.append(go.Scatter(x=n[mask],
                                     y=position_cov[mask],
                                     meta=times[mask],
                                     hovertemplate='Cov' +
                                                '<br>(%{y})' +
                                                '<br>Time: %{meta}'))

        traces_.append(trace_base + 2)

        frame.data = data_
        frame.traces = traces_

    fig.update_layout(height=800)
    return fig
