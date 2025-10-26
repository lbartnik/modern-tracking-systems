import numpy as np
from typing import Dict, List, Union

import plotly.graph_objects as go

import tracking_v2.callback as cb
import tracking_v2.tracker.mht as mht

from tracking_v2.sensor import SensorMeasurement
from tracking_v2.target import Target
from tracking_v2.tracker import Track
from tracking_v2.util import colorscale


class MhtInspectCallback:
    def __init__(self):
        self.time: List[float] = []
        self.targets: Dict[int, Target] = {}
        self.measurements: Dict[int, Union[List, np.ndarray]] = {}
        self.tracks: Dict[int, Union[List[np.ndarray], np.ndarray]] = {}

    @cb.target_cached
    def target_cached(self, target: Target):
        self.targets[target.target_id] = target

    @cb.measurement_frame
    def measurement_frame(self, time: float, measurements: List[SensorMeasurement]):
        # TODO at the moment we assume that each target gets a measurement in each iteration
        for m in measurements:
            assert m.time == time, f"measurement time {m.time} different from tracker time {time}"
            self.measurements.setdefault(m.target_id, []).append(np.concatenate(([time], m.z.squeeze())))
        
    @cb.tracks_estimated
    def time_tick(self, time: float, tracks: List[Track]):
        self.time.append(time)
        for tr in tracks:
            self.tracks.setdefault(tr.track_id, []).append(np.concatenate(([time], tr.mean.squeeze())))

    @cb.after_one
    def create_arrays(self):
        self.measurements = {target_id: np.asarray(data) for target_id, data in self.measurements.items()}
        self.tracks = {track_id: np.asarray(data).squeeze() for track_id, data in self.tracks.items()}

        # TODO after a single run, make sure that timestamps are aligned across all collected datasets


class MinMax3D:
    def __init__(self):
        self.m = [np.inf, np.inf, np.inf]
        self.M = [-np.inf, -np.inf, -np.inf]
    
    def accumulate(self, array):
        assert len(array.shape) == 2
        assert array.shape[1] == 3, f"shape: {array.shape}"

        m = np.min(array, axis=0)
        self.m = np.min([self.m, m], axis=0)

        M = np.max(array, axis=0)
        self.M = np.max([self.M, M], axis=0)
    
    def padded(self, padding: float = .05):
        margin = (self.M - self.m) * padding
        return self.m - margin, self.M + margin



def plot_mht(cb: MhtInspectCallback, animate: bool = True, lag: int = 5):
    tm = np.asarray(cb.time)


    ids = list(cb.targets.keys())
    ids.extend([f"m{target_id}" for target_id in cb.targets.keys()])
    ids.extend([f"tr{track_id}" for track_id in cb.tracks.keys()])

    colors = colorscale(ids)

    trace_count_base = len(cb.targets)
    trace_ids = {_id: _count + trace_count_base for _count, _id in enumerate(ids)}


    fig = go.Figure()

    # line trajectories of all targets
    for target in cb.targets.values():
        fig.add_trace(go.Scatter3d(
            x=target.cached_states[:, 0],
            y=target.cached_states[:, 1],
            z=target.cached_states[:, 2],
            mode='lines',
            line=dict(width=2, color=colors[target.target_id]),
            name=f"target {target.target_id}",
            legendgroup=f"t{target.target_id}",
            showlegend=True
        ))
    
    for target in cb.targets.values():
        fig.add_trace(go.Scatter3d(
            x=target.cached_states[:, 0],
            y=target.cached_states[:, 1],
            z=target.cached_states[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors[target.target_id]),
            legendgroup=f"t{target.target_id}",
            showlegend=False,
            meta=target.cached_time,
            hovertemplate=f'Target {target.target_id}' +
                           '<br>(%{x}, %{y}, %{z})' +
                           '<br>Time: %{meta}'
        ))

    for target_id, measurements in cb.measurements.items():
        fig.add_trace(go.Scatter3d(
            x=measurements[:, 1],
            y=measurements[:, 2],
            z=measurements[:, 3],
            mode='markers',
            marker=dict(size=4, color=colors[f"m{target_id}"]),
            legendgroup=f"m{target_id}",
            name=f"measurements {target_id}",
            showlegend=True,
            meta=measurements[:, 0],
            hovertemplate='Measurement' +
                        '<br>(%{x}, %{y}, %{z})' +
                        '<br>Time: %{meta}'
        ))

    for track_id, updates in cb.tracks.items():
        fig.add_trace(go.Scatter3d(
            x=updates[:, 1],
            y=updates[:, 2],
            z=updates[:, 3],
            mode='markers+lines',
            marker=dict(size=4, color=colors[f"tr{track_id}"]),
            legendgroup=f"tr{track_id}",
            name=f"track {track_id}",
            showlegend=True,
            meta=updates[:, 0],
            hovertemplate=f'Track {track_id}' +
                        '<br>(%{x}, %{y}, %{z})' +
                        '<br>Time: %{meta}'
        ))


    if animate:
        frames = []

        for ts in np.lib.stride_tricks.sliding_window_view(tm, lag):
            data, traces = [], []

            for target in cb.targets.values():
                i = np.isin(target.cached_time, ts)
                data.append(go.Scatter3d(
                    x=target.cached_states[i, 0],
                    y=target.cached_states[i, 1],
                    z=target.cached_states[i, 2],
                ))
                traces.append(trace_ids[target.target_id])


            for target_id, measurements in cb.measurements.items():
                i = np.isin(measurements[:, 0], ts)
                data.append(go.Scatter3d(
                    x=measurements[i, 1],
                    y=measurements[i, 2],
                    z=measurements[i, 3],
                ))
                traces.append(trace_ids[f"m{target_id}"])
            
            for track_id, updates in cb.tracks.items():
                i = np.isin(updates[:, 0], ts)
                data.append(go.Scatter3d(
                    x=updates[i, 1],
                    y=updates[i, 2],
                    z=updates[i, 3]
                ))
                traces.append(trace_ids[f"tr{track_id}"])

            frames.append(go.Frame(name=str(ts[-1]), data=data, traces=traces))
        
        fig.update(frames=frames)


        frame_duration = 1000
        updatemenus = [dict(type='buttons',
                            buttons=[{
                                "args": [None,
                                         {"frame": {"duration": frame_duration, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 0}}],
                                "label": "Play",
                                "method": "animate"
                            }, {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Stop",
                                "method": "animate"
                            }],
                            direction='left',
                            pad=dict(r=10, t=75),
                            showactive=True, x=0.1, y=0, xanchor='right', yanchor='top')
                       ]
        sliders = [{'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {'font': {'size': 16}, 'prefix': 'Time: ', 'visible': True,
                                     'xanchor': 'right'},
                    'transition': {'duration': 0},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9, 'x': 0.1, 'y': 0,
                    'steps': [{'args': [[frame.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'transition': {'duration': 0}}],
                               'label': frame.name,
                               'method': 'animate'} for frame in frames]
                   }]
        fig.update_layout(updatemenus=updatemenus, sliders=sliders)


    lim = MinMax3D()
    for t in cb.targets.values():
        lim.accumulate(t.cached_states[:, :3])
    for meas in cb.measurements.values():
        lim.accumulate(meas[:, 1:])
    for updates in cb.tracks.values():
        lim.accumulate(updates[:, 1:4])
    m, M = lim.padded()


    fig.update_layout(width=1000, height=800,
        scene=dict(
            xaxis=dict(range=[m[0], M[0]], autorange=False, zeroline=False),
            yaxis=dict(range=[m[1], M[1]], autorange=False, zeroline=False),
            zaxis=dict(range=[m[2], M[2]], autorange=False, zeroline=False)
        )
    )



    # at every step
    #  1. plot target
    #  2. measurement
    #  3. plot all finalized decisions
    #  4. plot all potential decisions

    return go.FigureWidget(fig)
