import numpy as np
from typing import Dict, List, Tuple, Union

import plotly.graph_objects as go
from ipywidgets import widgets

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
        self.measurements: Dict[int, Union[List, np.ndarray]] = {} # columns: time, x, y, z, measuremend ID
        self.tracks: Dict[int, Union[List[np.ndarray], np.ndarray]] = {} # track updates; time + state

        # update_to_meas: track id -> time -> (used, considered)
        # used, considered: [mht.UpdateTrack]
        self.update_to_meas: Dict[int, Dict[float, Tuple[List[mht.UpdateTrack],
                                                         List[mht.UpdateTrack]]]] = {}
        self.track_to_target = {}        

    @cb.target_cached
    def target_cached(self, target: Target):
        self.targets[target.target_id] = target

    @cb.measurement_frame
    def measurement_frame(self, time: float, measurements: List[SensorMeasurement]):
        self.time.append(time)

        # TODO at the moment we assume that each target gets a measurement in each iteration
        for m in measurements:
            assert m.time == time, f"measurement time {m.time} different from tracker time {time}"
            self.measurements.setdefault(m.target_id, []).append(np.concatenate(([time], m.z.squeeze(), [m.measurement_id])))
    
    @cb.initialize_track
    def initialize_track(self, tracker, d: mht.InitializeTrack):
        # TODO hold off from assigning target to the track until there are 2 out of 3 measurements
        #      from the same target
        #assert d.measurement0.target_id == d.measurement1.target_id
        d.track.target_id = d.measurement0.target_id

    @cb.consider_update_track
    def consider_update_track(self, tracker, d: mht.UpdateTrack):
        _, considred = self.update_to_meas.setdefault(d.track.track_id, {}).setdefault(d.track.time, ([], []))
        considred.append(d)

    @cb.update_track
    def update_track(self, tracker, d: mht.UpdateTrack):
        used, _ = self.update_to_meas.setdefault(d.track.track_id, {}).setdefault(d.track.time, ([], []))
        used.append(d)

        self.track_to_target[d.track.track_id] = d.track.target_id

    @cb.tracks_estimated
    def time_tick(self, time: float, tracks: List[Track]):
        for tr in tracks:
            self.tracks.setdefault(tr.track_id, []).append(np.concatenate(([time], tr.mean.squeeze())))
        
        # reset decision buffer before the next iteration
        self.decisions = []

    @cb.before_one
    def reset(self):
        self.measurements = {}
        self.tracks = {}
        self.time = []

    @cb.after_one
    def create_arrays(self):
        self.measurements = {target_id: np.asarray(data, dtype=np.float64) for target_id, data in self.measurements.items()}
        self.tracks = {track_id: np.asarray(data, dtype=np.float64).squeeze() for track_id, data in self.tracks.items()}
        self.time = np.asarray(self.time, dtype=np.float64)

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


# at every step
#  1. plot target
#  2. measurement
#  3. plot all finalized decisions
#  4. plot all potential decisions
def plot_mht(cb: MhtInspectCallback, animate: bool = True, lag: int = 5):
    tm = np.asarray(cb.time)

    target_ids = list([str(i) for i in cb.targets.keys()])
    measurement_ids = [f"m{target_id}" for target_id in cb.targets.keys()]
    track_ids = [f"tr{track_id}" for track_id in cb.tracks.keys()]
    ids = target_ids + measurement_ids + track_ids + ['decision_confirmed', 'decisions_considered']

    trace_ids = {}

    colors = colorscale(ids)

    if animate:
        fig = go.FigureWidget()
    else:
        fig = go.Figure()

    # line trajectories of all targets
    for target in cb.targets.values():
        fig.add_trace(go.Scatter3d(
            x=target.cached_states[:, 0],
            y=target.cached_states[:, 1],
            z=target.cached_states[:, 2],
            mode='lines',
            line=dict(width=2, color=colors[str(target.target_id)]),
            name=f"target {target.target_id}",
            legendgroup=f"t{target.target_id}",
            showlegend=True
        ))
    
    for target in cb.targets.values():
        trace_ids[str(target.target_id)] = len(fig.data)

        fig.add_trace(go.Scatter3d(
            x=target.cached_states[:, 0],
            y=target.cached_states[:, 1],
            z=target.cached_states[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors[str(target.target_id)]),
            legendgroup=f"t{target.target_id}",
            showlegend=False,
            meta=target.cached_time,
            hovertemplate=f'Target {target.target_id}' +
                           '<br>(%{x}, %{y}, %{z})' +
                           '<br>Time: %{meta}'
        ))

    for target_id, measurements in cb.measurements.items():
        trace_ids[f"m{target_id}"] = len(fig.data)

        fig.add_trace(go.Scatter3d(
            x=measurements[:, 1],
            y=measurements[:, 2],
            z=measurements[:, 3],
            mode='markers',
            marker=dict(size=4, color=colors[f"m{target_id}"]),
            legendgroup=f"m{target_id}",
            name=f"measurements {target_id}",
            showlegend=True,
            meta=measurements[:, (0, 4)],
            hovertemplate=f'Measurement %{{meta[1]}} ({target_id})' +
                           '<br>(%{x}, %{y}, %{z})' +
                           '<br>Time: %{meta[0]}'
        ))

    for track_id, updates in cb.tracks.items():
        trace_ids[f"tr{track_id}"] = len(fig.data)

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

    trace_ids['decision_confirmed'] = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[],
                               mode='lines',
                               marker_color=colors['decision_confirmed'],
                               name='confirmed decision'))

    trace_ids['decisions_considered'] = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[],
                               mode='lines',
                               line_dash="dash",
                               marker_color=colors['decisions_considered'],
                               name='considered decision(s)'))


    lim = MinMax3D()
    for t in cb.targets.values():
        lim.accumulate(t.cached_states[:, :3])
    for meas in cb.measurements.values():
        lim.accumulate(meas[:, 1:4])
    for updates in cb.tracks.values():
        lim.accumulate(updates[:, 1:4])
    m, M = lim.padded()


    fig.update_layout(width=800, height=600,
        scene=dict(
            xaxis=dict(range=[m[0], M[0]], autorange=False, zeroline=False),
            yaxis=dict(range=[m[1], M[1]], autorange=False, zeroline=False),
            zaxis=dict(range=[m[2], M[2]], autorange=False, zeroline=False)
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",  # Optional: make the legend horizontal
            yanchor="top",    # Anchor the legend to its top edge
            y=-0.2,           # Position the legend below the plot area (adjust as needed)
            xanchor="left",   # Anchor the legend to its left edge
            x=0               # Position the legend starting from the left
        )
    )

    # map from trace ID back to target/measurement/track ID
    reverse_trace_ids = {plotly_id: string_id for string_id, plotly_id in trace_ids.items()}


    if not animate:
        return fig
    
    time_slider = widgets.IntSlider(
        value=int(lag),
        min=int(lag),
        max=int(tm.max()),
        step=1,
        description='Time',
        continuous_update=False
    )
    fwd = widgets.Button(description=">")
    rew = widgets.Button(description="<")
    text = widgets.Textarea(disabled=True, layout=widgets.Layout(width='500px', height='600px'))


    def update_plot(change):
        end_time = time_slider.value
        ts = np.arange(np.max([0, end_time - lag]), end_time+1)

        with fig.batch_update():
            for target in cb.targets.values():
                i = trace_ids[str(target.target_id)]
                j = np.isin(target.cached_time, ts)
                
                fig.data[i].x = target.cached_states[j, 0]
                fig.data[i].y = target.cached_states[j, 1]
                fig.data[i].z = target.cached_states[j, 2]
                fig.data[i].meta = target.cached_time[j]


            for target_id, measurements in cb.measurements.items():
                i = trace_ids[f"m{target_id}"]
                j = np.isin(measurements[:, 0], ts)

                fig.data[i].x = measurements[j, 1]
                fig.data[i].y = measurements[j, 2]
                fig.data[i].z = measurements[j, 3]
                fig.data[i].meta = measurements[j, :][:, (0, 4)]
            
            for track_id, updates in cb.tracks.items():
                i = trace_ids[f"tr{track_id}"]
                j = np.isin(updates[:, 0], ts)

                fig.data[i].x = updates[j, 1]
                fig.data[i].y = updates[j, 2]
                fig.data[i].z = updates[j, 3]
                fig.data[i].meta = updates[j, 0]
    
    def forward_time(k):
        time_slider.value = time_slider.value + 1
        update_plot(None)
    
    def rewind_time(k):
        time_slider.value = time_slider.value - 1
        update_plot(None)
    
    def explain_measurement(trace, points, state):
        if not len(points.point_inds):
            return
        
        trace_symbol = reverse_trace_ids[points.trace_index]
        assert trace_symbol.startswith('m')

    def explain_track_update(trace, points, state):
        if not len(points.point_inds):
            return
                    
        trace_symbol = reverse_trace_ids[points.trace_index]
        assert trace_symbol.startswith('tr')
        
        track_id = int(trace_symbol[2:])
        j = points.point_inds[0]
        tm = trace.meta[j]

        used, considered = cb.update_to_meas.get(track_id, {}).get(tm, ([], []))

        with fig.batch_update():
            x, y, z = [], [], []
            for d in used:
                x.extend([float(trace.x[j]), float(d.measurement.z[0,0]), None])
                y.extend([float(trace.y[j]), float(d.measurement.z[0,1]), None])
                z.extend([float(trace.z[j]), float(d.measurement.z[0,2]), None])
            
            i = trace_ids['decision_confirmed']
            fig.data[i].x, fig.data[i].y, fig.data[i].z = x, y, z
            

            x, y, z = [], [], []
            for d in considered:
                x.extend([float(trace.x[j]), float(d.measurement.z[0,0]), None])
                y.extend([float(trace.y[j]), float(d.measurement.z[0,1]), None])
                z.extend([float(trace.z[j]), float(d.measurement.z[0,2]), None])
            
            i = trace_ids['decisions_considered']
            fig.data[i].x, fig.data[i].y, fig.data[i].z = x, y, z

        with np.printoptions(precision=2, floatmode="fixed"):
            i = cb.tracks[track_id][:, 0] == tm
            s = cb.tracks[track_id][i, 1:].squeeze()
            description = f"track {track_id} ({cb.track_to_target[track_id]})\npos: {s[:3]}\nvel: {s[3:]}\n\n"

            def _describe(d: mht.UpdateTrack):
                err = np.linalg.norm(d.measurement.z[:3,0] - d.track.kf.x_hat[:3,0])
                desc  = f"{d.measurement.measurement_id} ({d.measurement.target_id}): {d.measurement.z.squeeze()}: "
                desc += f"{d.NIS:.2f} {d.LLR:.2f} {err:.2f} {d.S_det:.1f}\n"
                return desc

            description += "(ID, z, NIS, dLLR, err)\n\nused:\n"
            for d in used:
                description += _describe(d)
            
            description += "\nconsidered:\n"
            for d in considered:
                description += _describe(d)

        text.value = description
        
        

            
    time_slider.observe(update_plot, names="value")
    fwd.on_click(forward_time)
    rew.on_click(rewind_time)
    
    for i in measurement_ids:
        fig.data[trace_ids[i]].on_click(explain_measurement)
    for i in track_ids:
        fig.data[trace_ids[i]].on_click(explain_track_update)

    return widgets.HBox([
        widgets.VBox([fig, widgets.HBox([time_slider, rew, fwd])]),
        text
    ])

