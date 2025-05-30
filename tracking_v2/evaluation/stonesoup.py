from stonesoup.plotter import _Plotter, Plotterly
from stonesoup.types.state import StateMutableSequence

from collections.abc import Collection
from datetime import datetime, timedelta
from itertools import chain
from mergedeep import merge
from plotly import colors
from plotly.subplots import make_subplots

import warnings
import numpy as np
import plotly.graph_objects as go


class AnimatedPlotterly(_Plotter):
    """
    Class for a 2D animated plotter that uses Plotly graph objects rather than matplotlib.
    This gives the user the ability to see how tracking works through time, while being
    able to interact with tracks, truths, etc, in the same way that is enabled by
    Plotly static plots.

    Simplifies the process of plotting ground truths, measurements, clutter, and tracks.
    Tracks can be plotted with uncertainty ellipses or particles if required. Legends
    are automatically generated with each plot.

    Parameters
    ----------
    timesteps: Collection
        Collection of equally-spaced timesteps. Each animation frame is a timestep.
    tail_length: float
        Percentage of sim time for which previous values will still be displayed for.
        Value can be between 0 and 1. Default is 0.3.
    equal_size: bool
        Makes x and y axes equal when figure is resized. Default is False.
    sim_duration: int
        Time taken to run animation (s). Default is 6
    \\*\\*kwargs
        Additional arguments to be passed in the initialisation.

    Attributes
    ----------

    """

    def __init__(self, timesteps, tail_length=0.3, equal_size=False,
                 sim_duration=6, figure: go.Figure=None, **kwargs):
        """
        Initialise the figure and checks that inputs are correctly formatted.
        Creates an empty frame for each timestep, and configures
        the buttons and slider.


        """
        if go is None or colors is None:
            raise RuntimeError("Usage of Plotterly plotter requires installation of `plotly`")

        self.equal_size = equal_size

        # checking that there are multiple timesteps
        if len(timesteps) < 2:
            raise ValueError("Must be at least 2 timesteps for animation.")

        # checking that timesteps are evenly spaced
        time_spaces = np.unique(np.diff(timesteps))

        # gives the unique values of time gaps between timesteps. If this contains more than
        # one value, then timesteps are not all evenly spaced which is an issue.
        if len(time_spaces) != 1:
            warnings.warn("Timesteps are not equally spaced, so the passage of time is not linear")
        self.timesteps = timesteps

        # checking input to tail_length
        if tail_length > 1 or tail_length < 0:
            raise ValueError("Tail length should be between 0 and 1")
        self.tail_length = tail_length

        # checking sim_duration
        if sim_duration <= 0:
            raise ValueError("Simulation duration must be positive")

        # time window is calculated as sim_length * tail_length. This is
        # the window of time for which past plots are still visible
        self.time_window = (timesteps[-1] - timesteps[0]) * tail_length

        self.colorway = colors.qualitative.Plotly[1:]  # plotting colours

        self.all_masks = dict()  # dictionary to be filled up later

        self.plotting_function_called = False  # keeps track if anything has been plotted or not
        # so that only the first data plotted will override the default axis max and mins.

        if figure is None:
            figure = go.Figure()
        self.fig = figure

        layout_kwargs = dict(
            xaxis=dict(title=dict(text="<i>x</i>")),
            yaxis=dict(title=dict(text="<i>y</i>")),
            colorway=self.colorway,  # Needed to match colours later.
            height=550,
            autosize=True
        )
        # layout_kwargs.update(kwargs)
        self.fig.update_layout(layout_kwargs)

        # initialise frames according to simulation timesteps
        self.fig.frames = [dict(
            name=str(time),
            data=[],
            traces=[]
        ) for time in timesteps]

        self.fig.update_xaxes(range=[0, 10])
        self.fig.update_yaxes(range=[0, 10])

        frame_duration = sim_duration * 1000 / len(self.fig.frames)

        # if the gap between timesteps is greater than a day, it isn't necessary
        # to display hour and minute information, so remove this to give a cleaner display.
        # a and b are used in the slider steps label later
        if time_spaces[0] >= timedelta(days=1):
            start_cut_off = None
            end_cut_off = 10

        # if the simulation is over a day long, display all information which
        # looks clunky but is necessary
        elif timesteps[-1] - timesteps[0] > timedelta(days=1):
            start_cut_off = None
            end_cut_off = None

        # otherwise, remove day information and just show
        # hours, mins, etc. which is cleaner to look at
        else:
            start_cut_off = 11
            end_cut_off = None

        # create button and slider
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
                    'transition': {'duration': frame_duration, 'easing': 'linear'},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9, 'x': 0.1, 'y': 0,
                    'steps': [{'args': [[frame.name], {
                        'frame': {'duration': 1.0, 'easing': 'linear', 'redraw': True},
                        'transition': {'duration': 0, 'easing': 'linear'}}],
                               'label': frame.name[start_cut_off: end_cut_off],
                               'method': 'animate'} for frame in
                              self.fig.frames
                              ]}]
        self.fig.update_layout(updatemenus=updatemenus, sliders=sliders)
        self.fig.update_layout(kwargs)

    def show(self):
        """
        Display the animation.
        """
        return self.fig

    def _resize(self, data, type="track"):
        """
        Reshape figure so that everything is in view.

        Parameters
        ----------

        data:
            Collection of values that are being added to the figure.
            Will be a list if coming from plot_ground_Truths or
            plot_tracks, but will be a dictionary if coming from plot_measurements.
        """

        # fill in all data. If there is no data, fill all_x, all_y with current axis limits
        if not data:
            all_x = list(self.fig.layout.xaxis.range)
            all_y = list(self.fig.layout.xaxis.range)
        else:
            all_x = list()
            all_y = list()

        # fill in data
        if type == "measurements":

            for key, item in data.items():
                all_x.extend(data[key]["x"])
                all_y.extend(data[key]["y"])

        elif type in ("ground_truth", "tracks"):

            for n, _ in enumerate(data):
                all_x.extend(data[n]["x"])
                all_y.extend(data[n]["y"])

        elif type == "sensor":
            sensor_xy = np.array([sensor.position[[0, 1], 0] for sensor in data])
            all_x.extend(sensor_xy[:, 0])
            all_y.extend(sensor_xy[:, 1])

        elif type == "particle_or_uncertainty":
            # data comes in format of list of dictionaries. Each dictionary contains 'x' and 'y',
            # which are a list of lists.
            for dictionary in data:
                for x_values in dictionary["x"]:
                    all_x.extend([np.nanmax(x_values), np.nanmin(x_values)])
                for y_values in dictionary["y"]:
                    all_y.extend([np.nanmax(y_values), np.nanmin(y_values)])

        xmax = max(all_x)
        ymax = max(all_y)
        xmin = min(all_x)
        ymin = min(all_y)

        if self.equal_size:
            xmax = ymax = max(xmax, ymax)
            xmin = ymin = min(xmin, ymin)

        # if it's first time plotting data, want to ensure plotter is bound to that data
        # and not the default values. Issues arise if the initial plotted data is much
        # smaller than the default 0 to 10 values.
        if not self.plotting_function_called:

            self.fig.update_xaxes(range=[xmin, xmax])
            self.fig.update_yaxes(range=[ymin, ymax])

        # need to check if it's actually necessary to resize or not
        if xmax >= self.fig.layout.xaxis.range[1] or xmin <= self.fig.layout.xaxis.range[0]:

            xmax = max(xmax, self.fig.layout.xaxis.range[1])
            xmin = min(xmin, self.fig.layout.xaxis.range[0])
            xrange = xmax - xmin

            # update figure while adding a small buffer to the mins and maxes
            self.fig.update_xaxes(range=[xmin - xrange / 20, xmax + xrange / 20])

        if ymax >= self.fig.layout.yaxis.range[1] or ymin <= self.fig.layout.yaxis.range[0]:

            ymax = max(ymax, self.fig.layout.yaxis.range[1])
            ymin = min(ymin, self.fig.layout.yaxis.range[0])
            yrange = ymax - ymin

            self.fig.update_yaxes(range=[ymin - yrange / 20, ymax + yrange / 20])

    def plot_ground_truths(self, truths, mapping, label="Ground Truth",
                           resize=True, **kwargs):

        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change linestyle, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : Collection of :class:`~.GroundTruthPath`
            Collection of  ground truths which will be plotted. If not a collection and instead a
            single :class:`~.GroundTruthPath` type, the argument is modified to be a set to allow
            for iteration.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        label: str
            Name of ground truths in legend/plot
        resize: bool
            if True, will resize figure to ensure that ground truths are in view
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.


        .. deprecated:: 1.5
           ``label`` has replaced ``truths_label``. In the current implementation
           ``truths_label`` overrides ``label``. However, use of ``truths_label``
           may be removed in the future.
        """
        label = kwargs.pop('truths_label', None) or label

        if not isinstance(truths, Collection) or isinstance(truths, StateMutableSequence):
            truths = {truths}  # Make a set of length 1

        data = [dict() for _ in truths]  # put all data into one place for later plotting
        for n, truth in enumerate(truths):

            # initialise arrays that go inside the dictionary
            data[n].update(x=np.zeros(len(truth)),
                           y=np.zeros(len(truth)),
                           time=np.array([0 for _ in range(len(truth))], dtype=object),
                           time_str=np.array([0 for _ in range(len(truth))], dtype=object),
                           type=np.array([0 for _ in range(len(truth))], dtype=object))

            for k, state in enumerate(truth):
                # fill the arrays here
                data[n]["x"][k] = state.state_vector[mapping[0]]
                data[n]["y"][k] = state.state_vector[mapping[1]]
                data[n]["time"][k] = state.timestamp
                data[n]["time_str"][k] = str(state.timestamp)
                data[n]["type"][k] = type(state).__name__

        trace_base = len(self.fig.data)  # number of traces currently in the animation

        # add a trace that keeps the legend up for the entire simulation (will remain
        # even if no truths are present), then add a trace for each truth in the simulation.
        # initialise keyword arguments, then add them to the traces
        truth_kwargs = dict(x=[], y=[], mode="lines", hoverinfo='none', legendgroup=label,
                            line=dict(dash="dash", color=self.colorway[0]), legendrank=100,
                            name=label, showlegend=True)
        merge(truth_kwargs, kwargs)
        # legend dummy trace
        self.fig.add_trace(go.Scatter(truth_kwargs), row=1, col=1)

        # we don't want the legend for any of the actual traces
        truth_kwargs.update({"showlegend": False})

        for n, _ in enumerate(truths):
            # change the colour of each truth and include n in its name
            merge(truth_kwargs, dict(line=dict(color=self.colorway[n % len(self.colorway)])))
            merge(truth_kwargs, kwargs)
            self.fig.add_trace(go.Scatter(truth_kwargs), row=1, col=1)  # add to traces

        for frame in self.fig.frames:

            # get current fig data and traces
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # convert string to datetime object
            frame_time = datetime.fromisoformat(frame.name)
            cutoff_time = (frame_time - self.time_window)

            # for the legend
            data_.append(go.Scatter(x=[0, 0], y=[0, 0]))
            traces_.append(trace_base)

            for n, truth in enumerate(truths):
                # all truth points that come at or before the frame time
                t_upper = [data[n]["time"] <= frame_time]

                # only select detections that come after the time cut-off
                t_lower = [data[n]["time"] >= cutoff_time]

                # put together
                mask = np.logical_and(t_upper, t_lower)

                # find x, y, time, and type
                truth_x = data[n]["x"][tuple(mask)]
                # add in np.inf to ensure traces are present for every timestep
                truth_x = np.append(truth_x, [np.inf])
                truth_y = data[n]["y"][tuple(mask)]
                truth_y = np.append(truth_y, [np.inf])
                times = data[n]["time_str"][tuple(mask)]

                data_.append(go.Scatter(x=truth_x,
                                        y=truth_y,
                                        meta=times,
                                        hovertemplate='GroundTruthState' +
                                                      '<br>(%{x}, %{y})' +
                                                      '<br>Time: %{meta}'))

                traces_.append(trace_base + n + 1)  # append data to correct trace

                frame.data = data_
                frame.traces = traces_

        if resize:
            self._resize(data, type="ground_truth")

        # we have called a plotting function so update flag (gets used in _resize)
        self.plotting_function_called = True

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          resize=True, label="Measurements",
                          convert_measurements=True, **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        resize: bool
            If True, will resize figure to ensure measurements are in view
        label : str
            Label for the measurements.  Default is "Measurements".
        convert_measurements : bool
            Should the measurements be converted from measurement space to state space before
            being plotted. Default is True
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for detections. Defaults are
            ``marker=dict(color="#636EFA")``.


        .. deprecated:: 1.5
           ``label`` has replaced ``measurements_label``. In the current implementation
           ``measurements_label`` overrides ``label``. However, use of ``measurements_label``
           may be removed in the future.
        """
        label = kwargs.pop('measurements_label', None) or label

        if not isinstance(measurements, Collection):
            measurements = {measurements}  # Make a set of length 1

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = measurements
        plot_detections, plot_clutter = self._conv_measurements(measurements_set,
                                                                mapping,
                                                                measurement_model,
                                                                convert_measurements)
        plot_combined = {'Detection': plot_detections,
                         'Clutter': plot_clutter}  # for later reference

        # this dictionary will store all the plotting data that we need
        # from the detections and clutter into numpy arrays that we can easily
        # access to plot
        combined_data = dict()

        # only add clutter or detections to plot if necessary
        if plot_detections:
            combined_data.update(dict(Detection=dict()))
        if plot_clutter:
            combined_data.update(dict(Clutter=dict()))

        # initialise combined_data
        for key in combined_data.keys():
            length = len(plot_combined[key])
            combined_data[key].update({
                "x": np.zeros(length),
                "y": np.zeros(length),
                "time": np.array([0 for _ in range(length)], dtype=object),
                "time_str": np.array([0 for _ in range(length)], dtype=object),
                "type": np.array([0 for _ in range(length)], dtype=object)})

        # and now fill in the data

        for key in combined_data.keys():
            for n, det in enumerate(plot_combined[key]):
                x, y = list(plot_combined[key].values())[n]
                combined_data[key]["x"][n] = x
                combined_data[key]["y"][n] = y
                combined_data[key]["time"][n] = det.timestamp
                combined_data[key]["time_str"][n] = str(det.timestamp)
                combined_data[key]["type"][n] = type(det).__name__

        # get number of traces currently in fig
        trace_base = len(self.fig.data)

        if plot_detections:
            # initialise detections
            if plot_clutter:
                name = label + "<br>(Detections)"
            else:
                name = label
            measurement_kwargs = dict(x=[], y=[], mode='markers',
                                      name=name,
                                      legendgroup=name,
                                      legendrank=200, showlegend=True,
                                      marker=dict(color="#636EFA"), hoverinfo='none')
            merge(measurement_kwargs, kwargs)

            self.fig.add_trace(go.Scatter(measurement_kwargs), row=1, col=1)  # trace for legend

            measurement_kwargs.update({"showlegend": False})
            self.fig.add_trace(go.Scatter(measurement_kwargs), row=1, col=1)  # trace for plotting

        if plot_clutter:
            # change necessary kwargs to initialise clutter trace
            name = label + "<br>(Clutter)"
            clutter_kwargs = dict(x=[], y=[], mode='markers',
                                  name=name,
                                  legendgroup=name,
                                  legendrank=300, showlegend=True,
                                  marker=dict(symbol="star-triangle-up", color='#FECB52'),
                                  hoverinfo='none')
            merge(clutter_kwargs, kwargs)

            self.fig.add_trace(go.Scatter(clutter_kwargs), row=1, col=1)  # trace for plotting clutter

        # add data to frames
        for frame in self.fig.frames:

            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # add blank data to ensure detection legend stays in place
            data_.append(go.Scatter(x=[-np.inf, np.inf], y=[-np.inf, np.inf]))
            traces_.append(trace_base)  # ensure data is added to correct trace

            frame_time = datetime.fromisoformat(frame.name)  # convert string to datetime object

            # time at which dets will disappear from the fig
            cutoff_time = (frame_time - self.time_window)

            for j, key in enumerate(combined_data.keys()):
                # only select measurements that arrive by the time of the current frame
                t_upper = [combined_data[key]["time"] <= frame_time]

                # only select detections that come after the time cut-off
                t_lower = [combined_data[key]["time"] >= cutoff_time]

                # put them together to create the final mask
                mask = np.logical_and(t_upper, t_lower)

                # find x and y points for true detections and clutter
                det_x = combined_data[key]["x"][tuple(mask)]
                det_x = np.append(det_x, [np.inf])
                det_y = combined_data[key]["y"][tuple(mask)]
                det_y = np.append(det_y, [np.inf])
                det_times = combined_data[key]["time_str"][tuple(mask)]

                data_.append(go.Scatter(x=det_x,
                                        y=det_y,
                                        meta=det_times,
                                        hovertemplate=f'{key}' +
                                                      '<br>(%{x}, %{y})' +
                                                      '<br>Time: %{meta}'))
                traces_.append(trace_base + j + 1)

            frame.data = data_  # update the figure
            frame.traces = traces_

        if resize:
            self._resize(combined_data, "measurements")

        # we have called a plotting function so update flag (gets used in resize)
        self.plotting_function_called = True

    def plot_tracks(self, tracks, mapping, uncertainty=False, resize=True,
                    particle=False, plot_history=False, ellipse_points=30,
                    label="Tracks", **kwargs):
        """
        Plots each track generated, generating a legend automatically. If 'uncertainty=True',
        error ellipses are plotted. Tracks are plotted as solid lines with point markers
        and default colours.

        Users can change linestyle, color, and marker using keyword arguments. Uncertainty metrics
        will also be plotted with the user defined colour and any changes will apply to all tracks.

        Parameters
        ----------
        tracks: Collection of :class '~Track'
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:'~Track' type, the argument is modified to be a set to allow for iteration

        mapping: list
            List of items specifying the mapping of the position
            components of the state space
        uncertainty: bool
            If True, function plots uncertainty ellipses
        resize: bool
            If True, plotter will change bounds so that tracks are in view
        particle: bool
            If True, function plots particles
        plot_history: bool
            If true, plots all particles and uncertainty ellipses up to current time step
        ellipse_points: int
            Number of points for polygon approximating ellipse shape
        label: str
            Label to apply to all tracks for legend
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='s'`` for :class:`~.Update` and ``marker='o'`` for other states.


        .. deprecated:: 1.5
           ``label`` has replaced ``track_label``. In the current implementation
           ``track_label`` overrides ``label``. However, use of ``track_label``
           may be removed in the future.
        """
        label = kwargs.pop('track_label', None) or label

        if not isinstance(tracks, Collection) or isinstance(tracks, StateMutableSequence):
            tracks = {tracks}  # Make a set of length 1

        # So that we can plot tracks for both the current time and for some previous times,
        # we put plotting data for each track into a dictionary so that it can be easily
        # accessed later.
        data = [dict() for _ in tracks]

        for n, track in enumerate(tracks):  # sum up means - accounts for particle filter

            xydata = np.concatenate(
                [(getattr(state, 'mean', state.state_vector)[mapping, :])
                 for state in track],
                axis=1)

            # initialise arrays that go inside the dictionary
            data[n].update(x=xydata[0],
                           y=xydata[1],
                           time=np.array([0 for _ in range(len(track))], dtype=object),
                           time_str=np.array([0 for _ in range(len(track))], dtype=object),
                           type=np.array([0 for _ in range(len(track))], dtype=object))

            for k, state in enumerate(track):
                # fill the arrays here
                data[n]["time"][k] = state.timestamp
                data[n]["time_str"][k] = str(state.timestamp)
                data[n]["type"][k] = type(state).__name__

        trace_base = len(self.fig.data)  # number of traces

        # add dummy trace for legend for track

        track_kwargs = dict(x=[], y=[], mode="markers+lines", line=dict(color=self.colorway[2]),
                            legendgroup=label, legendrank=400, name=label,
                            showlegend=True)
        track_kwargs.update(kwargs)
        self.fig.add_trace(go.Scatter(track_kwargs), row=1, col=1)

        # and initialise traces for every track. Need to change a few kwargs:
        track_kwargs.update({'showlegend': False})

        for k, _ in enumerate(tracks):
            # update track colours
            track_kwargs.update({'line': dict(color=self.colorway[(k + 2) % len(self.colorway)])})
            track_kwargs.update(kwargs)
            self.fig.add_trace(go.Scatter(track_kwargs), row=1, col=1)

        for frame in self.fig.frames:
            # get current fig data and traces
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # convert string to datetime object
            frame_time = datetime.fromisoformat(frame.name)

            self.all_masks[frame_time] = dict()  # save mask for later use
            cutoff_time = (frame_time - self.time_window)
            # add blank data to ensure legend stays in place
            data_.append(go.Scatter(x=[-np.inf, np.inf], y=[-np.inf, np.inf]))
            traces_.append(trace_base)  # ensure data is added to correct trace

            for n, track in enumerate(tracks):

                # all track points that come at or before the frame time
                t_upper = [data[n]["time"] <= frame_time]
                # only select detections that come after the time cut-off
                t_lower = [data[n]["time"] >= cutoff_time]

                # put together
                mask = np.logical_and(t_upper, t_lower)

                # put into dictionary for later use
                if plot_history:
                    self.all_masks[frame_time][n] = np.logical_and(t_upper, t_lower)
                else:
                    self.all_masks[frame_time][n] = [data[n]["time"] == frame_time]

                # find x, y, time, and type
                track_x = data[n]["x"][tuple(mask)]
                # add np.inf to plot so that the traces are present for entire simulation
                track_x = np.append(track_x, [np.inf])

                # repeat for y
                track_y = data[n]["y"][tuple(mask)]
                track_y = np.append(track_y, [np.inf])
                track_type = data[n]["type"][tuple(mask)]
                times = data[n]["time_str"][tuple(mask)]

                data_.append(go.Scatter(x=track_x,  # plot track
                                        y=track_y,
                                        meta=track_type,
                                        customdata=times,
                                        hovertemplate='%{meta}' +
                                                      '<br>(%{x}, %{y})' +
                                                      '<br>Time: %{customdata}'))

                traces_.append(trace_base + n + 1)  # add to correct trace

                frame.data = data_
                frame.traces = traces_

        if resize:
            self._resize(data, "tracks")

        if uncertainty:  # plot ellipses
            name = f'{label}<br>Uncertainty'
            uncertainty_kwargs = dict(x=[], y=[], legendgroup=name, fill='toself',
                                      fillcolor=self.colorway[2],
                                      opacity=0.2, legendrank=500, name=name,
                                      hoverinfo='skip',
                                      mode='none', showlegend=True)
            uncertainty_kwargs.update(kwargs)

            # dummy trace for legend for uncertainty
            self.fig.add_trace(go.Scatter(uncertainty_kwargs), row=1, col=1)

            # and an uncertainty ellipse trace for each track
            uncertainty_kwargs.update({'showlegend': False})
            for k, _ in enumerate(tracks):
                uncertainty_kwargs.update(
                    {'fillcolor': self.colorway[(k + 2) % len(self.colorway)]})
                uncertainty_kwargs.update(kwargs)
                self.fig.add_trace(go.Scatter(uncertainty_kwargs), row=1, col=1)

            # following function finds uncertainty data points and plots them
            self._plot_particles_and_ellipses(tracks, mapping, resize, method="uncertainty")

        if particle:  # plot particles

            # initialise traces. One for legend and one per track
            name = f'{label}<br>Particles'
            particle_kwargs = dict(mode='markers', marker=dict(size=2, color=self.colorway[2]),
                                   opacity=0.4,
                                   hoverinfo='skip', legendgroup=name, name=name,
                                   legendrank=520, showlegend=True)
            # apply any keyword arguments
            particle_kwargs.update(kwargs)
            self.fig.add_trace(go.Scatter(particle_kwargs), row=1, col=1)  # legend trace

            particle_kwargs.update({"showlegend": False})

            for k, track in enumerate(tracks):  # trace for each track

                particle_kwargs.update(
                    {'marker': dict(size=2, color=self.colorway[(k + 2) % len(self.colorway)])})
                particle_kwargs.update(kwargs)
                self.fig.add_trace(go.Scatter(particle_kwargs), row=1, col=1)

            self._plot_particles_and_ellipses(tracks, mapping, resize, method="particles")

        # we have called a plotting function so update flag
        self.plotting_function_called = True

    def _plot_particles_and_ellipses(self, tracks, mapping, resize, method="uncertainty"):

        """
        The logic for plotting uncertainty ellipses and particles is nearly identical,
        so it is put into one function.

        Parameters
        ----------
        tracks: Collection of :class '~Track'
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:'~Track' type, the argument is modified to be a set to allow for iteration
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        method: str
            Can either be "uncertainty" or "particles". Depends on what the function is plotting.
        """

        data = [dict() for _ in tracks]
        trace_base = len(self.fig.data)
        for n, track in enumerate(tracks):

            # initialise arrays that store particle/ellipse for later plotting
            data[n].update(x=np.array([0 for _ in range(len(track))], dtype=object),
                           y=np.array([0 for _ in range(len(track))], dtype=object))

            for k, state in enumerate(track):

                # find data points
                if method == "uncertainty":

                    data_x, data_y = Plotterly._generate_ellipse_points(state, mapping)
                    data_x = list(data_x)
                    data_y = list(data_y)
                    data_x.append(np.nan)  # necessary to draw multiple ellipses at once
                    data_y.append(np.nan)
                    data[n]["x"][k] = data_x
                    data[n]["y"][k] = data_y

                elif method == "particles":

                    data_xy = state.state_vector[mapping[:2], :]
                    data[n]["x"][k] = data_xy[0]
                    data[n]["y"][k] = data_xy[1]

                else:
                    raise ValueError("Should be 'uncertainty' or 'particles'")

        for frame in self.fig.frames:

            frame_time = datetime.fromisoformat(frame.name)

            data_ = list(frame.data)  # current data in frame
            traces_ = list(frame.traces)  # current traces in frame

            data_.append(go.Scatter(x=[-np.inf], y=[np.inf]))  # add empty data for legend trace
            traces_.append(trace_base - len(tracks) - 1)  # ensure correct trace

            for n, track in enumerate(tracks):
                # now plot the data
                _x = list(chain(*data[n]["x"][tuple(self.all_masks[frame_time][n])]))
                _y = list(chain(*data[n]["y"][tuple(self.all_masks[frame_time][n])]))
                _x.append(np.inf)
                _y.append(np.inf)
                data_.append(go.Scatter(x=_x, y=_y))
                traces_.append(trace_base - len(tracks) + n)

            frame.data = data_
            frame.traces = traces_

        if resize:
            self._resize(data, type="particle_or_uncertainty")

    def plot_sensors(self, sensors, label="Sensors", resize=True, **kwargs):
        """Plots sensor(s)

        Plots sensors.  Users can change the color and marker of detections using keyword
        arguments. Default is a black 'x' marker. Currently only works for stationary
        sensors.

        Parameters
        ----------
        sensors : Collection of :class:`~.Sensor`
            Sensors to plot
        label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for detections. Defaults are
            ``marker=dict(symbol='x', color='black')``.


        .. deprecated:: 1.5
           ``label`` has replaced ``sensor_label``. In the current implementation
           ``sensor_label`` overrides ``label``. However, use of ``sensor_label``
           may be removed in the future.
        """
        label = kwargs.pop('sensor_label', None) or label

        if not isinstance(sensors, Collection):
            sensors = {sensors}

        # don't run any of this if there is no data input
        if sensors:
            trace_base = len(self.fig.data)  # number of traces currently in figure
            sensor_kwargs = dict(mode='markers', marker=dict(symbol='x', color='black'),
                                 legendgroup=label, legendrank=50,
                                 name=label, showlegend=True)
            merge(sensor_kwargs, kwargs)

            self.fig.add_trace(go.Scatter(sensor_kwargs), row=1, col=1)  # initialises trace

            # sensor position
            sensor_xy = np.array([sensor.position[[0, 1], 0] for sensor in sensors])
            if resize:
                self._resize(sensors, "sensor")

            for frame in self.fig.frames:  # the plotting bit
                traces_ = list(frame.traces)
                data_ = list(frame.data)

                data_.append(go.Scatter(x=sensor_xy[:, 0], y=sensor_xy[:, 1]))
                traces_.append(trace_base)

                frame.traces = traces_
                frame.data = data_

        # we have called a plotting function so update flag (used in _resize)
        self.plotting_function_called = True