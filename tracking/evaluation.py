import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.typing import ArrayLike
from typing import List, Tuple, Union

from .simulation import EvaluationResult
from .util import colorscale


__all__ = ['Track', 'create_track', 'create_truth_track', 'compare_tracks']


class Track(object):
    def __init__(self, time: ArrayLike, positions: ArrayLike, position_covariance: ArrayLike = None):
        time = np.array(time)
        positions = np.array(positions)

        # time is a single-column vector
        assert len(time.shape) == 1
        # positions match the number of timestamps and contain at least 3 dimensions
        # (positions might come from the Kalman Filter state vector, and contain also
        # velocity, acceleration, etc.)
        assert len(positions.shape) == 2
        assert positions.shape[0] == time.shape[0]
        assert positions.shape[1] >= 3

        self.time = time
        self.positions = positions[:, :3]

        if position_covariance is not None:
            position_covariance = np.array(position_covariance)

            # position covariance is an array of D x D matrices, where D (the number of rows
            # and columns) is at least 3; it also matches the number of timestamps
            assert len(position_covariance.shape) == 3
            assert position_covariance.shape[0] == time.shape[0]
            assert position_covariance.shape[1] >= 3
            assert position_covariance.shape[2] >= 3

            self.position_covariance = position_covariance[:, :3, :3]
        else:
            self.position_covariance = None

    def interpolate(self, other: Union["Track", ArrayLike]) -> "Track":
        if isinstance(other, Track):
            time = other.time
        else:
            time = np.array(other)
            other = None
        assert len(time.shape) == 1

        positions = [np.interp(time, self.time, column, np.nan, np.nan) for column in self.positions.T]
        covariance = [np.interp(time, self.time, column, np.nan, np.nan) for column in self.position_covariance.reshape((-1, 9)).T]
        return InterpolatedTrack(other, time, np.array(positions).T, np.array(covariance).T.reshape((-1, 3, 3)))


class InterpolatedTrack(Track):
    def __init__(self, reference, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = reference



def create_track(x: EvaluationResult) -> Track:
    """Produce a Track from the given Evaluation Result.

    Args:
        x (Union[EvaluationResult, ArrayLike]): Evaluation Result of an array of timestamps.

    Raises:
        Exception: if `x` is not an Evaluation Result.

    Returns:
        Track: Estimated rack.
    """
    if isinstance(x, EvaluationResult):
        time = np.arange(x.x_hat.shape[0]) * x.T
        return Track(time, x.x_hat, x.P_hat)
    else:
        raise Exception("Unable to create a track from provided inputs")


def create_truth_track(x: EvaluationResult) -> Track:
    """Produce a Truth Track from the given Evaluation Result.

    Args:
        x (EvaluationResult): A single Evaluation Result.

    Raises:
        Exception: if `x` is not an Evaluation Result.

    Returns:
        Track: Truth track.
    """
    if isinstance(x, EvaluationResult):
        time = np.arange(x.truth.shape[0]) * x.T
        return Track(time, x.truth)
    else:
        raise Exception("Unable to create a truth track from provided inputs")


# see https://stackoverflow.com/a/53265922
def _mask_to_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Given boolean mask array, calculate ranges of True values.

    Args:
        mask (np.ndarray): Boolean array.

    Returns:
        List[Tuple[int, int]]: List of tuples of indices, each denoting a range of True values in mask.
    """
    mask = np.array(mask)
    mask_ext = np.r_[False, mask]
    indices = np.flatnonzero(mask_ext[:-1] != mask_ext[1:])
    if len(indices) % 2:
        indices = np.r_[indices, len(mask)-1]
    return indices.reshape((-1, 2))


def _calculate_x_of_intersect(a1: float, a2: float, b1: float, b2: float) -> float:
    """Calculate X coordinate of intersection given Y coordinates of two lines.

    Args:
        a1 (float): First Y value, line 1.
        a2 (float): Second Y value, line 1.
        b1 (float): First Y value, line 2.
        b2 (float): Second Y value, line 2.

    Returns:
        float: Coordinate X of the intersection of those two lines.
    """
    return (a1-b1) / ((b2-b1) - (a2-a1))


def _calculate_highlight_regions(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Calculate coordinates of highlight regions where y2 > y1.

    Args:
        x (np.ndarray): X coordinates matching y1 and y2.
        y1 (np.ndarray): Y coordinates of plot 1.
        y2 (np.ndarray): Y coordinates of plot 2.

    Returns:
        List[Tuple[int, int, int, int]]: Coordinates of highlight rectangles.
    """
    x, y1, y2 = np.array(x), np.array(y1), np.array(y2)
    lower = np.min((np.min(y1), np.min(y2)))
    upper = np.max((np.max(y1), np.max(y2)))

    #return _mask_to_ranges(y1 < y2)

    ans = []
    for begin_idx, end_idx in _mask_to_ranges(y1 < y2):
        if begin_idx > 0:
            x_start = _calculate_x_of_intersect(y1[begin_idx-1], y1[begin_idx], y2[begin_idx-1], y2[begin_idx])
            x_start += x[begin_idx-1]
        else:
            x_start = x[0]

        if end_idx < len(x)-1:
            x_end = _calculate_x_of_intersect(y1[end_idx-1], y1[end_idx], y2[end_idx-1], y2[end_idx])
            x_end += x[end_idx-1]
        else:
            x_end = x[-1]

        ans.append((x_start, lower, x_end, upper))
    
    return ans


def _plot_lines_with_highlights(fig: go.Figure, x: np.ndarray, y1: np.ndarray, y2: np.ndarray, show_legend: bool = True):
    """Plot two lines with highlight regions.

    Args:
        fig (go.Figure): Plotly figure object to draw on.
        x (np.ndarray): X coordinates.
        y1 (np.ndarray): Y coordinates of the first line.
        y2 (np.ndarray): Y coordinates of the second line.
        show_legend (bool): Whether to add legend to the plot.
    """
    for y, color, legend_id in zip([y1, y2], colorscale(n=2), ['a', 'b']):
        fig.add_trace(go.Scatter(
            x = x,
            y = y,
            marker=dict(
                color=color,
                size=3
            ),
            legendgroup=legend_id,
            name=legend_id,
            showlegend=show_legend
        ))

    for x0, y0, x1, y1 in _calculate_highlight_regions(x, y1, y2):
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1, fillcolor="rgba(255, 0, 0, .2)", line_width=0)


def compare_tracks(track_a: Track, track_b: Track, reference: Track) -> go.Figure:
    """Plot Euclidean distances of tracks A and B from a reference (truth) track.
    Highlight regions where track A is closer to the reference track.

    Args:
        track_a (Track): Track A.
        track_b (Track): Track B.
        reference (Track): Reference (truth) track.

    Returns:
        go.Figure: Plotly figure object.
    """
    track_a = track_a.interpolate(reference)
    track_b = track_b.interpolate(reference)

    dist_a = np.sqrt(np.power(track_a.positions - reference.positions, 2).sum(axis=1))
    dist_b = np.sqrt(np.power(track_b.positions - reference.positions, 2).sum(axis=1))

    return _plot_lines_with_highlights(reference.time, dist_a, dist_b)


class SubplotFigure(object):
    def __init__(self, fig: go.Figure, row: int, col: int):
        self.fig = fig
        self.row = row
        self.col = col
    
    def add_trace(self, *args, **kwargs):
        return self.fig.add_trace(row=self.row, col=self.col, *args, **kwargs)

    def add_shape(self, *args, **kwargs):
        return self.fig.add_shape(row=self.row, col=self.col, *args, **kwargs)


def compare_tracks_mahalanobis(track_a: Track, track_b: Track, reference: Track, mode: str = "agg") -> go.Figure:
    """Plot Mahalanobis distances of tracks A and B from a reference (truth) track.
    Highlight regions where track A is closer to the reference track.

    Args:
        track_a (Track): Track A.
        track_b (Track): Track B.
        reference (Track): Reference (truth) track.
        mode (str): Single plot of distance aggregated across dimensions ("agg")
                    or 3 plots, one for each dimension ("split").

    Returns:
        go.Figure: Plotly figure object.
    """
    assert track_a.position_covariance is not None
    assert track_b.position_covariance is not None
    assert reference.position_covariance is None

    track_a = track_a.interpolate(reference)
    track_b = track_b.interpolate(reference)

    if mode == "agg":
        mah_a = _mahalanobis(track_a.positions - reference.positions, track_a.position_covariance)
        mah_b = _mahalanobis(track_b.positions - reference.positions, track_b.position_covariance)

        fig = go.Figure()
        _plot_lines_with_highlights(fig, reference.time, mah_a, mah_b)
    else:
        fig = make_subplots(rows=3, cols=1, specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
                            shared_xaxes=True, row_titles=['x', 'y', 'z'])
        
        for i in (0, 1, 2):
            mah_a = np.abs(track_a.positions[:,i] - reference.positions[:,i]) / np.sqrt(track_a.position_covariance[:,i,i])
            mah_b = np.abs(track_b.positions[:,i] - reference.positions[:,i]) / np.sqrt(track_b.position_covariance[:,i,i])
            _plot_lines_with_highlights(SubplotFigure(fig, i+1, 1), reference.time, mah_a, mah_b, i<1)
        
    return fig


def _mahalanobis(diff: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance given Euclidean difference vectors and covariance matrices.

    Args:
        diff (np.ndarray): Euclidean difference vectors, of shape (N, 3).
        cov (np.ndarray): Covariance matrices matching distances, of shape (N, 3, 3).

    Returns:
        np.ndarray: Mahalanobis distances.
    """
    assert len(diff.shape) == 2
    assert len(cov.shape) == 3
    assert diff.shape[0] == cov.shape[0]

    ncol = diff.shape[1]
    diff = diff.reshape((-1, ncol, 1))
    diffT = np.transpose(diff, (0, 2, 1))
    covInv = np.linalg.inv(cov)

    # ( d^T * cov^-1 * d ) ^ 1/2
    distSq = np.matmul(np.matmul(diffT, covInv), diff)
    return np.sqrt(distSq.squeeze())


def mahalanobis_distance(track: Track, reference: Track) -> np.ndarray:
    """Compute Mahalanobis distance between two tracks.
     
    The track must provide estimated position mean and covariance, while the reference
    track must provide the true target position.

    Args:
        track (Track): Estimated track.
        reference (Track): Truth track.

    Returns:
        np.ndarray: A vector of Mahalanobis distance values along the reference timestamps.
    """
    assert track.position_covariance is not None
    assert reference.position_covariance is None
    track = track.interpolate(reference)
    return _mahalanobis(track.positions - reference.positions, track.position_covariance)
