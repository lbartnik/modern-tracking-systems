import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from typing import List, Tuple, Union

from .simulation import EvaluationResult


__all__ = ['Track', 'create_track', 'create_truth_track', 'compare_tracks']


class Track(object):
    def __init__(self, time: ArrayLike, positions: ArrayLike):
        time = np.array(time)
        positions = np.array(positions)

        assert len(time.shape) == 1
        assert positions.shape[0] == time.shape[0]
        assert positions.shape[1] >= 3

        self.time = time
        self.positions = positions[:,:3]

    def interpolate(self, other: Union["Track", ArrayLike]) -> "Track":
        if isinstance(other, Track):
            time = other.time
        else:
            time = np.array(other)
        assert len(time.shape) == 1

        positions = [np.interp(time, self.time, column, np.nan, np.nan) for column in self.positions.T]
        return Track(time, np.array(positions).T)


def create_track(*args) -> Track:
    if len(args) == 1 and isinstance(args[0], EvaluationResult):
        time = np.arange(args[0].x_hat.shape[0]) * args[0].T
        return Track(time, args[0].x_hat)
    elif len(args) == 2:
        return Track(*args)
    else:
        raise Exception("Unable to create a track from provided inputs")


def create_truth_track(*args) -> Track:
    if len(args) == 1 and isinstance(args[0], EvaluationResult):
        time = np.arange(args[0].truth.shape[0]) * args[0].T
        return Track(time, args[0].truth)
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


def _plot_lines_with_highlights(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> go.Figure:
    """Plot two lines with highlight regions.

    Args:
        x (np.ndarray): X coordinates.
        y1 (np.ndarray): Y coordinates of the first line.
        y2 (np.ndarray): Y coordinates of the second line.

    Returns:
        go.Figure: Plotly figure object.
    """
    data = pd.DataFrame(dict(time=x, a=y1, b=y2))
    data = data.melt(['time'], None, 'track', 'dist')

    fig = ex.line(data, x='time', y='dist', color='track')

    #return _calculate_highlight_regions(x, y1, y2)

    for x0, y0, x1, y1 in _calculate_highlight_regions(x, y1, y2):
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1, fillcolor="rgba(255, 0, 0, .2)", line_width=0)
    
    return fig


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
