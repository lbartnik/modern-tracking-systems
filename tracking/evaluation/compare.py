import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

from .track import Track
from ..util import colorscale


__all__ = ['compare_tracks']


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
