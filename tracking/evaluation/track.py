import numpy as np
from numpy.typing import ArrayLike
from typing import Union

from ..simulation import SimulationResult

__all__ = ['Track', 'create_track', 'create_truth_track']


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



def create_track(x: SimulationResult) -> Track:
    """Produce a Track from the given Evaluation Result.

    Args:
        x (Union[SimulationResult, ArrayLike]): Evaluation Result of an array of timestamps.

    Raises:
        Exception: if `x` is not an Evaluation Result.

    Returns:
        Track: Estimated rack.
    """
    if isinstance(x, SimulationResult):
        time = np.arange(x.x_hat.shape[0]) * x.T
        return Track(time, x.x_hat, x.P_hat)
    else:
        raise Exception("Unable to create a track from provided inputs")


def create_truth_track(x: SimulationResult) -> Track:
    """Produce a Truth Track from the given Evaluation Result.

    Args:
        x (SimulationResult): A single Evaluation Result.

    Raises:
        Exception: if `x` is not an Evaluation Result.

    Returns:
        Track: Truth track.
    """
    if isinstance(x, SimulationResult):
        time = np.arange(x.truth.shape[0]) * x.T
        return Track(time, x.truth)
    else:
        raise Exception("Unable to create a truth track from provided inputs")
