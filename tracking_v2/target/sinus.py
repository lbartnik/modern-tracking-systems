import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from .target import Target


__all__ = ['SinusTarget']


class SinusTarget(Target):
    """Target which moves at a constant speed and alternates between moving in
    a straight line and making a 180-degree turn.
    """

    def __init__(self, speed: float = 30, heading_change_rate: float = 2):
        """Initialize target generator.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
            heading_change_rate (float, optional): The rate at which heading changes. Defaults to 2.
        """
        self.name = "sinus"

        self.speed = speed
        self.heading_change_rate = heading_change_rate


    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None) -> np.ndarray:
        """Generate target positions.

        Args:
            T (Union[float, ArrayLike]): Sampling interval or array of specific timestamps.
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            np.ndarray: (n, 3) array of positions.
        """        
        return _sigmoid_trace(_time_array(T, n), [0.0, 0.0, 0.0], self.speed, self.heading_change_rate)[:, :3]

    def position_at_time(self, t: float) -> np.ndarray:
        return self.true_states([t], 1)[0, :]
    
    def heading(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None) -> np.ndarray:
        return _sigmoid_trace(_time_array(T, n), [0.0, 0.0, 0.0], self.speed, self.heading_change_rate)[:, 6]


def _time_array(T: Union[float, ArrayLike], n: int) -> np.ndarray:
    if isinstance(T, (int, float)):
        return np.arange(0, n, T)
    else:
        return np.array(T)


def _sigmoid_trace(t: ArrayLike, initial_position: ArrayLike, speed: float, heading_change_rate: float) -> ArrayLike:
    """Generates trajectory of a target which alternates between moving in a straight
    line and making a 180-degree turn.

    Args:
        t (ArrayLike): Timestamps.
        initial_position (ArrayLike): 3D starting position of the target.
        speed (float): Constant speed of the target.
        heading_change_rate (float): Rate at which the U-turn is taken.

    Returns:
        ArrayLike: States, shape (N, 7): positions (3), velocity (3), heading (1).
    """
    slope_factor = 20  # how smoothly does each slope start and end
    period_parts = 4   # how many parts in each period (straight, U-turn, straight, U-turn)

    u_turn_time = 180 / heading_change_rate
    period = u_turn_time * period_parts

    t = np.array(t)
    t_full = np.arange(np.min(t), np.max(t)+1, .1)
    t_periodical = np.mod(t_full, period)
    
    upward_slope = _sigma(heading_change_rate/slope_factor * (t_periodical - .25 * period))
    downward_slope = _sigma(-heading_change_rate/slope_factor * (t_periodical - .75 * period))

    heading_0_1 = -3 + 2*(upward_slope + downward_slope)

    # heading changes between -90 and +90 degrees
    heading_rad = heading_0_1 * np.pi/2
    current_pos = initial_position
    states = []

    for current_heading, dt in zip(heading_rad, np.concatenate((np.diff(t_full), [0]))):
        current_vel = speed * np.array([np.sin(current_heading), np.cos(current_heading), 0])
        states.append(np.hstack((current_pos, current_vel, [current_heading])))
        current_pos = current_pos + current_vel * dt

    states = np.array(states)
    interpolated = [np.interp(t, t_full, states[:, i]) for i in range(states.shape[1])]

    return np.array(interpolated).T


def _sigma(x):
    return 1 / (1 + np.exp(-x))
