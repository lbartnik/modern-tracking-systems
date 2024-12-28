import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union

from .target import Target


__all__ = ['ConstantVelocityTarget']


class ConstantVelocityTarget(Target):
    """Constant-velocity target."""

    def __init__(self, speed: float = 30, initial_position: ArrayLike = [0, 0, 0], report: str = "position+velocity"):
        """Initialize target generator.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
            initial_position (ArrayLike): Initial position of the target.
            report (str): State parts to report. Accepted values as "position" and "position+velocity".
        """
        self.speed = float(speed)
        self.velocity = np.array([1, 0, 0]) # velocity direction, unit vector
        self.spatial_dim = 3
        self.initial_position = np.array(initial_position)

        assert report in ['position', 'position+velocity']
        self.report = report

    @property
    def name(self):
        return f"cv_{self.speed}"

    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400) -> np.ndarray:
        """Generate target states.

        Args:
            T (Union[float, ArrayLike]): Sampling interval or array of specific timestamps.
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            np.ndarray: (n, 6) array of states.
        """
        states = []
        current_pos = self.initial_position
        vel = self.velocity * self.speed

        if isinstance(T, (int, float)):
            tm = np.arange(0, n, T)
        else:
            tm = np.array(T)
            T  = 1 # TODO better would be to use the most frequent value of np.diff(T)

        # time is absolute and always starts at zero; this is so that elsewhere target
        # positions can be queried starting at arbitrary timestamp and yet return
        # values consistent across multiple trackers
        for dt in np.concatenate(([tm[0]], np.diff(tm))):
            current_pos = current_pos + vel * dt

            if self.report == 'position+velocity':
                states.append(np.concatenate((current_pos, vel)))
            else:
                states.append(current_pos)

        return np.array(states)
