import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union


__all__ = ['Target', 'ConstantVelocityTarget']


class Target(object):
    spatial_dim: int
    seed: int

    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None) -> np.ndarray:
        """Generate target states.

        Args:
            T (Union[float, ArrayLike]): Sampling interval or array of specific timestamps.
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Raises:
            Exception: If not implemented by the subclass.

        Returns:
            np.ndarray: (n, 6) array of states.
        """
        raise Exception(f"Target {self.__class__.__name__} does not implement true_states()")

    def position_at_time(self, t: float) -> List[np.ndarray]:
        raise Exception(f"Target {self.__class__.__name__} does not implement position_at_time()")


class ConstantVelocityTarget(Target):
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
        """Target model name"""
        return "cv"

    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None) -> np.ndarray:
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
            T = np.arange(0, n, T)
        else:
            T = np.array(T)

        # time is absolute and always starts at zero; this is so that elsewhere target
        # positions can be queried starting at arbitrary timestamp and yet return
        # values consistent across multiple trackers
        for dt in np.concatenate(([T[0]], np.diff(T))):
            current_pos = current_pos + vel * dt
            
            if self.report == 'position+velocity':
                states.append(np.concatenate((current_pos, vel)))
            else:
                states.append(current_pos)

        return np.array(states)

    def position_at_time(self, t: float) -> List[np.ndarray]:
        return self.initial_position + t * (self.velocity * self.speed)

