import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union

from .target import Target


__all__ = ['ConstantVelocityTarget']


class ConstantVelocityTarget(Target):
    """Constant-velocity target.
    
    Described in "Estimation with Applications to Tracking and Navigation", pp. 269-270."""

    def __init__(self, speed: float = 30, initial_position: ArrayLike = [0, 0, 0], noise_intensity: float = 0,
                 seed: int = 1, report: str = "position+velocity"):
        """Initialize target generator.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
            initial_position (ArrayLike): Initial position of the target.
            noise_intensity (float): Noise intensity.
            seed (int): Seed for random generator (used when noise intensity is non-zero).
            report (str): State parts to report. Accepted values as "position" and "position+velocity".
        """
        self.speed = float(speed)
        self.velocity = np.array([1, 0, 0]) # velocity direction, unit vector
        self.spatial_dim = 3
        self.initial_position = np.array(initial_position)
        self.seed = seed

        assert noise_intensity >= 0
        self.noise_intensity = noise_intensity

        assert report in ['position', 'position+velocity']
        self.report = report

    @property
    def name(self):
        if self.noise_intensity > 0:
            return f"cv_{self.seed}"
        else:
            return "cv"

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

        rng = np.random.default_rng(seed=self.seed)

        # time is absolute and always starts at zero; this is so that elsewhere target
        # positions can be queried starting at arbitrary timestamp and yet return
        # values consistent across multiple trackers
        for dt in np.concatenate(([tm[0]], np.diff(tm))):
            current_pos = current_pos + vel * dt

            if self.noise_intensity > 0:
                vel += rng.normal(0, T * self.noise_intensity, size=3)
            
            if self.report == 'position+velocity':
                states.append(np.concatenate((current_pos, vel)))
            else:
                states.append(current_pos)

        return np.array(states)
