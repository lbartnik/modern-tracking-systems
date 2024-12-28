import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union

from .target import Target


__all__ = ['NearConstantVelocityTarget']


class NearConstantVelocityTarget(Target):
    """Near-Constant-Velocity target. Approximates a continuous-time random process
    where acceleration is models as (continuous-time) white noise.
    
    Described in "Estimation with Applications to Tracking and Navigation", pp. 269-270."""

    def __init__(self, speed: float = 30, initial_position: ArrayLike = [0, 0, 0], noise_intensity: float = 0,
                 seed: int = None, report: str = "position+velocity", integration_steps_count: int = 100):
        """Initialize target generator.

        The random seed defaults to None such that this target by default matches statistical properties
        of the Near-Constant Velocity Motion Model when executed across multiple Monte-Carlo trials.
        Specifically, in order to get point-in-time NEES means from Monte-Carlo samples to match the
        confidence interval predicted from Chi-squared distribution, both measurement noise and process
        noise must be random. If only the measurement noise is random (controlled by the sensor) but
        the process noise (controlled by this class) is not, then Monte-Carlo trials are not in fact
        distributed according to the Chi-squared distribution given the R and Q covariance matrices.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
            initial_position (ArrayLike): Initial position of the target.
            noise_intensity (float): Noise intensity. Physical unit is [length]^2 / [time]^3
            seed (int): Seed for random generator (used when noise intensity is non-zero).
            report (str): State parts to report. Accepted values as "position" and "position+velocity".
            integration_steps_count (int): Approximate the continuous-time white-noise acceleration by
                integrating over this many steps.
        """
        self.speed = float(speed)
        self.velocity = np.array([1, 0, 0]) # velocity direction, unit vector
        self.spatial_dim = 3
        self.initial_position = np.array(initial_position)
        self.seed = seed

        assert noise_intensity >= 0
        self.noise_intensity = noise_intensity
        self.integration_steps_count = integration_steps_count

        assert report in ['position', 'position+velocity']
        self.report = report

    @property
    def name(self):
        if self.ncv is not None:
            return f"ncv_{self.seed}"
        else:
            return "ncv_random"

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
            if self.noise_intensity > 0:
                dt = T / self.integration_steps_count
                sigma = np.sqrt(self.noise_intensity * dt)
                for _ in range(self.integration_steps_count):
                    vel += rng.normal(0, sigma, 3)
                    current_pos = current_pos + vel * dt
            else:
                current_pos = current_pos + vel * dt
            
            if self.report == 'position+velocity':
                states.append(np.concatenate((current_pos, vel)))
            else:
                states.append(current_pos)

        return np.array(states)
