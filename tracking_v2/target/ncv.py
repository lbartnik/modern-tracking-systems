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

        The random seed defaults to `None` which means that each generation of target
        trajectory uses a different sequence of random white-noise acceleration values.
        This, in turn, means that each Monte-Carlo trial using this target class is
        executed against a different target path. Finally, this leads the NEES (Normalized
        Estimation Error Squared) of a Kalman Filter configured with the Near-Constant
        Velocity Motion Model, tracking this target, to follow the theoretical Chi-squared
        distribution. This means that the point-in-time NEES means from Monte-Carlo trials
        will fall into the confidence interval predicted from the Chi-squared distribution.

        Conversely, if the target path in each Monte-Carlo trial is exactly the same, and
        only the measurement noise (controlled by the sensor implementation) is random, then
        the covariance matrix estimated by the Kalman Filter will not match the actual error
        distribution. This will be observed as point-in-time means from Monte-Carlo falling
        out of the predicted confidence interval.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
            initial_position (ArrayLike): Initial position of the target.
            noise_intensity (float): Noise intensity. Physical unit is [length]^2 / [time]^3
            seed (int): Seed for random generator (used when noise intensity is non-zero).
            report (str): State parts to report. Accepted values as "position" and "position+velocity".
            integration_steps_count (int): Approximate the continuous-time white-noise acceleration by
                integrating over this many steps.
        """
        self.target_id = 0
        self.speed = float(speed)
        self.velocity = np.array([1, 0, 0]) # velocity direction, unit vector
        self.spatial_dim = 3
        self.initial_position = np.array(initial_position)
        self.reset_seed(seed)

        assert noise_intensity >= 0
        self.noise_intensity = noise_intensity
        self.integration_steps_count = integration_steps_count

        assert report in ['position', 'position+velocity']
        self.report = report
    
    def reset_seed(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def reset_rng(self, rng: np.random.Generator):
        self.seed = None
        self.rng = rng

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
            tm = np.arange(0, n) * T
        else:
            tm = np.array(T)
            T  = 1 # TODO better would be to use the most frequent value of np.diff(T)

        # time is absolute and always starts at zero; this is so that elsewhere target
        # positions can be queried starting at arbitrary timestamp and yet return
        # values consistent across multiple trackers
        for dt in np.concatenate(([tm[0]], np.diff(tm))):
            if self.noise_intensity > 0:
                dt = T / self.integration_steps_count
                sigma = np.sqrt(self.noise_intensity * dt)
                for _ in range(self.integration_steps_count):
                    vel += self.rng.normal(0, sigma, 3)
                    current_pos = current_pos + vel * dt
            else:
                current_pos = current_pos + vel * dt
            
            if self.report == 'position+velocity':
                states.append(np.concatenate((current_pos, vel)))
            else:
                states.append(current_pos)

        return np.array(states)
