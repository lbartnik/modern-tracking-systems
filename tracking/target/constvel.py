import numpy as np
import pandas as pd
from ..util import to_df


class ConstantVelocityTarget:
    def __init__(self, speed: float = 30):
        """Initialize target generator.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
        """
        self.speed = speed
        self.velocity = np.array([1, 0, 0]) # velocity direction, unit vector

    @property
    def name(self):
        """Target model name"""
        return "cv"

    def positions(self, T: float = 1, n: int = 400, seed: int = 0) -> np.ndarray:
        """Generate target positions.

        Args:
            T (float): Sampling interval.
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            np.ndarray: (n, 3) array of positions.
        """
        positions = []
        current_pos = np.array([0.0, 0.0, 0.0])
        current_time = 0
        vel = self.velocity * self.speed

        for _ in range(n):
            positions.append(current_pos)
            current_pos = current_pos + vel * T
            current_time += T

        return np.array(positions)

    def positions_df(self, T: float = 1, n: int = 400, seed: int = 0) -> pd.DataFrame:
        return to_df(self.positions(T, n, seed), columns=['x','y','z'])