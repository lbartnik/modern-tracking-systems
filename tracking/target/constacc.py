import numpy as np
import pandas as pd
from ..util import to_df


class ConstantAccelerationTarget:
    def __init__(self, acceleration: float = 0.1):
        """Initialize target generator.

        Args:
            acceleration (float, optional): Linear acceleration, in m/s^2. Defaults to 0.1.
        """
        a = np.array([1, 0, 0]) # velocity direction, unit vector
        self.acceleration = a * acceleration
        self.acceleration_scalar = acceleration

    @property
    def name(self):
        """Target model name"""
        return f"ca_{self.acceleration_scalar}"

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
        current_vel = 0

        for _ in range(n):
            positions.append(current_pos)
            current_vel += self.acceleration * T
            current_pos = current_pos + current_vel * T + self.acceleration * T**2 / 2
            current_time += T

        return np.array(positions)

    def positions_df(self, T: float = 1, n: int = 400, seed: int = 0) -> pd.DataFrame:
        return to_df(self.positions(T, n, seed), columns=['x','y','z'])
