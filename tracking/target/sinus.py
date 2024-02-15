import numpy as np
import pandas as pd
from ..util import to_df


class SinusTarget:
    def __init__(self, speed: float = 30, heading_change_rate: float = 2):
        """Initialize target generator.

        Args:
            speed (float, optional): Linear velocity, in m/s. Defaults to 30.
            heading_change_rate (float, optional): The rate at which heading changes. Defaults to 2.
        """
        self.speed = speed
        self.heading_change_rate = heading_change_rate

    @property
    def name(self):
        """Target model name"""
        return "sinus"

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
        current_heading = 0
        current_time = 0

        for _ in range(n):
            positions.append(current_pos)

            # heading changes between -45 and +45 degrees
            current_heading = np.sin(self.heading_change_rate/180.0*np.pi * current_time) * np.pi/4
            current_vel = self.speed * np.array([np.sin(current_heading), np.cos(current_heading), 0])
            current_pos = current_pos + current_vel * T

            current_time += T

        return np.array(positions)

    def positions_df(self, T: float = 1, n: int = 400, seed: int = 0) -> pd.DataFrame:
        return to_df(self.positions(), columns=['x','y','z'])
