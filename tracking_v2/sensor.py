import numpy as np
from numpy.typing import ArrayLike
from typing import List


class SensorMeasurement:
    def __init__(self, time: float, z: ArrayLike, R: ArrayLike):
        """Construct a sensor measurement.

        Args:
            time (float): Time of the measurement.
            z (ArrayLike): Measurement.
            R (ArrayLike): Measurement error covariance matrix.
        """
        self.time = time
        self.z = np.array(z)
        self.R = np.array(R)

        assert self.z.squeeze().shape[0] == self.R.squeeze().shape[0]
        assert self.R.squeeze().shape[0] == self.R.squeeze().shape[1]


class GeometricSensor:
    spatial_dim: int

    def __init__(self, R: np.ndarray = np.eye(3), seed: int = None):
        self.spatial_dim = 3
        self.R = np.array(R)
        assert self.R.shape == (self.spatial_dim, self.spatial_dim)

        self.rng = np.random.default_rng(seed=seed)


    def generate_measurement(self, t: float, position: np.ndarray) -> SensorMeasurement:
        position = np.array(position).squeeze()
        assert position.shape == (self.spatial_dim,), f"Position shape ({position.shape}) not equal to spatial dim {self.spatial_dim}"

        measurement = self.rng.multivariate_normal(position, self.R, size=1)
        return SensorMeasurement(t, measurement, self.R)
