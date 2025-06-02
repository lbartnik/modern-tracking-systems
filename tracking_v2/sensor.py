import numpy as np
from numpy.typing import ArrayLike

from .target import Target


class SensorMeasurement:
    def __init__(self, target_id: int, time: float, z: ArrayLike, R: ArrayLike, error: ArrayLike):
        """Construct a sensor measurement.

        Args:
            time (float): Time of the measurement.
            z (ArrayLike): Measurement.
            R (ArrayLike): Measurement error covariance matrix.
        """
        self.target_id = target_id
        self.time = time
        self.z = np.asarray(z)
        self.R = np.asarray(R)
        self.error = np.asarray(error)

        assert self.z.squeeze().shape[0] == self.R.squeeze().shape[0]
        assert self.R.squeeze().shape[0] == self.R.squeeze().shape[1]


class Sensor(object):
    sensor_id: int

    def generate_measurement(self, t: float, target: Target) -> SensorMeasurement:
       raise Exception(f"generate_measurement() method not implemented in {self.__class__.__name__}")



class GeometricSensor(Sensor):
    spatial_dim: int

    def __init__(self, R: np.ndarray = np.eye(3), seed: int = 12345, sensor_id: int = 0):
        self.sensor_id = sensor_id
        self.spatial_dim = 3

        self.R = np.array(R)
        assert self.R.shape == (self.spatial_dim, self.spatial_dim)
        
        self.reset_rng(np.random.default_rng(seed=seed))
        
    def reset_rng(self, rng: np.random.Generator = None):
        self.rng = rng

    def generate_measurement(self, t: float, target: Target) -> SensorMeasurement:
        position = target.true_state(t)
        position = np.asarray(position).squeeze()
        position = position[:self.spatial_dim]

        error = self.rng.multivariate_normal(np.zeros(self.spatial_dim), self.R, size=1)
        measurement = position + error
        return SensorMeasurement(target.id, t, measurement, self.R, error)
