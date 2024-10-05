import numpy as np

from .sensor import SensorMeasurement
from .kalman import KalmanFilter


__all__ = ['Module', 'SingleTargetTracker']


class Module:
    def update(self, *args):
        for arg in args:
            if isinstance(arg, SensorMeasurement):
                self.update_with_measurement(arg)
            else:
                raise Exception(f'Update: input type not supported: {type(arg)}')
    
    def update_with_measurement(self, m: SensorMeasurement):
        raise Exception(f'Module {self.__class__.__name__} does not implement update_with_measurement()')



class SingleTargetTracker(Module):
    def __init__(self, kalman_filter: KalmanFilter):
        self.kalman_filter = kalman_filter
        self.time = None
        self.track_updates = []
        self.filter_trace = []
    
    def update_with_measurement(self, m: SensorMeasurement):
        if self.time is None:
            self.time = m.time
            self.kalman_filter.initialize(m.z, m.R)
        else:
            assert m.time >= self.time
            dt = m.time - self.time

            self.time = m.time
            self.kalman_filter.predict(dt)
            self.kalman_filter.update(m.z, m.R)
        
        x_hat, P_hat = np.copy(self.kalman_filter.x_hat), np.copy(self.kalman_filter.P_hat)
        innovation = None if self.kalman_filter.innovation is None else np.copy(self.kalman_filter.innovation)

        self.track_updates.append((self.time, x_hat, P_hat))
        self.filter_trace.append((self.time, x_hat, P_hat, innovation))
