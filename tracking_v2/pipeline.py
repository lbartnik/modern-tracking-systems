import numpy as np
from numpy.typing import ArrayLike
from typing import List, Any


class Pipeline:
    def __init__(self, target, sensor, tracker):
        self.target = target
        self.sensor = sensor
        self.tracker = tracker
    
    def execute(self, timestamps: np.ndarray):
        for t in timestamps:
            true_position = self.target.position_at_time(t)
            measurement = self.sensor.generate_measurement(t, true_position)
            self.tracker.update(t, measurement)

