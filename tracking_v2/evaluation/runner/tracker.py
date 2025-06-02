import numpy as np
from typing import List

from .base import Runner
from ...sensor import Sensor
from ...target import Target
from ...tracker import Tracker

__all__ = []


class TrackerRunner(Runner):
    def __init__(self, tracker: Tracker, targets: List[Target], sensors: List[Sensor]):
        self.tracker = tracker
        self.targets = targets
        self.sensors = sensors

        self.truths = None
        self.n = None
        self.m = None
        self.seeds = None

        assert len(np.unique([t.target_id for t in self.targets])) == len(self.targets), "target ids not unique"
        assert len(np.unique([s.sensor_id for s in self.sensors])) == len(self.sensors), "sensor ids not unique"

    
    def run_one(self, n: int = 400, T: float = 1):
        t = 0
        self.n = n
        
        self.before_one()
    
        for target in self.targets:
            target.cache(T, n+1)
        
        for target in self.targets:
            self.truths.append(target.cached_states)
        
        for t in np.arange(1, n+1) * T:
            for sensor in self.sensors:
                measurements = []
                for target in self.targets:
                    m = sensor.generate_measurement(t, target)
                    measurements.append(m)
                
                self.tracker.add_measurements(measurements)
            
            tracks = self.tracker.estimate_tracks(t)
            self.after_estimate(tracks)

        self.after_one()


    def before_one(self):
        self._execute_user_callbacks('before_one')

    def after_one(self):
        self._execute_user_callbacks('after_one')

    def after_estimate(self):
        self._execute_user_callbacks('after_estimate')


