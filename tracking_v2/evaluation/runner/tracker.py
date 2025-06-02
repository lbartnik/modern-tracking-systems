import numpy as np
from typing import List

from .base import Runner
from ...sensor import Sensor
from ...target import Target
from ...tracker import Tracker, Track

__all__ = []


class TrackerRunner(Runner):
    def __init__(self, targets: List[Target], sensors: List[Sensor], tracker: Tracker):
        self.tracker = tracker
        self.targets = targets
        self.sensors = sensors

        self.n = None
        self.m = None
        self.seeds = None

        self.one_truths = None
        self.one_x_hat, self.one_P_hat = None, None
        self.many_x_hat, self.many_P_hat = [], []

        assert len(np.unique([t.target_id for t in self.targets])) == len(self.targets), "target ids not unique"
        assert len(np.unique([s.sensor_id for s in self.sensors])) == len(self.sensors), "sensor ids not unique"

    
    def run_one(self, n: int, T: float = 1):
        t = 0
        self.n = n
        
        self.before_one()
    
        for target in self.targets:
            target.cache(T, n+1)
        
        for target in self.targets:
            self.one_truths.append(target.cached_states)
        
        self.tracker.reset()

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
    

    def run_many(self, m: int, n: int, T: float = 1, seeds: List[int] = None):
        if seeds is None:
            seeds = np.arange(m)
        else:
            assert len(seeds) == m, f"Number of seeds not equal to m={m}"
        
        self.n = n
        self.m = m
        self.seeds = seeds

        self.before_many()
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)

            for sensor in self.sensors:
                sensor.reset_rng(rng)
            for target in self.targets:
                target.reset_rng(rng)

            self.run_one(n, T)
        
        self.after_many()

    def before_one(self):
        self.one_truths = []
        self.one_x_hat = {}
        self.one_P_hat = {}
        self._execute_user_callbacks('before_one')

    def after_one(self):
        self.one_truths = np.asarray(self.one_truths)
        self.one_x_hat = {tid: np.asarray(updates) for tid, updates in self.one_x_hat.items()}
        self.one_P_hat = {tid: np.asarray(updates) for tid, updates in self.one_P_hat.items()}

        self.many_truths.append(self.one_truths)
        self.many_x_hat.append(self.one_x_hat)
        self.many_P_hat.append(self.one_P_hat)

        self._execute_user_callbacks('after_one')

    def before_many(self):
        self.many_truths = []
        self.many_x_hat = []
        self.many_P_hat = []

        self._execute_user_callbacks('before_many')

    def after_many(self):
        self._execute_user_callbacks('after_many')

    def after_estimate(self, tracks: List[Track]):
        for track in tracks:
            self.one_x_hat.setdefault(track.track_id, []).append(track.mean)
            self.one_P_hat.setdefault(track.track_id, []).append(track.cov)

        self._execute_user_callbacks('after_estimate', tracks)


