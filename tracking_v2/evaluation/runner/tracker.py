import numpy as np
from typing import Callable, List

from ...callback import after_one, before_many, before_one, target_cached, tracks_estimated, execute_callback
from ...sensor import Sensor
from ...target import Target
from ...tracker import Tracker, Track

__all__ = ['TrackerRunner', 'TrackerCallback']



class TrackerCallback:
    def __init__(self):
        self.one_truths = None
        self.one_x_hat, self.one_P_hat = None, None
        self.many_x_hat, self.many_P_hat, self.many_truths = [], [], []

    @target_cached
    def register_truth(self, target_states):
        self.one_truths.append(target_states)

    @before_one
    def prepare_single_run(self):
        self.one_truths = []
        self.one_x_hat = {}
        self.one_P_hat = {}

    @after_one
    def process_single_run(self):
        self.one_truths = np.asarray(self.one_truths)
        self.one_x_hat = {tid: np.asarray(updates) for tid, updates in self.one_x_hat.items()}
        self.one_P_hat = {tid: np.asarray(updates) for tid, updates in self.one_P_hat.items()}

        self.many_truths.append(self.one_truths)
        self.many_x_hat.append(self.one_x_hat)
        self.many_P_hat.append(self.one_P_hat)

    @before_many
    def prepare_multiple_runs(self):
        self.many_truths = []
        self.many_x_hat = []
        self.many_P_hat = []

    @tracks_estimated
    def collect_track_states(self, tracks: List[Track]):
        for track in tracks:
            self.one_x_hat.setdefault(track.track_id, []).append(track.mean)
            self.one_P_hat.setdefault(track.track_id, []).append(track.cov)



class TrackerRunner:
    def __init__(self, targets: List[Target], sensors: List[Sensor], tracker: Tracker, callbacks: List[Callable] = 'standard_callbacks'):
        self.tracker = tracker
        self.targets = targets
        self.sensors = sensors

        self.n = None
        self.m = None
        self.seeds = None

        if callbacks == 'standard_callbacks':
            callbacks = [TrackerCallback()]
        self.callbacks = callbacks

        assert len(np.unique([t.target_id for t in self.targets])) == len(self.targets), "target ids not unique"
        assert len(np.unique([s.sensor_id for s in self.sensors])) == len(self.sensors), "sensor ids not unique"

    
    def run_one(self, n: int, T: float = 1):
        t = 0
        self.n = n
        
        for cb in self.callbacks:
            execute_callback(cb, 'before_one')
    
        for target in self.targets:
            target.cache(T, n+1)
            for cb in self.callbacks:
                execute_callback(cb, 'target_cached', target.cached_states)
                
        self.tracker.reset()

        for t in np.arange(1, n+1) * T:
            for sensor in self.sensors:
                measurements = []
                for target in self.targets:
                    m = sensor.generate_measurement(t, target)
                    measurements.append(m)
                
                self.tracker.add_measurements(measurements, callbacks=self.callbacks)
            
            for cb in self.callbacks:
                tracks = self.tracker.estimate_tracks(t)
                execute_callback(cb, 'tracks_estimated', tracks)

        for cb in self.callbacks:
            execute_callback(cb, 'after_one')
    

    def run_many(self, m: int, n: int, T: float = 1, seeds: List[int] = None):
        if seeds is None:
            seeds = np.arange(m)
        else:
            assert len(seeds) == m, f"Number of seeds not equal to m={m}"
        
        self.n = n
        self.m = m
        self.seeds = seeds

        for cb in self.callbacks:
            execute_callback(cb, 'before_many')

        for seed in seeds:
            rng = np.random.default_rng(seed=seed)

            for sensor in self.sensors:
                sensor.reset_rng(rng)
            for target in self.targets:
                target.reset_rng(rng)

            self.run_one(n, T)
        
        for cb in self.callbacks:
            execute_callback(cb, 'after_many')
