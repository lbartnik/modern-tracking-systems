import numpy as np
import scipy as sp
from math import log, pi
from typing import Callable, List

from ..np import as_column 
from .base import Tracker, Track, initialize_velocity
from ..kalman import linear_ncv, KalmanFilter
from ..sensor import SensorMeasurement
from ..callback import execute_callback


__all__ = ['MultiTargetTracker', 'InitializeTrack', 'MaybeNewTrack', 'UpdateTrack', 'TrackNotUpdated']


class TrackHypothesis:
    def __init__(self, track_id: int, kf: KalmanFilter, time: float, llr: float):
        self.track_id = track_id
        self.kf = kf
        self.time = time
        self.llr = llr
        self.target_id = None
    
    def __repr__(self):
        with np.printoptions(precision=2):
            return f"TrackHypothesis({self.track_id}, {self.time}, {round(self.llr, 2)}, {self.kf.x_hat.T.squeeze()})"

# decisions.append(('initialize_new_track', LLR_2, t, um, m))
class InitializeTrack(object):
    def __init__(self, track: TrackHypothesis, measurement0: SensorMeasurement, measurement1: SensorMeasurement, LLR: float):
        self.track = track
        self.measurement0 = measurement0
        self.measurement1 = measurement1
        self.LLR = LLR

    def __repr__(self):
        return f"InitializeTrack({round(self.LLR, 2)}, {self.track.track_id}, " + \
               f"{self.measurement0.measurement_id}({self.measurement0.target_id}), " + \
               f"{self.measurement1.measurement_id}({self.measurement1.target_id}))"

# decisions.append(('measurement_not_associated', llr, m))
class MaybeNewTrack(object):
    def __init__(self, measurement: SensorMeasurement, LLR: float):
        self.measurement = measurement
        self.LLR = LLR
    
    def __repr__(self):
        return f"MaybeNewTrack({round(self.LLR, 2)}, {self.measurement.measurement_id}({self.measurement.target_id}))"


# decisions.append(('associate_to_track', t.llr + dLLR, t, m))
class UpdateTrack(object):
    def __init__(self, track: TrackHypothesis, measurement: SensorMeasurement, dLLR: float):
        self.track = track
        self.measurement = measurement
        self.dLLR = dLLR
        self.LLR = track.llr + dLLR
    
    def __repr__(self):
        return f"UpdateTrack({round(self.LLR, 2)}, {self.track.track_id}({self.track.target_id}), {self.measurement.measurement_id}({self.measurement.target_id}))"

# decisions.append(('track_not_updated', t.llr + dLLR, t))
class TrackNotUpdated(object):
    def __init__(self, track: TrackHypothesis, LLR: float):
        self.track = track
        self.LLR = LLR

    def __repr__(self):
        return f"TrackNotUpdated({round(self.LLR, 2)}, {self.track.track_id}({self.track.target_id}))"



class MultiTargetTracker(Tracker):
    def __init__(self):
        self.reset()
    
        
    def reset(self):
        self.track_id_generator = 0

        self.tracks: List[TrackHypothesis] = []
        self.unassociated_measurements = []

        # general LLR parameters
        self.P_d = 1  # probability of detection
        self.B_NT = 1e-5 # probability of new target
        self.B_FT = 1e-100 # probability of false target

        # two-point initialization LLR parameters
        self.v_max = 100
    
    def next_track_id(self):
        self.track_id_generator += 1
        return self.track_id_generator
    
    def add_measurements(self, ms: List[SensorMeasurement], callbacks: List[Callable] = None):
        if len(ms) == 0:
            return
        
        dim = len(ms[0].z.squeeze())
        assert dim == 3, "only 3D spaces are supported"

        if callbacks is None:
            callbacks = []

        # possible decisions (mutually exclusive only if concerning the same track or measurement):
        #   - track update with a new measurement
        #   - new track from a previously unassociated measurement and a new measurement
        #   - new measurement is left unassociated
        #   - an existing track does not have an update in this iteration
        #   - old measurement is false
        decisions = []

        # 1. for each new measurement
        for m in ms:
            # 1.1. score measurement against track
            for t in self.tracks:
                # TODO consider only if passes a gate
                if t.time < m.time:
                    t.kf.predict(m.time - t.time)
                    t.time = m.time
                
                assert t.time == m.time, "Track is ahead of measurement"

                t.kf.calculate_innvovation(m.z, m.R)
                
                S_inv = np.linalg.inv(t.kf.S)
                d2 = t.kf.innovation.T @ S_inv @ t.kf.innovation
                
                dLLR = log(self.P_d / self.B_FT) - dim/2*log(2*pi) - .5 * log(np.linalg.det(t.kf.S)) - .5 * d2
                dLLR = float(dLLR)
                
                d = UpdateTrack(t, m, dLLR)
                decisions.append(d)

                _execute_callbacks(callbacks, 'consider_update_track', self, d)

            # 1.2. score measurement against initiating a new track, using two-point initialization
            for um in self.unassociated_measurements:
                # approximating the velocity prior as uniform within a ball of radius V_max
                # (arbitrary direction, bounded speed), yielding per-component variance V_max^2/(m+2)
                P_vv = np.eye(dim) * self.v_max ** 2 / (dim + 2)

                # predict to the second measurement, F = constant velocity, Q = no process noise
                dt = m.time - um.time

                # innovation and its covariance
                v = as_column(m.z - um.z)
                S = dt*dt * P_vv + um.R + m.R

                # skip if velocity required to move is out of bounds
                if np.linalg.norm(v) > self.v_max * dt:
                    continue

                # Chi-squared gate
                d2 = v.T @ np.linalg.inv(S) @ v
                G = float(sp.stats.chi2.ppf(.95, dim))

                if d2 > G:
                    continue

                # accepted; start by calculating the initial LLR due to initializing a track with um
                LLR_1 = log(self.P_d) + log(self.B_NT) - log(self.B_FT)

                log_g_z2 = -dim/2 * log(2*pi) - .5 * log(np.linalg.det(S)) - .5 * d2
                dLLR = log(self.P_d) - log(self.B_FT) + log_g_z2

                LLR_2 = float(LLR_1 + dLLR.squeeze())

                x, P = initialize_velocity(um, m)
                kf = linear_ncv(noise_intensity=1)
                kf.initialize(x, P)

                # TODO here LLR can be recomputed without P_vv
                t = TrackHypothesis(self.next_track_id(), kf, m.time, LLR_2)
                d = InitializeTrack(t, um, m, LLR_2)
                decisions.append(d)

                _execute_callbacks(callbacks, 'consider_initialize_track', self, d)

            # 1.3. leave measurement unassociated: false track or new track
            llr = log(self.P_d) + log(self.B_NT) - log(self.B_FT)
            d = MaybeNewTrack(m, llr)
            decisions.append(d)

            _execute_callbacks(callbacks, 'consider_maybe_new_track', self, d)

        # 2. a possibility that a track did not receive an update
        if self.P_d < 1:
            dLLR = log(1 - self.P_d)
        else:
            dLLR = 0
        
        for t in self.tracks:
            d = TrackNotUpdated(t, t.llr + dLLR)
            decisions.append(d)
            
            _execute_callbacks(callbacks, 'consider_track_not_updated', self, d)
        
        # TODO one more option: previous measurement is a false measurement, so we do
        #      not initialize a new track but keep the new measurement for the next
        #      iteration
        
        # sort with LLR descending
        decisions.sort(key=lambda x: x.LLR, reverse=True)

        _execute_callbacks(callbacks, 'mht_decisions', self, decisions)

        new_tracks = []
        new_track_ids = set()
        new_unassociated_measurements = []
        decided_measurement_ids = set()

        for d in decisions:
            if isinstance(d, UpdateTrack):
                if d.track.track_id in new_track_ids:
                    continue
                if d.measurement.measurement_id in decided_measurement_ids:
                    continue

                # TODO make sure that timestamps match

                d.track.kf.calculate_innvovation(d.measurement.z, d.measurement.R)
                d.track.kf.update()
                d.track.llr = d.LLR

                new_tracks.append(d.track)
                new_track_ids.add(d.track.track_id)
                decided_measurement_ids.add(d.measurement.measurement_id)

                _execute_callbacks(callbacks, 'update_track', self, d)

            # TODO maybe allow initializing new tracks from pairs of measurements with one or more
            #      measurements missing in between them? (gaps)
            elif isinstance(d, InitializeTrack):
                if d.measurement0.measurement_id in decided_measurement_ids:
                    continue
                if d.measurement1.measurement_id in decided_measurement_ids:
                    continue
                
                new_tracks.append(d.track)
                new_track_ids.add(d.track.track_id)
                decided_measurement_ids.add(d.measurement0.measurement_id)
                decided_measurement_ids.add(d.measurement1.measurement_id)

                _execute_callbacks(callbacks, 'initialize_track', self, d)

            elif isinstance(d, MaybeNewTrack):
                if d.measurement.measurement_id in decided_measurement_ids:
                    continue

                new_unassociated_measurements.append(d.measurement)
                decided_measurement_ids.add(d.measurement.measurement_id)

                _execute_callbacks(callbacks, 'maybe_new_track', self, d)

            elif isinstance(d, TrackNotUpdated):
                if d.track.track_id in new_track_ids:
                    continue

                d.track.llr = d.LLR
                new_tracks.append(d.track)
                new_track_ids.add(d.track.track_id)

                _execute_callbacks(callbacks, 'track_not_updated', self, d)
            
            else:
                raise Exception(f"Unknown decision '{d}'")

        # TODO report unassociated_measurements as ignored

        # 3. replace previous set of unassociated measurements: they either already produced
        #    a track or are considered false detections
        self.unassociated_measurements = new_unassociated_measurements
        self.tracks = new_tracks



    def estimate_tracks(self, t: float):
        return [Track(t.track_id, t, t.kf.x_hat, t.kf.P_hat) for t in self.tracks]



def _execute_callbacks(callbacks, stage: str, *args):
    if callbacks is not None:
        for cb in callbacks:
            execute_callback(cb, stage, *args)
