import numpy as np
import scipy as sp
from math import log, pi
from typing import List

from ..np import as_column 
from .interface import Tracker, Track
from ..kalman import linear_ncv, KalmanFilter
from ..sensor import SensorMeasurement
from ..callback import execute_callback


__all__ = ['SingleTargetTracker', 'MultiTargetTracker']


class SingleTargetTracker(Tracker):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.time = None
        self.first_meas = None
        self.kf = None
        self.llr = None

        # LLR parameters
        self.P_d = 1  # probability of detection
        self.B_NT = 1 # probability of new target
        self.B_FT = 1 # probability of false target
    
    def add_measurements(self, ms):
        assert len(ms) == 1, "more than 1 measurement not supported"
        dim = len(ms[0].z.squeeze())
        
        assert dim == 3, "only 3D spaces are supported"
        m = ms[0]

        if self.time is None:
            self.time = m.time

        assert m.time >= self.time, "measurement in the past"

        if self.first_meas is None:
            self.first_meas = m
        
        elif self.track is None:
            x, P = _initialize_velocity(self.first_meas, m)
            self.kf = linear_ncv(noise_intensity=1)
            self.kf.initialize(x, P)
            
            # initial LLR: new target appears
            self.llr = log(self.P_d * self.B_NT / self.B_FT)

        else:
            self.kf.predict(m.time - self.time)
            self.kf.calculate_innvovation(m.z, m.R)
            self.kf.update()

            # hit: measurement associated
            self.llr += log(self.P_d / self.B_FT) - dim/2*log(2*pi) - .5 * np.linalg.det(self.kf.S) \
                        - .5 * self.kf.innovation.T @ self.kf.S @ self.kf.innovation
    
            # not handling LLR for miss (measurement not associated)

    def estimate_tracks(self, t: float):
        if self.kf is None:
            return []
    
        assert t == self.time, "cannot estimate tracks for arbitrary t"
        return [Track(0, self.kf.x_hat, self.kf.P_hat)]



def _initialize_velocity(m0, m1):
    """Initialize position+velocity mean and covariance from two measurements.
    """
    dt = m1.time - m0.time
    dp = m1.z - m0.z
    
    vel = dp / dt
    P_vel = (m0.R + m1.R) / (dt * dt)
    P_pos_vel = m1.R / dt
    
    x = np.concatenate((m1.z.squeeze(), vel.squeeze()))
    P = np.zeros((6, 6))
    P[:3, :3] = m1.R
    P[3:, 3:] = P_vel
    P[:3, 3:] = P_pos_vel
    P[3:, :3] = P_pos_vel

    return x, P




class Track:
    def __init__(self, track_id: int, kf: KalmanFilter, time: float, llr: float):
        self.track_id = track_id
        self.kf = kf
        self.time = time
        self.llr = llr



class MultiTargetTracker(Tracker):
    def __init__(self):
        self.reset()
        self.callback = None
    
    def reset(self):
        self.track_id_generator = 0

        self.tracks: List[Track] = []
        self.unassociated_measurements = []

        # general LLR parameters
        self.P_d = 1  # probability of detection
        self.B_NT = 1 # probability of new target
        self.B_FT = 1 # probability of false target

        # two-point initialization LLR parameters
        self.v_max = 100
    
    def next_track_id(self):
        self.track_id_generator += 1
        return self.track_id_generator
    
    def add_measurements(self, ms: List[SensorMeasurement]):
        if len(ms) == 0:
            return
        
        dim = len(ms[0].z.squeeze())
        assert dim == 3, "only 3D spaces are supported"

        # possible decisions:
        #   - track update with a new measurement
        #   - new track from a previously unassociated measurement and a new measurement
        #   - new measurement is left unassociated
        #   - an existing track does not have an update in this iteration
        decisions = []

        # 1. for each new measurement
        for m in ms:
            # 1.1. score measurement against track
            for t in self.tracks:
                # TODO consider only if passes a gate
                t.kf.calculate_innvovation(m.z, m.R)
                
                llr = log(self.P_d / self.B_FT) - dim/2*log(2*pi) - .5 * np.linalg.det(self.kf.S) \
                        - .5 * self.kf.innovation.T @ self.kf.S @ self.kf.innovation
                
                decisions.append(('associate_to_track', t.llr + llr, t, m))

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

                d2 = v.T @ np.linalg.inv(S) @ v
                G = float(sp.stats.chi2.ppf(.95, dim))

                # Chi-squared gate
                if d2 > G:
                    continue

                # accepted; start by calculating the initial LLR due to initializing a track with um
                LLR_1 = log(self.P_d) + log(self.B_NT) - log(self.B_FT)

                log_g_z2 = -dim/2 * log(2*pi) - .5 * log(np.linalg.det(S)) - .5 * d2
                dLLR = log(self.P_d) - log(self.B_FT) + log_g_z2

                LLR_2 = LLR_1 + dLLR

                x, P = _initialize_velocity(um, m)
                kf = linear_ncv(noise_intensity=1)
                kf.initialize(x, P)

                # TODO here LLR can be recomputed without P_vv
                t = Track(self.next_track_id(), kf, m.time, LLR_2)
                decisions.append(('initialize_new_track', LLR_2, t, um, m))


            # 1.3. leave measurement unassociated: false track or new track
            llr = log(self.P_d) + log(self.B_NT) - log(self.B_FT)
            decisions.append(('measurement_not_associated', llr, m))

        # 2. a possibility that a track did not receive an update
        if self.P_d < 1:
            dLLR = log(1 - self.P_d)
        else:
            dLLR = 0
        
        for t in self.tracks:
            decisions.append(('track_not_updated', t.llr + dLLR, t))
        
        # sort with LLR descending
        decisions.sort(key=lambda x: x[1], reverse=True)

        new_tracks = []
        new_track_ids = set()
        new_unassociated_measurements = []
        decided_measurement_ids = set()

        for d in decisions:
            if d[0] == 'associate_to_track':
                _, llr, t, m = d

                if t.track_id in new_track_ids:
                    continue
                if m.measurement_id in decided_measurement_ids:
                    continue

                t.kf.calculate_innvovation(m.z, m.R)
                t.kf.update()
                new_tracks.append(t)
                new_track_ids.add(t.track_id)
                decided_measurement_ids.add(m.measurement_id)

                if self.callback is not None:
                    execute_callback(self.callback, 'associated_to_track', llr, t, m)

            # TODO maybe allow initializing new tracks from pairs of measurements with one or more
            #      measurements missing in between them? (gaps)
            elif d[0] == 'initialize_new_track':
                _, llr, t, m0, m1 = d

                if m.measurement_id in decided_measurement_ids:
                    continue

                new_tracks.append(t)
                new_track_ids.add(t.track_id)
                decided_measurement_ids.add(m.measurement_id)

                if self.callback is not None:
                    execute_callback(self.callback, 'initialized_new_track', llr, t, m0, m1)

            elif d[0] == 'measurement_not_associated':
                _, llr, m = d

                if m.measurement_id in decided_measurement_ids:
                    continue

                new_unassociated_measurements.append(m)
                decided_measurement_ids.add(m.measurement_id)

                if self.callback is not None:
                    execute_callback(self.callback, 'measurement_not_associated', llr, m)

            elif d[0] == 'track_not_updated':
                _, llr, t = d

                if t.track_id in new_track_ids:
                    continue

                t.llr = llr
                new_tracks.append(t)
                new_track_ids.add(t.track_id)

                if self.callback is not None:
                    execute_callback(self.callback, 'track_not_updated', llr, t)


        # 3. replace previous set of unassociated measurements: they either already produced
        #    a track or are considered false detections
        self.unassociated_measurements = new_unassociated_measurements



    def estimate_tracks(self, t: float):
        return [Track(t.track_id, t.kf.x_hat, t.kf.P_hat) for t in self.tracks]

