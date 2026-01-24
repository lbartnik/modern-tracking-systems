import numpy as np
import scipy as sp
from collections import defaultdict
from math import log, pi
from typing import Callable, Dict, List, Tuple

import tracking_v2.callback as cb

from ..np import as_column 
from .base import Tracker, Track, initialize_velocity
from ..kalman import linear_ncv, KalmanFilter
from ..sensor import SensorMeasurement
from ..target import Target
from ..callback import execute_callback
from ..util import to_df


__all__ = ['MultiTargetTracker', 'MhtDecisionClassifier', 'MhtEstimationMetrics',
           'InitializeTrack', 'MaybeNewTrack', 'UpdateTrack', 'TrackNotUpdated']


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
    def __init__(self, track: TrackHypothesis, measurement: SensorMeasurement, dLLR: float, NIS: float, S: np.ndarray, S_det: float):
        self.track = track
        self.measurement = measurement
        self.dLLR = float(dLLR)
        self.LLR = float(track.llr + dLLR)
        self.NIS = float(NIS)
        self.S = S
        self.S_det = S_det
    
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
                
                S_det = np.linalg.det(t.kf.S)
                dLLR = log(self.P_d / self.B_FT) - dim/2*log(2*pi) - .5 * log(S_det) - .5 * d2
                dLLR = float(dLLR)
                
                d = UpdateTrack(t, m, dLLR, NIS=d2, S=t.kf.S, S_det=S_det)
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

        _execute_callbacks(callbacks, 'considered_decisions', self, decisions)

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




class MhtDecisionClassifier:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.decisions_considered_in_this_iteration = None
        self.decided_in_this_iteration = {}
        self.time: float = None
        self.track_to_target = {}
        self.reset_one_counters()
        self.reset_many_counters()
    
    @cb.before_one
    def reset_one_counters(self):
        self.inconsistent_init_one = 0
        self.already_taken_one = 0
        self.better_score_one = 0
        self.track_to_target = {}

        self.debug_trace = {}
    
    @cb.before_many
    def reset_many_counters(self):
        self.inconsistent_init_many = []
        self.already_taken_many = []
        self.better_score_many = []
        self.track_per_target_many = []
        self.misassignment_in_time = defaultdict(float)

    def __repr__(self):
        return f"MhtDecisionClassifier(inconsistent_init: {self.inconsistent_init_one}, better_score: {self.better_score_one}, already_taken: {self.already_taken_one})"

    @cb.after_one
    def record_trace(self):
        run = len(self.already_taken_many)
        self.inconsistent_init_many.append((run, self.inconsistent_init_one))
        self.already_taken_many.append((run, self.already_taken_one))
        self.better_score_many.append((run, self.better_score_one))

        tracks_per_target = defaultdict(int)
        for _, counts in self.track_to_target.items():
            for target_id, _ in counts.items():
                tracks_per_target[target_id] += 1
        
        for target_id, count in tracks_per_target.items():
            self.track_per_target_many.append((run, target_id, count))
    
    @cb.after_many
    def to_numpy(self):
        self.inconsistent_init_many = np.asarray(self.inconsistent_init_many)
        self.already_taken_many = np.asarray(self.already_taken_many)
        self.better_score_many = np.asarray(self.better_score_many)
        self.track_per_target_many = np.asarray(self.track_per_target_many)
        
        self.misassignment_in_time = np.asarray((
            np.asarray(self.misassignment_in_time.keys()),
            np.asarray(self.misassignment_in_time.values())
        )).T

    @cb.measurement_frame
    def set_time(self, time: float, measurements: List):
        self.time = float(time)

    @cb.considered_decisions
    def set_considered_decisions(self, tracker, considered_decisions: List):
        assert len(considered_decisions) > 0
        self.decisions_considered_in_this_iteration = considered_decisions
        self.decided_in_this_iteration = {}

    # TODO account for multiple tracks for the same target in the same run: track
    #      drops and tracks maintained in parallel (will be possible once track
    #      expiration is implemented)

    def _append_debug(self, track_id: int, target_id: int, measurement: SensorMeasurement, reason: str = ''):
        track_trace: List = self.debug_trace.setdefault(track_id, [])
        track_trace.append((self.time, target_id, measurement.target_id, measurement.measurement_id,
                            reason, self.already_taken_one, self.better_score_one))

    @cb.after_one
    def _debug_to_pandas(self):
        self.debug_trace = {track_id: to_df(data, columns=['time', 'track_target', 'meas_target', 'meas_id',
                                                           'reason', 'already_taken', 'better_score'])
                            for track_id, data in self.debug_trace.items()}


    @cb.initialize_track
    def initialize_track(self, tracker, decision: InitializeTrack):
        target_counts: Dict = self.track_to_target.setdefault(decision.track.track_id, {})
        
        if decision.measurement0.target_id == decision.measurement1.target_id:
            target_counts[decision.measurement0.target_id] = 2
        else:
            self.inconsistent_init_one += 1
            target_counts[decision.measurement0.target_id] = 1
            target_counts[decision.measurement1.target_id] = 1

        if self.debug:
            self._append_debug(decision.track.track_id, decision.measurement0.target_id, decision.measurement0)
            self._append_debug(decision.track.track_id, decision.measurement0.target_id, decision.measurement1)


    @cb.update_track
    def update_track(self, tracker, decision: UpdateTrack):
        target_counts: Dict = self.track_to_target.setdefault(decision.track.track_id, {})
        if len(target_counts) > 0:
            track_target_id = max(target_counts, key=target_counts.get)
        else:
            track_target_id = None

        #print(decision.track.track_id, track_target_id, target_counts)

        # update counts after findings the most frequent target so far
        target_counts[decision.measurement.target_id] = target_counts.get(decision.measurement.target_id, 0) + 1

        # correct association: measurement target matches the track target
        if track_target_id is None or decision.measurement.target_id == track_target_id:
            assert decision.measurement.target_id not in self.decided_in_this_iteration
            self.decided_in_this_iteration[decision.measurement.target_id] = decision
            
            if self.debug:
                self._append_debug(decision.track.track_id, track_target_id, decision.measurement)
            
            return
    
        # misassociation: check if the target tracked by this track has been already
        # assigned elsewhere
        self.misassignment_in_time[self.time] += 1

        target_assigned_to: UpdateTrack = self.decided_in_this_iteration.get(track_target_id)

        # measurement for this target was already assigned to a different track, so this track
        # picks up one of the still unassigned measurements, generated for a different target
        if target_assigned_to is not None:
            assert decision.measurement.target_id not in self.decided_in_this_iteration
            self.decided_in_this_iteration[decision.measurement.target_id] = decision

            if self.debug:
                d = decision
                m = decision.measurement
                t = decision.track
                mm = target_assigned_to.measurement
                print(f"{self.time}: misassociation (already taken): measurement {m.measurement_id} ({m.target_id}) to track " +
                    f"{t.track_id} ({t.target_id}); score {d.LLR:.2f} {d.NIS:.2f}; matching measurement " +
                    f"{mm.measurement_id} ({mm.target_id}) already assigned to track {target_assigned_to.track.track_id} " +
                    f"({target_assigned_to.track.target_id})")
            
            # TODO excluding measurements already taken, we can still run the same set of checks
            #      and assertions as run below for the other branch of the target_assigned_to condition

            self.already_taken_one += 1
            
            if self.debug:
                self._append_debug(decision.track.track_id, track_target_id, decision.measurement, 'already taken')
            
            return

        # TODO on top of classifying mistakes, these checks look for anomalizes in decisions
        #      taken by MHT; pull into a separate method "check for anomalies"; anomaly is a
        #      mistake that doesn't make sense given what I know about the MHT algorithm: either
        #      a coding error or an edge case that could be handled better, with more tailored
        #      logic

        # measurement for this track not yet assigned; check all potential associations for
        # this track
        def updates_for_this_track(x):
            return isinstance(x, UpdateTrack) and x.track.track_id == decision.track.track_id
        
        decisions_for_track: List[UpdateTrack] = list(filter(updates_for_this_track, self.decisions_considered_in_this_iteration))
        assert len(decisions_for_track) > 0

        # if the wrong association does not have the maximum incremental LLR, it is inconsistent
        # with the MHT algorithm design
        decisions_for_track.sort(key=lambda x: x.dLLR, reverse=True)
        assert decisions_for_track[0].measurement.measurement_id == decision.measurement.measurement_id

        matching: List[UpdateTrack] = list(filter(lambda x: x.measurement.target_id == track_target_id, decisions_for_track))
        assert len(matching) == 1

        if self.debug:
            d = decision
            m = decision.measurement
            t = decision.track
            mm = matching[0].measurement
            print(f"{self.time}: misassociation (better score): measurement {m.measurement_id} ({m.target_id}) to track " +
                f"{t.track_id} ({t.target_id}); score {d.LLR:.2f} {d.NIS:.2f}; matching measurement {mm.measurement_id} " +
                f"({mm.target_id}) has lower score {matching[0].LLR:.2f} {matching[0].NIS:.2f}")
        
        self.better_score_one += 1

        if self.debug:
            self._append_debug(decision.track.track_id, track_target_id, decision.measurement, 'better score')

        # if by dLLR this measurement was best, it had to have the smallest NIS, too
        decisions_for_track.sort(key=lambda x: x.NIS)
        assert decisions_for_track[0].measurement.measurement_id == decision.measurement.measurement_id
        
        # same about absolute position error: if by dLLR this decision is best, it must be best by error, too
        decisions_for_track.sort(key=lambda x: np.linalg.norm(x.track.kf.x_hat.squeeze()[:3] - x.measurement.z.squeeze()[:3]))
        assert decisions_for_track[0].measurement.measurement_id == decision.measurement.measurement_id

        assert decision.measurement.target_id not in self.decided_in_this_iteration
        self.decided_in_this_iteration[decision.measurement.target_id] = decision

        # TODO collect bad decisions across all iterations, tally by type, present tallies as metrics
        
        # TODO then use those metrics to evaluate a switch from greedy assignment to Hungarian
    
    def to_df(self):
        init = self.inconsistent_init_many[:, 1]
        at   = self.already_taken_many[:, 1]
        bs   = self.better_score_many[:, 1]
        tpt  = self.track_per_target_many[:, 2]


        metrics = (init.mean(), np.std(init), np.max(init), at.mean(), np.std(at), np.max(at),
                   bs.mean(), np.std(bs), np.max(bs), tpt.mean(), np.std(tpt), np.max(tpt))
        return to_df([metrics], columns=['init', 'init.std', 'init.max', 'at', 'at.std', 'at.max',
                                         'bs', 'bs.std', 'bs.max', 'tpt', 'tpt.std', 'tpt.max'])


class MhtEstimationMetrics:
    def __init__(self):
        # track ID -> target ID -> count
        self.track_to_target: Dict[int, Dict[int, int]] = {}
        # track ID -> list of (time, x_hat, P_hat)
        self.track_trace: Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]] = {}
        self.targets: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.time: float = None
        self.nees = {}

    @cb.before_many
    def before_many(self):
        self.nees = {}

    @cb.before_one
    def before_one(self):
        self.track_to_target = {}
        self.track_trace = {}
        self.targets = {}
        self.time = None

    @cb.target_cached
    def cache_trajectory(self, target: Target):
        self.targets.setdefault(target.target_id, (target.cached_time, target.cached_states))

    @cb.measurement_frame
    def measurement_frame(self, time: float, measurements: List[SensorMeasurement]):
        self.time = time

    @cb.initialize_track
    def initialize_track(self, tracker, decision: InitializeTrack):
        target_counters = self.track_to_target.setdefault(decision.track.track_id, {})

        if decision.measurement0.target_id == decision.measurement1.target_id:
            target_counters.setdefault(decision.measurement0.target_id, 2)
        else:
            target_counters.setdefault(decision.measurement0.target_id, 1)
            target_counters.setdefault(decision.measurement0.target_id, 1)
        
        track_trace = self.track_trace.setdefault(decision.track.track_id, [])
        track_trace.append((self.time, np.copy(decision.track.kf.x_hat), np.copy(decision.track.kf.P_hat)))
            

    @cb.update_track
    def update_track(self, tracker, decision: UpdateTrack):
        target_counters = self.track_to_target.setdefault(decision.track.track_id, {})
        if decision.measurement.target_id in target_counters:
            target_counters[decision.measurement.target_id] += 1
        else:
            target_counters[decision.measurement.target_id] = 1
        
        track_trace = self.track_trace.setdefault(decision.track.track_id, [])
        track_trace.append((self.time, np.copy(decision.track.kf.x_hat), np.copy(decision.track.kf.P_hat)))

    # since NEES requires truth data, we pick the target with the highest count
    # of associated measurements
    @cb.after_one
    def calculate_nees(self):
        #print(self.track_to_target)
        all_nees = {}

        for track_id, trace in self.track_trace.items():
            # pick target with highest count
            target_counts = self.track_to_target[track_id]
            target_id = max(target_counts, key=target_counts.get)
            
            true_tm, true_x = self.targets[target_id]

            tm = np.asarray([x[0] for x in trace])
            x_hat = np.asarray([x[1] for x in trace])
            P_hat = np.asarray([x[2] for x in trace])

            # extract target positions matching recorded track updates            
            i = np.isin(true_tm, tm)
            true_pos = true_x[i, :3]
            assert len(true_pos) == len(x_hat), f"{len(true_pos)} == {len(x_hat)}"

            x_hat, P_hat = np.asarray(x_hat), np.asarray(P_hat)
            P_inv = np.linalg.inv(P_hat[:, :3, :3])

            diff = x_hat[:, :3] - np.expand_dims(true_pos, axis=-1)
            nees = np.matmul(np.matmul(diff.transpose(0, 2, 1), P_inv), diff)
            
            all_nees.setdefault(target_id, []).append((tm, nees.squeeze()))
        
        # accumulate over many runs
        for target_id, nees in all_nees.items():
            self.nees.setdefault(target_id, []).extend(nees)
    
    @cb.after_many
    def after_many(self):
        # construct a time array that covers the full timeline across all targets and tracks
        all_tm = []
        for _, all_recorded_nees in self.nees.items():
            for tm, _ in all_recorded_nees:
                all_tm.append(tm.squeeze())

        all_tm = np.sort(np.unique(np.concatenate(all_tm)))


        # collect all tracks into arrays alinged with all_tm
        nees_per_target = {}

        for target_id, all_recorded_nees in self.nees.items():
            
            aligned_nees = np.full((len(all_tm), len(all_recorded_nees)), np.nan)
            for col, (tm, individual_nees) in enumerate(all_recorded_nees):
                row = np.isin(all_tm, tm)
                aligned_nees[row, col] = individual_nees
            
            nees_per_target[target_id] = aligned_nees
        
        self.nees = nees_per_target

                
