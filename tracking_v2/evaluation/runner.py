import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
import inspect
from typing import Dict, List

from ..np import as_column



__all__ = ['before_one', 'after_one', 'before_many', 'after_many', 'after_update',
           'StateFilterRunner', 'evaluate_nees', 'evaluate_runner']



def before_one(method):
    method.runner_callback_tag = 'before_one'
    return method


def after_one(method):
    method.runner_callback_tag = 'after_one'
    return method


def before_many(method):
    method.runner_callback_tag = 'before_many'
    return method


def after_many(method):
    method.runner_callback_tag = 'after_many'
    return method


def after_initialize(method):
    method.runner_callback_tag = 'after_initialize'
    return method


def after_predict(method):
    method.runner_callback_tag = 'after_predict'
    return method


def after_update(method):
    method.runner_callback_tag = 'after_update'
    return method




class Runner:
    _callbacks: Dict = None


    def _execute_user_callbacks(self, stage, *args):
        if self._callbacks is None:
            self._callbacks = {}
        
        if stage not in self._callbacks:
            callbacks = []
            for name, member in inspect.getmembers(self, inspect.ismethod):
                if hasattr(member, 'runner_callback_tag') and member.runner_callback_tag == stage:
                    callbacks.append(member)
            self._callbacks[stage] = callbacks
        
        for callback in self._callbacks[stage]:
            callback(*args)



# ----------------------------------------------------------------------


class StateFilterRunner(Runner):
    def __init__(self, target, sensor, kf):
        self.target = target
        self.sensor = sensor
        self.kf = kf

        self.truth = None
        self.n = None
        self.m = None
        self.seeds = None

        self.one_truth, self.many_truth = [], []
        self.one_z, self.many_z = [], []
        
        # state and its covariance
        self.one_x_hat, self.one_P_hat = [], []
        self.many_x_hat, self.many_P_hat = [], []
        
        # innovation and its covariance
        self.one_v, self.one_S = [], []
        self.many_v, self.many_S = [], []

        self.dim = 3

    @after_initialize
    def __after_initialize(self, m):
        self.one_z.append(as_column(m.z))

    @after_predict
    def __after_predict(self):
        self.one_x_hat.append(np.copy(self.kf.x_hat))
        self.one_P_hat.append(np.copy(self.kf.P_hat))
    
    @after_update
    def __after_update(self, m):
        self.one_v.append(np.copy(self.kf.innovation))
        self.one_S.append(np.copy(self.kf.S))
        self.one_z.append(as_column(m.z))

    @before_one
    def __before_one(self):
        self.one_x_hat = []
        self.one_P_hat = []
        self.one_v = []
        self.one_S = []
        self.one_z = []

    @after_one
    def __after_one(self):
        self.one_x_hat = np.array(self.one_x_hat)
        self.one_P_hat = np.array(self.one_P_hat)

        assert self.one_x_hat.shape[0] == self.n, f"{self.one_x_hat.shape[0]} != {self.n}"
        assert self.one_P_hat.shape[0] == self.n, f"{self.one_P_hat.shape[0]} != {self.n}"

        self.many_x_hat.append(self.one_x_hat)
        self.many_P_hat.append(self.one_P_hat)

        self.one_v = np.array(self.one_v)
        self.one_S = np.array(self.one_S)

        assert self.one_v.shape[0] == self.n, f"{self.one_v.shape[0]} != {self.n}"
        assert self.one_S.shape[0] == self.n, f"{self.one_S.shape[0]} != {self.n}"

        self.many_v.append(self.one_v)
        self.many_S.append(self.one_S)

        self.one_truth = np.copy(self.truth)
        self.many_truth.append(self.one_truth)

        self.many_z.append(np.asarray(self.one_z))

    @before_many
    def __before_many(self):
        self.many_x_hat = []
        self.many_P_hat = []
        self.many_truth = []
        self.many_v = []
        self.many_S = []
        self.many_z = []

    @after_many
    def __after_many(self):
        self.many_x_hat = np.asarray(self.many_x_hat)
        self.many_P_hat = np.asarray(self.many_P_hat)
        self.many_truth = np.asarray(self.many_truth)
        self.many_v = np.asarray(self.many_v)
        self.many_S = np.asarray(self.many_S)
        self.many_z = np.asarray(self.many_z)

        assert self.many_x_hat.shape[0] == self.m
        assert self.many_P_hat.shape[0] == self.m
        assert self.many_truth.shape[0] == self.m
        assert self.many_v.shape[0] == self.m
        assert self.many_S.shape[0] == self.m
        assert self.many_z.shape[0] == self.m


    def run_one(self, n: int, T: float = 1):
        t = 0
        self.n = n
        
        self._execute_user_callbacks('before_one')
        self.target.cache(T, n+1)
        self.truth = self.target.cached_states
        
        m = self.sensor.generate_measurement(t, self.target)

        self.kf.reset()
        self.kf.initialize(m.z, m.R)

        self._execute_user_callbacks('after_initialize', m)

        for t in np.arange(1, n+1) * T:
            self.kf.predict(T)

            self._execute_user_callbacks('after_predict')

            m = self.sensor.generate_measurement(t, self.target)
            self.kf.prepare_update(m.z, m.R)
            self.kf.update()

            self._execute_user_callbacks('after_update', m)

        self._execute_user_callbacks('after_one')



    def run_many(self, m: int, n: int, T: float = 1, seeds=None):
        if seeds is None:
            seeds = np.arange(m)
        assert m == len(seeds)
        
        self.n = n
        self.m = m
        self.seeds = seeds

        self._execute_user_callbacks('before_many')
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            self.sensor.reset_rng(rng)
            self.target.reset_rng(rng)

            self.run_one(n, T)
        
        self._execute_user_callbacks('after_many')











class NScoreEvaluationResult:
    def __init__(self, scores: ArrayLike, dim: int, type: str):
        self.scores = scores
        self.dim = dim
        self.type = type


def _resolve_truth_shape(truth, x_hat):
    assert len(truth.shape) in [2, 3, 4]
        
    if len(truth.shape) == 2:
        assert truth.shape[0] == x_hat.shape[1]
        assert truth.shape[1] == x_hat.shape[2]
        truth = truth.reshape((1, truth.shape[0], truth.shape[1], 1))

    elif len(truth.shape) == 3:
        if truth.shape[-1] == 1:
            assert truth.shape[0] == x_hat.shape[1]
            assert truth.shape[1] == x_hat.shape[2]
            assert truth.shape[2] == x_hat.shape[3]
            truth = truth.reshape((1, truth.shape[0], truth.shape[1], truth.shape[2]))

        else:
            assert truth.shape[0] == x_hat.shape[0] or truth.shape[0] == 1 
            assert truth.shape[1] == x_hat.shape[1]
            assert truth.shape[2] == x_hat.shape[2]
            truth = truth.reshape((truth.shape[0], truth.shape[1], truth.shape[2], 1))

    return truth

def evaluate_nees(x_hat, P_hat, truth):
    # if data comes from single run, add a dimension in the front
    if len(x_hat.shape) == 3 and len(P_hat.shape) == 3:
        x_hat = np.expand_dims(x_hat, 0)
        P_hat = np.expand_dims(P_hat, 0)

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(x_hat.shape) == 4
    assert len(P_hat.shape) == 4

    truth = _resolve_truth_shape(truth, x_hat)
    dim   = truth.shape[-2]
    
    diff  = x_hat - truth
    P_inv = np.linalg.inv(P_hat)
    
    scores = np.matmul(np.matmul(diff.transpose(0, 1, 3, 2), P_inv), diff).squeeze()
    scores = np.atleast_2d(scores)
    
    return NScoreEvaluationResult(scores, dim, 'NEES')


def evaluate_error(x_hat: np.ndarray, truth: np.ndarray) -> np.ndarray:
    # if data comes from single run, add a dimension in the front
    if len(x_hat.shape) == 3:
        x_hat = np.expand_dims(x_hat, 0)

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(x_hat.shape) == 4

    truth = _resolve_truth_shape(truth, x_hat)
    
    diff = x_hat - truth
    err_sq = np.matmul(diff.transpose(0, 1, 3, 2), diff)
    err_sq = np.atleast_2d(err_sq.squeeze())
    return np.sqrt(err_sq)



def evaluate_nis(v: ArrayLike, S: ArrayLike) -> NScoreEvaluationResult:    
    # if data comes from single run, add a dimension in the front
    if len(v.shape) == 3 and len(S.shape) == 3:
        v = np.expand_dims(v, 0)
        S = np.expand_dims(S, 0)
    
    dim = v.shape[-2]

    # the rest of the code expects 4-dimensional data:
    # MC-runs x run-length x spatial-dimensions x (1 | spatial-dimensions)
    assert len(v.shape) == 4
    assert len(S.shape) == 4
    
    S_inv = np.linalg.inv(S)
    
    scores = np.matmul(np.matmul(v.transpose(0, 1, 3, 2), S_inv), v).squeeze()
    scores = np.atleast_2d(scores)
    return NScoreEvaluationResult(scores, dim, 'NIS')



class StateFilterEvaluationResult:
    def __init__(self, position_nees: NScoreEvaluationResult, velocity_nees: NScoreEvaluationResult,
                 position_nis: NScoreEvaluationResult, velocity_nis: NScoreEvaluationResult,
                 position_error: np.ndarray):
        self.position_nees = position_nees
        self.velocity_nees = velocity_nees
        self.position_nis = position_nis
        self.velocity_nis = velocity_nis
        self.position_error = position_error



def evaluate_runner(runner: StateFilterRunner):
    x_hat, P_hat, truth, v, S = runner.many_x_hat[:, :, :6, :], runner.many_P_hat[:, :, :6, :6], \
                                runner.many_truth[:, 1:, :6], runner.many_v[:, :, :6, :], \
                                runner.many_S[:, :, :6, :6]

    assert len(x_hat.shape) == 4
    assert len(P_hat.shape) == 4
    assert len(truth.shape) in [2, 3]
    assert len(v.shape) == 4
    assert len(S.shape) == 4
    
    assert x_hat.shape[0] == P_hat.shape[0] # number of independent runs
    assert x_hat.shape[1] == P_hat.shape[1] # length of a single run
    assert x_hat.shape[2] == P_hat.shape[2] # number of state dimensions
    assert P_hat.shape[2] == P_hat.shape[3] # P_hat is a square matrix
    assert x_hat.shape[3] == 1              # x_hat is a column vector
    
    assert v.shape[3] == 1
    assert S.shape[3] == v.shape[2]
    assert S.shape[2] == S.shape[3]
    
    if len(truth.shape) == 2:
        truth = np.expand_dims(truth, 0)

    return StateFilterEvaluationResult(
        evaluate_nees(x_hat[:,:,:3,:], P_hat[:,:,:3,:3], truth[:, :,:3]),
        evaluate_nees(x_hat[:,:,3:,:], P_hat[:,:,3:,3:], truth[:, :,3:]),
        evaluate_nis(v[:,:,:3,:], S[:,:,:3,:3]),
        evaluate_nis(v[:,:,3:,:], S[:,:,3:,3:]),
        evaluate_error(x_hat[:,:,:3,:], truth[:,:,:3])
    )


def nees_ci(runner: StateFilterRunner, qs: List[float] = [.025, .975]):
    return sp.stats.chi2.ppf(qs, runner.m * runner.dim) / runner.m
