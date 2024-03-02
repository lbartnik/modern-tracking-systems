import numpy as np
import pandas as pd
import itertools as it
from collections import UserList
from copy import deepcopy
from typing import List

from .kalman import KalmanFilter6D, KalmanFilter9D
from .util import to_df


class EvaluationTask:
    def __init__(self, target_model, motion_model, T: float, n: int, z_sigma: float, R: float, seed: int):
        self.target_model = target_model
        self.motion_model = motion_model
        self.T = T
        self.n = n
        self.z_sigma = z_sigma
        self.R = R
        self.seed = seed

    def __iter__(self):
        yield 'target', self.target_model.name
        yield 'motion', self.motion_model.name
        yield 'z_sigma', self.z_sigma
        yield 'seed', self.seed
        yield 'R', self.R
    
    @property
    def motion(self):
        return self.motion_model.name
    
    @property
    def target(self):
        return self.target_model.name
    
    def __repr__(self) -> str:
        return '{} {}'.format(self.__class__.__name__, dict(self))
    

class EvaluationResult(EvaluationTask):
    def __init__(self, task: EvaluationTask, truth: np.ndarray, x_hat: np.ndarray, P_hat: np.ndarray, K: np.ndarray, z: np.ndarray):
        super().__init__(target_model=task.target_model, motion_model=task.motion_model, T=task.T,
                         n=task.n, z_sigma=task.z_sigma, R=task.R, seed=task.seed)
        self.truth = truth
        self.x_hat = x_hat
        self.P_hat = P_hat
        self.K = K
        self.z = z
    
    def __repr__(self) -> str:
        return '{} {}'.format(self.__class__.__name__, dict(self))

    @property
    def truth_df(self) -> pd.DataFrame:
        columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
        return to_df(self.truth, columns=columns[:self.truth.shape[1]])

    @property
    def x_hat_df(self) -> pd.DataFrame:
        columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
        return to_df(self.x_hat, columns=columns[:self.x_hat.shape[1]])
    
    def copy(self, **kwargs):
        x = deepcopy(self)
        for name, value in kwargs.items():
            setattr(x, name, value)
        return x


class EvaluationResultList(UserList):
    def select(self, **kwargs) -> List[EvaluationResult]:
        """Select results which match the provided set of parameter values.

        Returns:
            List[EvaluationResult]: Evaluation results matching the given query.
        """
        ans = []
        for r in self.data:
            match = [str(getattr(r, name)) == str(value) for name, value in kwargs.items()]
            if all(match):
                ans.append(r)
        return EvaluationResultList(ans)
    
    def group(self, keys: List[str]) -> List[List[EvaluationResult]]:
        """Group results by given set of evaluation task parameters.

        Args:
            keys (List[str]): A list of parameter names.

        Returns:
            List[EvaluationResult]: List of groups of evaluation results.
        """
        if isinstance(keys, str):
            keys = [keys]          
        
        groups = {}
        for result in self.data:
            group_id = '_'.join([str(getattr(result, key)) for key in keys])
            groups.setdefault(group_id, []).append(result)
        
        return [EvaluationResultList(group) for group in groups.values()]


def execute(task: EvaluationTask) -> EvaluationResult:
    # calculate target positions over time
    true_states = task.target_model.true_states(T=task.T, n=task.n, seed=task.seed)

    # transform positions into measurements
    z_var = task.z_sigma**2
    meas = _cartesian_measurements(true_states[:,:3], np.diag([z_var, z_var, z_var]))

    # pick Kalman Filter of appropriate dimensionality
    if task.motion_model.state_dim == 9:
        KF = KalmanFilter9D
    else:
        KF = KalmanFilter6D
    
    # initialize track state
    kf = KF(R=task.R, motion_model=task.motion_model)
    kf.initialize(meas[0,:], np.eye(meas.shape[1]) * z_var)

    # iterate and collect state estimates
    x_hat, P_hat, K = [], [], []
    for z in meas:
        kf.predict(task.T)
        x_hat.append(kf.x_hat)
        P_hat.append(kf.P_hat)
        
        kf.update(z)
        K.append(kf.K)
    
    return EvaluationResult(task, true_states, np.array(x_hat), np.array(P_hat), np.array(K), meas)


def _cartesian_measurements(positions, noise_covariance):
    noise_mean = np.full(positions.shape[1], 0)
    noise = np.random.multivariate_normal(noise_mean, noise_covariance, size=positions.shape[0])
    return positions + noise


def _as_iterable(x):
    if hasattr(x, '__iter__'):
        return x
    return [x]


def monte_carlo(target, motion_model, z_sigma=.1, seeds=range(50), T: float = 1, n: int = 400):
    # make sure each dimension is iterable
    target = _as_iterable(target)
    motion_model = _as_iterable(motion_model)
    z_sigma = _as_iterable(z_sigma)
    seeds = _as_iterable(seeds)

    results = []

    # iterate over a Cartesian product of the following dimensions
    for tg, mm, zs, sd in it.product(target, motion_model, z_sigma, seeds):
        task = EvaluationTask(target_model=tg, motion_model=mm, T=T, n=n, z_sigma=zs, R=zs*zs, seed=sd)
        results.append(execute(task))
    
    return EvaluationResultList(results)


def rmse(results: List[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        spatial_dim = r.target_model.spatial_dim
        err = r.truth[:, :spatial_dim] - r.x_hat[:, :spatial_dim]

        row = dict(r)
        row['rmse'] = np.sqrt(np.power(err, 2).sum(axis=1).mean(axis=0))
        rows.append(row)
    
    return pd.DataFrame(rows)


def chi_squared(results: List[EvaluationResult]) -> pd.DataFrame:
    ans = []
    for r in results:
        diff = r.x_hat[:,:3] - r.truth[:,:3]
        diff = np.expand_dims(diff, 1)

        P = r.P_hat[:,:3,:3]

        chi_sq = np.matmul(diff, np.linalg.inv(P))
        chi_sq = np.matmul(chi_sq, np.transpose(diff, (0, 2, 1)))

        ans.append(chi_sq.squeeze())
    return ans



# ---

def evaluate(target, kf, time=np.arange(0, 100), z_sigma=.1, seed=0):
    # calculate target positions over time
    time = np.array(time)
    positions = target.positions(time, seed=0)
    
    # transform positions into measurements
    z_var = z_sigma*z_sigma
    meas = _cartesian_measurements(positions, np.diag([z_var, z_var, z_var]))
    
    # initialize track state
    mean = np.full(6, 0)
    mean[:3] = meas[0,:]
    cov = np.diag(np.full(6, z_var))
    
    kf.initialize(mean, cov)

    # iterate
    err = []
    pos = []
    vel = []
    
    for dt, z, true_pos in zip(np.diff(time), meas, positions):
        kf.predict(dt)
        pos.append(kf.x_hat[:3])
        vel.append(kf.x_hat[3:6])
        err.append(kf.x_hat[:3] - true_pos)
        kf.update(z)
    
    return positions, np.array(pos), np.array(vel), np.array(err)
