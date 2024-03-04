import numpy as np
import pandas as pd
import itertools as it
from collections import UserList
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from .kalman import kalman_pv, kalman_pva
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
        self.time = np.arange(0, n, T)

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
        kf = kalman_pva(task.motion_model, task.R)
    else:
        kf = kalman_pv(task.motion_model, task.R)
    
    # initialize track state
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


def state_residuals_np(results: Union[EvaluationResult,EvaluationResultList]) -> List[Tuple[Dict, np.ndarray]]:
    # each entry in the list is a list of results 
    if type(results) == EvaluationResult:
        groups = [[results]]
    else:
        # aggregate across seeds
        parameters = set(dict(results[0]).keys())
        parameters.discard('seed')
        groups = results.group(parameters)

    ans = []
    for group in groups:
        for r in group:
            dims = min(r.truth.shape[1], r.x_hat.shape[1])
            data = r.truth[:,:dims] - r.x_hat[:,:dims]
            ans.append((dict(r), data))
    return ans


def state_residuals(results: Union[EvaluationResult,EvaluationResultList]) -> pd.DataFrame:
    col_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']

    ans = []
    for params, data in state_residuals_np(results):
        data = to_df(data, columns=col_names[:data.shape[1]])
        data['time'] = np.arange(len(data))
        for key, value in dict(params).items():
            data[key] = value
        ans.append(data)

    # final transformations: combine data frames into one, melt, add dimensions category
    data = pd.concat(ans)

    id_vars = set(data.columns).difference(col_names)
    value_vars = set(data.columns).intersection(col_names)
    data = data.melt(id_vars=id_vars, value_vars=value_vars, var_name='state_dim', value_name='residual')
    
    data['category'] = data.state_dim.map(dict(
                            x='position', y='position', z='position',
                            vx='velocity', vy='velocity', vz='velocity',
                            ax='acceleration', ay='acceleration', az='acceleration'))
    data['dim'] = data.state_dim.map(dict(x='x', y='y', z='z', vx='x', vy='y', vz='z', ax='x', ay='y', az='z'))

    return data



def rmse(results: List[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for params, data in state_residuals_np(results):
        row = deepcopy(params)
        row['category'] = 'position'
        row['rmse'] = np.sqrt(np.power(data[:,:3], 2).sum(axis=1).mean(axis=0))
        rows.append(row)

        row = deepcopy(params)
        row['category'] = 'velocity'
        row['rmse'] = np.sqrt(np.power(data[:,3:], 2).sum(axis=1).mean(axis=0))
        rows.append(row)

    return pd.DataFrame(rows)
