import numpy as np
import scipy as sp

from numpy.typing import ArrayLike
from typing import List, Tuple

from .module import SingleTargetTracker
from .target import Target


__all__ = ['nees', 'validate_kalman_filter']


def nees(truth: ArrayLike, x_hat: ArrayLike, P_hat: ArrayLike):
    truth, x_hat, P_hat = np.array(truth), np.array(x_hat), np.array(P_hat)

    assert len(truth.shape) == 2

    # single run
    if len(x_hat.shape) == 2:
        assert len(P_hat.shape) == 3
        assert truth.shape == x_hat.shape
        assert P_hat.shape[0] == x_hat.shape[0]
        assert P_hat.shape[1] == P_hat.shape[2]
        assert P_hat.shape[1] == x_hat.shape[1]

        d = (x_hat-truth).reshape(x_hat.shape[0], x_hat.shape[1], 1) # column vector
        P_inv = np.linalg.inv(P_hat)
        
        return np.matmul(np.matmul(d.transpose(0, 2, 1), P_inv), d).squeeze()

    # multiple run case
    elif len(x_hat.shape) == 3:
        assert len(P_hat.shape) == 4
        assert truth.shape == x_hat.shape[1:]   # number of steps and state dimensions
        assert x_hat.shape[0] == P_hat.shape[0] # number of runs
        assert x_hat.shape[1] == P_hat.shape[1] # number of steps in each run
        assert x_hat.shape[2] == P_hat.shape[2] # number of dimensions in KF state
        assert P_hat.shape[2] == P_hat.shape[3] # covariance matrix is square
        
        d = x_hat - truth
        d = d.reshape(x_hat.shape[0], x_hat.shape[1], x_hat.shape[2], 1) # column vector
        P_inv = np.linalg.inv(P_hat)

        return np.matmul(np.matmul(d.transpose(0, 1, 3, 2), P_inv), d).squeeze()
    
    else:
        raise Exception("Unsupported input shape")



class KalmanFilterValidationReport:
    def __init__(self, nees_scores: ArrayLike, mc_runs: int):
        nees_scores = np.array(nees_scores)
        assert len(nees_scores.shape) == 2

        self.nees_scores = nees_scores
        self.mc_runs = mc_runs

    def confidence_interval(self):
        """Two-tailed Chi square confidence interval for the p-value 0.05
        """

        # the number of columns is equal to state_dim * mc_runs (the product of the
        # number of KF state dimensions and the number of Monte-Carlo runs)
        degrees_of_freedom = self.nees_scores.shape[1]

        p = 0.05
        return sp.stats.chi2.ppf([p/2, 1-p/2], degrees_of_freedom) / self.mc_runs

    def is_valid(self):
        average_nees_scores = np.sum(self.nees_scores, axis=0) / self.mc_runs
        ci = self.confidence_interval()
        within_range = np.bitwise_and(average_nees_scores > ci[0],
                                      average_nees_scores < ci[1])
        
        if np.mean(within_range) > 0.05:
            print(f"More than 5% NEES scores outside of confidence interval ({ci[0]}, {ci[1]}): {np.mean(within_range)}")
            return False
        else:
            print(f"Less than 5% NEES scores outside of confidence interval ({ci[0]}, {ci[1]}): {np.mean(within_range)}")
            return True


def validate_kalman_filter(trackers: List[SingleTargetTracker], target: Target):
    time, x_hat, P_hat, innovation = None, [], [], []

    for tracker in trackers:
        t_time, t_x_hat, t_P_hat, t_innovation = list(zip(*tracker.filter_trace))
        
        assert t_innovation[0] is None
        if time is None:
            time = np.array(t_time)
        else:
            assert np.array_equal(time, t_time)

        x_hat.append(np.array(t_x_hat))
        P_hat.append(np.array(t_P_hat))
        innovation.append(np.array(t_innovation[1:]))

    x_hat, P_hat, innovation = np.array(x_hat).squeeze(), np.array(P_hat).squeeze(), np.array(innovation).squeeze()

    mc_runs = len(trackers)
    nees_scores = nees(target.true_states(T=time), x_hat, P_hat)

    return KalmanFilterValidationReport(nees_scores, mc_runs)
