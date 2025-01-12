import numpy as np

from typing import List, Tuple
from numpy.typing import ArrayLike

from .interface import KalmanFilter
from ..np import as_column


__all__ = ['ImmNestedFilter', 'InteractingMultipleModels', 'CartesianStateTransform']


class CartesianStateTransform:
    """Linear transform of Cartesian mean and covariance to and from the common
    number of state dimensions. Missing dimensions in both mean and covariance
    (e.g. acceleration missing from a constant-velocity model) are filled with
    zeros.

    See: Section 4.5.4. "Combining Different State Models", pp. 227-229 in
    "Design and Analysis of Modern Tracking Systems".
    """
    def __init__(self, common_state_dim: int):
        self.common_state_dim = common_state_dim
    
    def to_common(self, a: ArrayLike) -> np.array:
        """Expand or shrink a to conform to the common state size. Mean is always
        returned as a column vector.

        Args:
            a (ArrayLike): Input mean or covariance.

        Returns:
            np.array: Mean or covariance matching the common number of state dimensions.
        """
        a = np.array(a)

        # all nested motion models must have at most the common number of state dimensions;
        # alert if a nested model has more dimensions than the common IMM dimensionality
        assert all(map(lambda x: x <= self.common_state_dim, a.shape))

        # covariance
        if len(a.shape) == 2 and a.shape[0] == a.shape[1]:
            if a.shape[0] == self.common_state_dim:
                return a
            
            r, c = a.shape
            return np.pad(a, ((0, self.common_state_dim-r), (0, self.common_state_dim-c)), mode='constant', constant_values=0)
        # mean
        else:
            a = as_column(a)
            if len(a) == self.common_state_dim:
                return a

            r = a.shape[0]
            return np.pad(a, ((0, self.common_state_dim-r), (0, 0)), mode='constant', constant_values=0)
    
    def from_common(self, a: ArrayLike, shape_like: Tuple[int, int]) -> np.array:
        a = np.array(a)

        # all nested motion models must have at most the common number of state dimensions;
        # alert if a nested model has more dimensions than the common IMM dimensionality
        assert all(map(lambda x: x <= self.common_state_dim, a.shape))

        r, c = shape_like
        return a[:r, :c]
                

class ImmNestedFilter:
    def __init__(self, filter: KalmanFilter, transform: CartesianStateTransform):
        self.filter = filter
        self.transform = transform
    
    def initialize(self, x: ArrayLike, P: ArrayLike):
        self.filter.initialize(x, P)

    def predict(self, dt: float):
        self.filter.predict(dt)
    
    def update(self, z: ArrayLike, R: ArrayLike):
        self.filter.update(z, R)
    
    def to_common_x_hat(self):
        return self.transform.to_common(self.filter.x_hat)

    def from_common_x_hat(self, x: ArrayLike):
        self.filter.x_hat = self.transform.from_common(x, self.filter.x_hat.shape)
    
    def to_common_P_hat(self):
        return self.transform.to_common(self.filter.P_hat)

    def from_common_P_hat(self, P: ArrayLike):
        self.filter.P_hat = self.transform.from_common(P, self.filter.P_hat.shape)
    
    @property
    def innovation(self):
        return self.filter.innovation

    @property
    def S(self):
        return self.filter.S



class InteractingMultipleModels(object):
    """IMM: Interacting Multiple Models

    Generic IMM implementation as described in "Design and Analysis of Modern
    Tracking Systems", pp. 221-232. The sequence of steps (mixing, prediction,
    gating, probabilities update, filtered estimates), illustrated in the
    diagram on p. 223, is split between the predict() and update() methods of
    this class such that its interface is compatible with that of individual
    Kalman Filters implemented in this package.
    """

    def __init__(self, filters: List[ImmNestedFilter], transition_ps: ArrayLike):
        transition_ps = np.array(transition_ps)

        assert transition_ps.shape == (len(filters), len(filters))
        assert np.array_equal(transition_ps.sum(axis=1), np.full(len(filters), 1))
        assert len(np.unique([f.transform.common_state_dim for f in filters])) == 1 # all transforms use the same common state representation

        self.filters = filters

        # probability of target transitioning from state i to j
        self.P_ij = transition_ps
        
        # probability of target being in each of the modeled states
        self.mu_i = np.full(len(filters), 1 / len(filters))

        # probability of target transitioning from i to j
        self.mu_ij = np.zeros((3, 3))

        # probability that target is in state j after the interaction
        self.C_j = np.full(len(filters), 1 / len(filters))

        self.x_hat = None
        self.P_hat = None

    def initialize(self, x: ArrayLike, P: ArrayLike):
        for f in self.filters:
            f.initialize(x, P)

    def predict(self, dt: float):
        """Mix states calculated in the previous iteration and the forward-predict
        them by the given time delta.

        Args:
            dt (float): Time delta used in forward-prediction.
        """
        # 4.5.1 IMM Interaction/Mixing and Prediction
        self.mix_states()

        for f in self.filters:
            f.predict(dt)

    def update(self, z: ArrayLike, R: ArrayLike):
        """Update each nested model (Kalman Filter) using measurement `z` and its
        covariance `R`, then recalculate probabilities of nested models and calculate
        the collective estimate of the target state.

        Args:
            z (ArrayLike): Measurement.
            R (ArrayLike): Covariance matrix of the measurement error.
        """
        z, R = np.asarray(z), np.asarray(R)

        # each filter will retain its innovation vector and the accompanying
        # covariance matrix S
        for f in self.filters:
            f.update(z, R)

        # 4.5.3 Likelihood calculation and Model Probability Update
        self.update_probabilities()

        # the collective state estimate is a weighted sum of estimates produced
        # by the nested Kalman Filters
        x_hat = [f.to_common_x_hat() * C_j for f, C_j in zip(self.filters, self.C_j)]
        self.x_hat = np.asarray(x_hat).sum(axis=0)

        P_hat = [f.to_common_P_hat() * C_j for f, C_j in zip(self.filters, self.C_j)]
        self.P_hat = np.asarray(P_hat).sum(axis=0)

    def mix_states(self):
        """Mix state estimates given the current probability of each nested model
        and the transition matrix.
        """
        # 4.5.1 IMM Interaction/Mixing and Prediction
        
        # part 1: update posterior probabilities

        # inputs to equations (4.46) and (4.47): multiply each row of P_ij (probability
        # of transition from state i to j) by respective mu_i (probability of being in
        # state i)
        P_times_mu = (self.P_ij.T * self.mu_i).T

        # (4.47): the probability that after interaction, the target is in state j
        self.C_j = np.sum(P_times_mu, axis=0)
        
        # (4.46): conditional probability given that the target is in state j that
        # the transition occurred from state i
        self.mu_ij = P_times_mu * 1/self.C_j

        # part 2: mix states

        # obtain states represented in the common state space
        states = [(f.to_common_x_hat(), f.to_common_P_hat()) for f in self.filters]

        # mix means: equation (4.48)
        x0s = []
        for j in range(len(self.filters)):
            means = np.hstack([mean for mean, _ in states])
            means = means * self.mu_ij[:, j].squeeze()
            mean = means.sum(axis=1)
            x0s.append(as_column(mean))
        
        # mix covariances: equations (4.49) and (4.50)
        P0s = []
        for j in range(len(self.filters)):
            P0_j = []
            for i in range(len(self.filters)):
                mean_i, cov_i = states[i]

                # difference in state estimates between models i and j, before and after
                # the mixing; even for i==j this is non-zero as x0s[i] has been already
                # updated and differs from mean_i
                Dx = mean_i - x0s[j]

                DP_ij = Dx @ Dx.T

                P0_j.append(self.mu_ij[i, j] * (cov_i + DP_ij))

            P0s.append(np.array(P0_j).sum(axis=0))

        # transform mixed states back into filter-specific state representations
        for x0, P0, f in zip(x0s, P0s, self.filters):
            f.from_common_x_hat(x0)
            f.from_common_P_hat(P0)

    def update_probabilities(self):
        """Update probabilities of nested models `self.mu_i` given the normalized
        estimation error (squared Mahalanobis distance) for each model.
        """

        # dimensionality of the measurement space
        M = len(self.filters[0].innovation)

        # 4.5.3 Likelihood calculation and Model Probability Update

        # likelihood of the observation given model, equation (4.54)
        gamma = []
        for f in self.filters:
            S_inv = np.linalg.inv(f.S)
            d2 = (f.innovation.T @ S_inv @ f.innovation).squeeze()
            gamma_i = np.exp(-d2/2) / np.sqrt( (2*np.pi)**M * np.trace(f.S))
            gamma.append(gamma_i)
        
        # update probabilities of models using Bayes rule, equation (4.55)
        u_i = np.array(gamma) * self.C_j
        self.mu_i = u_i / np.sum(u_i)
