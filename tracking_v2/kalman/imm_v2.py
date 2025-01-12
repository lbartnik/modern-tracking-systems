import numpy as np

from typing import Callable, List, Tuple
from numpy.typing import ArrayLike


from .interface import KalmanFilter



class InteractingMultipleModels(object):
    """IMM: Interacting Multiple Models

    As defined in "Estimation with Applications to Tracking and Navigation" by
    Bar-Shalom, pp. 453-457.
    """

    def __init__(self, filters: List[Tuple[KalmanFilter, Callable, Callable]], transition_ps: ArrayLike):
        transition_ps = np.asarray(transition_ps)

        # state transition probabilities: the numbers of rows and columns must match
        # the number of filters; probabilities in rows much sum to 1
        assert transition_ps.shape == (len(filters), len(filters))
        assert np.array_equal(transition_ps.sum(axis=1), np.full(len(filters), 1))

        self.filters = filters

        # probability of target transitioning from state "i" to "j"
        self.p_ij = transition_ps
        
        # probability of target being in each of the modeled states
        self.mu_i = np.full(len(filters), 1 / len(filters))

        # mixing probabilities: posterior probability of target being previously
        # in state "i" given that it is currently in state "j"
        self.mu_ij = np.zeros((3, 3))

        self.x_hat = None
        self.P_hat = None
    
    def initialize(self, x: ArrayLike, P: ArrayLike):
        for f in self.filters:
            f.initialize(x, P)
    
    def predict(self, dt: float):
        # the "predict" API of a generic "Kalman Filter" defined in this package maps
        # to the first three steps of the IMM algorithm as defined on pp. 455-457 of
        # "Estimation with Applications to Tracking and Navigation":
        #
        #   1. calculation of the mixing probabilities
        #   2. mixing
        #   3. model-matched filtering - only the forward-prediction step
        self.calculate_mixing_probabilities()
        self.mix_states()

        for f in self.filters:
            f.predict(dt)
    
    def calculate_mixing_probabilities(self):
        r = len(self.filters)

        # normalizing constants
        c_j = np.zeros(r)
        for j in range(r):
            c_j[j] = np.sum([self.p_ij[i, j] * self.mu_i[i] for i in range(r)])

        for i in range(r):
            for j in range(r):
                self.mu_ij[i, j] = 1/c_j[j] * self.p_ij[i,j] * self.mu_i[i]
    
    def mix_states(self):
        r = self.filters

        # pre-calculate representation of each state in the common-dimensionality space
        state_in_common_space = [to_common(f.x_hat, f.P_hat) for f, to_common, _ in self.filters]

        x_0j = []
        for j in range(r):
            x_0 = [x_hat * self.mu_ij[i, j] for i, (x_hat, _) in enumerate(state_in_common_space)]
            x_0j.append(x_0)
        
        P_0j = []
        for j in range(r):
            P_0 = np.zeros_like(state_in_common_space[0][1])

            for i, (x_hat, P_hat) in enumerate(state_in_common_space):
                P_0 += self.mu_ij[i, j] * P_hat

                # spread of the means
                x_d = x_hat - x_0j[j]
                P_0 += self.mu_ij[i, j] * (x_d @ x_d.T)
            
            P_0j.append(P_0)
        
        # assign mixes states back to filters
        for (filter, _, from_common), x_0, P_0 in zip(self.filters, x_0j, P_0j):
            filter.x_hat, filter.P_hat = from_common(x_0, P_0)

    def update(self, z: ArrayLike, R: ArrayLike):
        # the "update" API of a generic "Kalman Filter" defined in this package maps
        # to the first three steps of the IMM algorithm as defined on pp. 455-457 of
        # "Estimation with Applications to Tracking and Navigation":
        #
        #   3. model-matched filtering - only the update step
        #   4. mode probability update
        #   5. estimate and covariance combination
        z, R = np.asarray(z), np.asarray(R)

        # each filter will retain its innovation and innovation covariance matrix S
        for f in self.filters:
            f.update(z, R)
        
