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

        # mode transition probabilities: the numbers of rows and columns must match
        # the number of filters; probabilities in rows much sum to 1
        assert transition_ps.shape == (len(filters), len(filters))
        assert np.array_equal(transition_ps.sum(axis=1), np.full(len(filters), 1))

        self.filters: List[Tuple[KalmanFilter, Callable, Callable]] = filters

        # probability of target transitioning from mode "i" to "j" (Eq. 11.6.3-8)
        #
        # * rows and columns meaning consistent with (4.45) (p. 222) of "Design and
        #   Analysis of Modern Tracking Systems"
        self.p_ij = transition_ps
        
        # probability of target being in each of the modes
        self.mu_j = np.full(len(filters), 1 / len(filters))

        # mixing probabilities: posterior probability of target being previously
        # in mode "i" given that it is currently in mode "j"
        self.mu_ij = np.zeros((len(filters), len(filters)))

        # likelihood of target being in mode "j" given the measurement received in
        # the current iteration
        self.lambda_j = np.zeros(len(filters))

        # reset() initializes x_hat and P_hat to values taken from the first filter
        self.x_hat, self.P_hat = None, None
        self.reset()
    
    def reset(self):
        for filter, _, _ in self.filters:
            filter.reset()
        
        kf, to_common, _ = self.filters[0]
        self.x_hat, self.P_hat = to_common(kf.x_hat, kf.P_hat)
        

    def initialize(self, x: ArrayLike, P: ArrayLike):
        for f, _, _ in self.filters:
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

        for f, _, _ in self.filters:
            f.predict(dt)
        
        # during evaluation, we collect x_hat and P_hat right after calling predict:
        # this is why we need to combine estimates not only in update(), but also here
        self.combine_estimates()

    # c_j is the probability that the target moved into mode "j"
    #
    # it is used as the normalizing factor when calculating mixing probalities (which
    # are posterior probabilities; see calculate_mixing_probabilities()) and when
    # calculating mode probabilities for the next iteration (see update_mode_probabilities())
    def calculate_c_j(self):
        r = len(self.filters)

        # normalizing constants, eq. (11.6.6-8)
        c_j = np.zeros(r)
        for j in range(r):
            c_j[j] = np.sum([self.p_ij[i, j] * self.mu_j[i] for i in range(r)])
        
        return np.asarray(c_j)

    def calculate_mixing_probabilities(self):
        r = len(self.filters)
        c_j = self.calculate_c_j()

        for i in range(r):
            for j in range(r):
                self.mu_ij[i, j] = 1/c_j[j] * self.p_ij[i,j] * self.mu_j[i]
    
    def mix_states(self):
        r = len(self.filters)

        # pre-calculate representation of each state in the common-dimensionality space
        state_in_common_space = [to_common(f.x_hat, f.P_hat) for f, to_common, _ in self.filters]

        x_0j = []
        for j in range(r):
            x_0 = np.zeros_like(state_in_common_space[0][0])
            for i, (x_hat, _) in enumerate(state_in_common_space):
                x_0 += x_hat * self.mu_ij[i, j]
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
            filter.x_hat, filter.P_hat = from_common(x_0, P_0, filter.x_hat, filter.P_hat)

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
        for f, _, _ in self.filters:
            f.update(z, R)
        

        self.calculate_likelihoods()
        self.update_mode_probabilities()
        self.combine_estimates()
    
    def calculate_likelihoods(self):
        # Part of step 3: mode-matched filtering
        # Eq. (11.6.6-12)
        for j, (f, _, _) in enumerate(self.filters):
            self.lambda_j[j] = _normal_pdf(f.innovation, f.S)
    
    def update_mode_probabilities(self):
        # Step 4: mode probability update
        c_j = self.calculate_c_j()
        
        # Eq. (11.6.6-15)
        self.mu_j = c_j * self.lambda_j
        self.mu_j = self.mu_j / np.sum(self.mu_j)
    
    def combine_estimates(self):
        # Step 5: estimate and covariance combination
        state_in_common_space = [to_common(f.x_hat, f.P_hat) for f, to_common, _ in self.filters]

        x_hat = np.zeros_like(state_in_common_space[0][0])
        for mu_j, (x_j, _) in zip(self.mu_j, state_in_common_space):
            x_hat += mu_j * x_j

        P_hat = np.zeros_like(state_in_common_space[0][1])
        for mu_j, (x_j, P_j) in zip(self.mu_j, state_in_common_space):
            xd = x_j - x_hat
            P_hat += mu_j * (P_j + xd @ xd.T)
        
        self.x_hat = x_hat
        self.P_hat = P_hat


def _normal_pdf(x: ArrayLike, Sigma: ArrayLike) -> float:
    x, Sigma = np.asarray(x), np.asarray(Sigma)

    assert x.shape == (len(x), 1)           # column vector
    assert Sigma.shape == (len(x), len(x))  # square matrix, matches dimensionality

    N         = len(x)
    Sigma_inv = np.linalg.inv(Sigma)
    d2        = (x.T @ Sigma_inv @ x).squeeze()
    assert d2.size == 1

    return np.exp(-d2/2) / np.sqrt( (2*np.pi)**N * np.linalg.det(Sigma))