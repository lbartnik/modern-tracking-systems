import pytest
import numpy as np


from tracking_v2.kalman.imm_v2 import InteractingMultipleModels, _normal_pdf
from tracking_v2.kalman import LinearKalmanFilter
from tracking_v2.motion import ConstantVelocityModel


def test_imm_instantiate():
    # during instantiation only the number of filters matters, not whether
    # they are valid objects
    filters = [None, None, None]

    transition_ps = [[.8, .1, .1],
                     [.1, .6, .3],
                     [.15, .05, .8]]
    
    # correct instantiation
    imm = InteractingMultipleModels(filters, transition_ps)

    # 2 filters but 3 transitions probabilities
    with pytest.raises(AssertionError):
        InteractingMultipleModels([None, None], transition_ps)

    # prob more than one
    with pytest.raises(AssertionError):
        transition_ps[0][2] = .2
        InteractingMultipleModels(filters, transition_ps)

    # prob less than one
    with pytest.raises(AssertionError):
        transition_ps[0][2] = .05
        InteractingMultipleModels(filters, transition_ps)


def _create_cv(x0 = 0):
    kf = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 0]])
    kf.x_hat[0] = x0
    return kf


def _create_imm() -> InteractingMultipleModels:
    def _identity(x, P, *args): return x, P

    filters = [(_create_cv(1), _identity, _identity),
               (_create_cv(2), _identity, _identity),
               (_create_cv(3), _identity, _identity)]
    
    transition_ps = [[.8, .1, .1],
                     [.2, .6, .2],
                     [.2, .4, .4]]
    
    imm = InteractingMultipleModels(filters, transition_ps)
    imm.mu_j = np.array([.1, .2, .7])

    return imm


# probabilities that target transitions to mode "j"
def test_calculate_c_j():
    c_j = _create_imm().calculate_c_j()

    assert c_j[0] == .1 * .8 + .2 * .2 + .7 * .2
    assert c_j[1] == .1 * .1 + .2 * .6 + .7 * .4
    assert c_j[2] == .1 * .1 + .2 * .2 + .7 * .4


# mixing probability: probability that target was in mode "i" multiplied by
# the probability that, if target is in mode "j", it transitioned from mode "i"
def test_calculate_mixing_probabilities():
    c_j = np.zeros(3)
    c_j[0] = .1 * .8 + .2 * .2 + .7 * .2
    c_j[1] = .1 * .1 + .2 * .6 + .7 * .4
    c_j[2] = .1 * .1 + .2 * .2 + .7 * .4

    imm = _create_imm()
    imm.calculate_mixing_probabilities()
    
    # [i, j]
    assert np.allclose(imm.mu_ij[0, 0], .1 * .8 / c_j[0], 1e-4)
    assert np.allclose(imm.mu_ij[0, 1], .1 * .1 / c_j[1], 1e-4)
    assert np.allclose(imm.mu_ij[0, 2], .1 * .1 / c_j[2], 1e-4)
    assert np.allclose(imm.mu_ij[1, 0], .2 * .2 / c_j[0], 1e-4)
    assert np.allclose(imm.mu_ij[1, 1], .2 * .6 / c_j[1], 1e-4)
    assert np.allclose(imm.mu_ij[1, 2], .2 * .2 / c_j[2], 1e-4)
    assert np.allclose(imm.mu_ij[2, 0], .7 * .2 / c_j[0], 1e-4)
    assert np.allclose(imm.mu_ij[2, 1], .7 * .4 / c_j[1], 1e-4)
    assert np.allclose(imm.mu_ij[2, 2], .7 * .4 / c_j[2], 1e-4)


def test_mix_states():
    imm = _create_imm()
    imm.mu_ij = np.array([[.1, .3, .6],
                          [.2, .3, .5],
                          [.3, .5, .2]])
    
    imm.mix_states()

    assert np.allclose(imm.filters[0][0].x_hat.T, [.1 * 1 + .2 * 2 + .3 * 3, 0, 0, 0, 0, 0], 1e-6)
    assert np.allclose(imm.filters[1][0].x_hat.T, [.3 * 1 + .3 * 2 + .5 * 3, 0, 0, 0, 0, 0], 1e-6)
    assert np.allclose(imm.filters[2][0].x_hat.T, [.6 * 1 + .5 * 2 + .2 * 3, 0, 0, 0, 0, 0], 1e-6)

    assert np.allclose(imm.filters[0][0].P_hat,
                       np.diag([.1 + .2 + .3 + .1 * (1 - 1.4)**2 + .2 * (2 - 1.4)**2 + .3 * (3 - 1.4)**2, .6, .6, .6, .6, .6]),
                       1e-6)

    assert np.allclose(imm.filters[1][0].P_hat,
                       np.diag([.3 + .3 + .5 + .3 * (1 - 2.4)**2 + .3 * (2 - 2.4)**2 + .5 * (3 - 2.4)**2, 1.1, 1.1, 1.1, 1.1, 1.1]),
                       1e-6)

    assert np.allclose(imm.filters[2][0].P_hat,
                       np.diag([.6 + .5 + .2 + .6 * (1 - 2.2)**2 + .5 * (2 - 2.2)**2 + .2 * (3 - 2.2)**2, 1.3, 1.3, 1.3, 1.3, 1.3]),
                       1e-6)


def test_normal_pdf():
    assert np.allclose(_normal_pdf([[0]], [[1]]), 1/np.sqrt(2 * np.pi), 1e-6)
    assert np.allclose(_normal_pdf(np.zeros((2, 1)), np.eye(2)), 1/(2 * np.pi), 1e-6)

    # scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 3]]).pdf([1,2])
    assert np.allclose(_normal_pdf([[1], [2]], [[1, .5], [.5, 3]]), .0386670, 1e-6)


def test_calculate_likelihoods():
    imm = _create_imm()

    imm.filters[0][0].innovation = np.zeros((2, 1))
    imm.filters[0][0].S          = np.eye(2)

    imm.filters[1][0].innovation = [[1], [2]]
    imm.filters[1][0].S          = [[1, .5], [.5, 3]]

    imm.filters[2][0].innovation = [[1], [2]]
    imm.filters[2][0].S          = np.diag([2, 3])

    imm.calculate_likelihoods()

    # scipy.stats.multivariate_normal(mean=[0, 0], cov=[[2, 0], [0, 3]]).pdf([1,2])
    assert np.allclose(imm.lambda_j, [1/(2 * np.pi), .0386670, .0259801], 1e-6)


def test_update_mode_probabilities():
    imm = _create_imm()
    imm.lambda_j = [.1, .2, .3]

    c_j = [0.26, 0.41, 0.33]
    assert np.allclose(imm.calculate_c_j(), c_j, 1e-2)  

    imm.update_mode_probabilities()

    mu_j = np.asarray([.26 * .1, .41 * .2, .33 * .3]) / (.026 + .082 + .099)
    assert np.allclose(imm.mu_j, mu_j, 1e-3)


def test_combine_estimates():
    imm = _create_imm()

    imm.filters[0][0].P_hat *= 2
    imm.filters[1][0].P_hat *= 3
    imm.filters[2][0].P_hat *= 4

    imm.mu_j = [.1, .3, .6]

    imm.combine_estimates()

    # x_hat for filters initialized to [1, 0, ...], [2, 0, ...], [3, 0, ...]
    assert np.allclose(imm.x_hat.T, [.1 * 1 + .3 * 2 + .6 * 3, 0, 0, 0, 0, 0])
    assert np.allclose(imm.P_hat, np.diag([.1 * (2 + (2.5 - 1)**2) + .3 * (3 + (2.5-2)**2) + .6 * (4 + (2.5-3)**2),
                                           .1*2 + .3 * 3 + .6 * 4,
                                           .1*2 + .3 * 3 + .6 * 4,
                                           .1*2 + .3 * 3 + .6 * 4,
                                           .1*2 + .3 * 3 + .6 * 4,
                                           .1*2 + .3 * 3 + .6 * 4]))
