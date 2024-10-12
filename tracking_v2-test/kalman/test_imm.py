import numpy as np
import pytest

from tracking_v2.motion import ConstantVelocityModel, ConstantAccelerationModel
from tracking_v2.kalman import LinearKalmanFilter, InteractingMultipleModels, CartesianStateTransform, ImmNestedFilter
from tracking_v2.target import SinusTarget
from tracking_v2.sensor import GeometricSensor


def test_cartesian_state_transform_mean():
    t = CartesianStateTransform(3)

    a = t.to_common(np.array([[1, 2]]).T)
    assert np.array_equal(a, np.array([[1, 2, 0]]).T)

    t = CartesianStateTransform(9)
    a = t.to_common(np.ones((6, 1)))
    assert np.array_equal(a, np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0]]).T)


def test_cartesian_state_transform_covariance():
    t = CartesianStateTransform(3)
    
    a = t.to_common(np.full((2, 2), 1))
    assert np.array_equal(a, np.array([[1, 1, 0],
                                       [1, 1, 0],
                                       [0, 0, 0]]))
    
    a = t.to_common(np.full((3, 3), 1))
    assert np.array_equal(a, np.array([[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]]))
    
    a = t.from_common(np.full((3, 3), 1), (2, 2))
    assert np.array_equal(a, np.array([[1, 1],
                                       [1, 1]]))
    

def test_imm_instantiate():
    transform = CartesianStateTransform(9)
    filters = [ImmNestedFilter(None, transform)] * 3

    transition_ps = [[.8, .1, .1],
                     [.1, .6, .3],
                     [.15, .05, .8]]
    
    InteractingMultipleModels(filters, transition_ps)

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


# verify that model probabilities are correctly updated
def test_imm_model_probabilities():
    # three valid nested Kalman Filters must be provided in order to avoid
    # errors but their states are not checked in this test
    cv = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 0]])
    transform = CartesianStateTransform(9)
    filters = [ImmNestedFilter(cv, transform)] * 3

    # dimensions must match the number of nested filters
    transition_ps = [[.8, .1, .1],
                     [.1, .6, .3],
                     [.15, .05, .8]]
    
    imm = InteractingMultipleModels(filters, transition_ps)
    imm.mu_i = np.array([.75, .2, 0.05])

    # calculates new model probabilities and mixes means and covariances
    imm.mix_states()
    
    # as calculated in "Modern Tracking Systems", p. 225
    assert np.allclose(imm.C_j, [.6275, .1975, .175])
    assert np.allclose(imm.mu_ij, [[0.9561753,  0.37974684, 0.42857143],
                                   [0.03187251, 0.60759494, 0.34285714],
                                   [0.01195219, 0.01265823, 0.22857143]])


def test_imm_update_probabilities():
    cv1 = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                       [0, 1, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0]])
    cv2 = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                       [0, 1, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0]])

    # set up states for mixing: initialize and update, leaving non-zero
    # innovation and S
    cv1.initialize([0, 0, 0], np.eye(3))
    cv2.initialize([3, 3, 3], np.eye(3))

    cv1.update([1, 1, 1], np.eye(3))
    assert np.array_equal(cv1.innovation.squeeze(), [1, 1, 1])
    assert np.array_equal(cv1.S, np.eye(3)*2)

    cv2.update([1, 1, 1], np.eye(3))
    assert np.array_equal(cv2.innovation.squeeze(), [-2, -2, -2])
    assert np.array_equal(cv2.S, np.eye(3)*2)

    # mix: recalculate conditional probabilities and update means and covariances
    transform = CartesianStateTransform(6)
    filters = [ImmNestedFilter(cv1, transform), ImmNestedFilter(cv2, transform)]
    imm = InteractingMultipleModels(filters, [[.8, .2], [.2, .8]])

    imm.update_probabilities()

    # using (4.54) and (4.55)
    lambda_0 = np.exp( -(1/2 + 1/2 + 1/2)/2 ) / np.sqrt((2*np.pi)**3 * 2*2*2)
    lambda_1 = np.exp( -(4/2 + 4/2 + 4/2)/2 ) / np.sqrt((2*np.pi)**3 * 2*2*2)

    assert np.allclose(imm.mu_i[0], lambda_0 / (lambda_0 + lambda_1))
    assert np.allclose(imm.mu_i[1], lambda_1 / (lambda_0 + lambda_1))


def test_imm_mix_states():
    cv1 = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                       [0, 1, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0]])
    cv2 = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                       [0, 1, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0]])

    cv1.initialize([0, 0, 0], np.eye(3))
    cv2.initialize([3, 3, 3], np.eye(3))

    # mix: recalculate conditional probabilities and update means and covariances
    transform = CartesianStateTransform(6)
    filters = [ImmNestedFilter(cv1, transform), ImmNestedFilter(cv2, transform)]
    imm = InteractingMultipleModels(filters, [[.8, .2], [.2, .8]])

    # IMM initializes model probabilities to equal
    assert np.array_equal(imm.mu_i, [.5, .5])

    imm.mix_states()

    # state-mixing does not change the model probabilities
    assert np.array_equal(imm.mu_i, [.5, .5])

    # conditional probabilities remain the same because mu_i is still [0.5, 0.5]
    assert np.array_equal(imm.C_j, [.5, .5])
    assert np.array_equal(imm.mu_ij, [[.8, .2], [.2, .8]])

    print(imm.C_j, imm.mu_ij)
    print(cv2.x_hat, cv2.P_hat)

    # .8 * [0, 0, 0] + .2 * [3, 3, 3]
    assert np.allclose(cv1.x_hat.squeeze(), [.6, .6, .6, 0, 0, 0])
    
    # Dx[0, 0] = [0, 0, 0] - [.6, .6, .6] = [-.6, -.6, -.6]
    # Dx[0, 1] = [3, 3, 3] - [.6, .6, .6] = [2.4, 2.4, 2.4]
    #
    # \sum_{i=1}^r ( mu_ij * (P_hat + Dx @ Dx.T) )
    #
    # [0,0]: .8 * (1 + (-.6)^2) + .2 * (1 + 2.4^2) = 2.44
    # [i,j]: .8 * .6^2 + .2 * 2.4^2 = 1.44
    assert np.allclose(cv1.P_hat, [[2.44, 1.44, 1.44, 0, 0, 0],
                                   [1.44, 2.44, 1.44, 0, 0, 0],
                                   [1.44, 1.44, 2.44, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]])

    # .2 * [0, 0, 0] + .8 * [3, 3, 3]
    assert np.allclose(cv2.x_hat.squeeze(), [2.4, 2.4, 2.4, 0, 0, 0])

    # Dx[0, 0] = [0, 0, 0] - [2.4, 2.4, 2.4] = [-2.4, -2.4, -2.4]
    # Dx[0, 1] = [3, 3, 3] - [2.4, 2.4, 2.4] = [.6, .6, .6]
    #
    # \sum_{i=1}^r ( mu_ij * (P_hat + Dx @ Dx.T) )
    #
    # [0,0]: .2 * (1 + (-2.4)^2) + .8 * (1 + .6^2) = 2.44
    # [i,j]: .2 * (-2.4)^2 + .2 * .6^2 = 1.44
    assert np.allclose(cv2.P_hat, [[2.44, 1.44, 1.44, 0, 0, 0],
                                   [1.44, 2.44, 1.44, 0, 0, 0],
                                   [1.44, 1.44, 2.44, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]])


# run a simple, single-target tracking test and confirm that the Constant-Acceleration
# model is better (has a higher probability) than the Constant-Velocity model during
# target U-turns
def test_imm_track():
    target = SinusTarget(30, 3)
    sensor = GeometricSensor(seed=0)
    true_positions = target.true_states()

    cv = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 0]])

    ca = LinearKalmanFilter(ConstantAccelerationModel(), [[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0, 0, 0, 0]])
    transform = CartesianStateTransform(9)
    filters = [ImmNestedFilter(cv, transform), ImmNestedFilter(ca, transform)]

    imm = InteractingMultipleModels(filters, [[.8, .2], [.2, .8]])
    imm.initialize(true_positions[0, :], np.eye(3))

    ps = []
    for t in range(1, 399):
        m = sensor.generate_measurement(t, true_positions[t, :3])

        imm.predict(1)
        imm.update(m.z, m.R)

        ps.append(imm.mu_i)
    ps = np.array(ps)
    
    # first U-turn, Constant-Acceleration model is better with a single (random) exception
    assert np.sum(ps[44:82, 0] < ps[44:82, 1]) == (82-44-1)

    # second U-turn, Constant-Acceleration model is once again better with another
    # (random) exception
    assert np.sum(ps[155:192, 0] < ps[155:192, 1]) == (192-155-1)
