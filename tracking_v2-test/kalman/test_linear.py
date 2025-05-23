import numpy as np
from numpy.testing import assert_equal, assert_allclose
from copy import deepcopy

from tracking_v2.kalman.linear import LinearKalmanFilter
from tracking_v2.motion import ConstantVelocityModel
from tracking_v2.sensor import GeometricSensor
from tracking_v2.target import ConstantVelocityTarget


def test_initialize():
    kf = LinearKalmanFilter(ConstantVelocityModel(), [[1, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 0]])
    kf.initialize([1, 2, 3], np.diag([4, 5, 6]))

    assert_equal(kf.x_hat.T, [[1, 2, 3, 0, 0, 0]])
    assert_equal(kf.P_hat, np.diag([4, 5, 6, 1, 1, 1]))



def test_update():
    kf = LinearKalmanFilter(ConstantVelocityModel(0), [[1, 0, 0, 0, 0, 0],
                                                       [0, 1, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0]])
    kf.initialize([1, 1, 1], np.diag([1, 1, 1]))
    kf.predict(1)
    kf.update([2, 3, 4], np.diag([1, 1, 1]))

    assert_allclose(kf.x_hat.T, [[1.666, 2.333, 3, .333, .666, 1]], .01)
    assert_allclose(kf.P_hat, [[.667, 0, 0, .333, 0, 0],
                               [0, .667, 0, 0, .333, 0],
                               [0, 0, .667, 0, 0, .333],
                               [.333, 0, 0, .667, 0, 0],
                               [0, .333, 0, 0, .667, 0],
                               [0, 0, .333, 0, 0, .667]], .01)


def test_nees():
    # zero noise intensity matches the lack of actual motion noise in the output
    # of Constant-Velocity Target; using non-zero noise intensity leads to
    # overestimated covariance values which then leads to underestimated NEES
    # values which do not pass the Chi-square test
    motion_model = ConstantVelocityModel(noise_intensity=0)

    target = ConstantVelocityTarget()
    monte_carlo_n = 100

    # mean and covariance estimated by a Kalman Filter are drawn from a non-ergodic
    # random process (implemented by the Kalman Filter) and thus we cannot analyze
    # a single test run but instead need to draw samples from multiple independent
    # Monte-Carlo runs
    monte_carlo_x, monte_carlo_P = [], []

    for i in range(monte_carlo_n):
        # use distinct random seeds to provide different noise values in each run
        sensor = GeometricSensor(seed=i)
        
        kf = LinearKalmanFilter(motion_model, [[1, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0]])

        true_positions = target.true_states()
        kf.initialize(true_positions[0, :], np.eye(3))

        # iterate over the rest of measurements and keep track of filter's state (mean and cov)
        x_hat, P_hat = [], []
        for t, position in enumerate(true_positions[1:, :3]):
            m = sensor.generate_measurement(t, [position])

            kf.predict(1)

            x_hat.append(np.copy(kf.x_hat))
            P_hat.append(np.copy(kf.P_hat))

            kf.update(m.z, m.R)

        monte_carlo_x.append(np.array(x_hat))
        monte_carlo_P.append(np.array(P_hat))

    monte_carlo_x, monte_carlo_P = np.array(monte_carlo_x).squeeze(), np.array(monte_carlo_P)

    # x_hat is an array of column vectors
    d = monte_carlo_x - true_positions[1:, :]
    d = d.reshape(monte_carlo_n, 399, 6, 1)

    # P_inv is an array of square matrices
    P_inv = np.linalg.inv(monte_carlo_P)

    mc_nees_scores = np.matmul(np.matmul(d.transpose(0, 1, 3, 2), P_inv), d).squeeze()
    average_nees_scores = np.sum(mc_nees_scores, axis=0) / monte_carlo_n

    # confidence interval for the test is derived as the critical values for a
    # two-tailed Chi-square test with 600 degrees of freedom; since we are average
    # NEES scores across MC runs, we divide these critical values by the number of
    # MC runs; this follows the approach presented in chapter 5.4 of "Estimation
    # with Applications to Tracking and Navigation" by Bar-Shalom, Li, Kirubarajan
    #
    # state_dim, monte_carlo_n = 6, 100
    # scipy.stats.chi2.ppf([0.025, 0.975], monte_carlo_n * state_dim) / monte_carlo_n
    # array([5.3401855 , 6.69769152])
    critical_region_lower, critical_region_upper = [5.3401855 , 6.69769152]

    assert np.mean(average_nees_scores[75:] < critical_region_lower) <= 0.025
    assert np.mean(average_nees_scores[75:] > critical_region_upper) <= 0.025


def test_predict():
    kf1 = LinearKalmanFilter(ConstantVelocityModel(1), [[1, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0],
                                                        [0, 0, 1, 0, 0, 0]])
    kf1.x_hat[3,0] = 10
    kf1.x_hat[4,0] = 5

    kf2 = deepcopy(kf1)

    kf1.predict(2)
    kf1.predict(3)

    kf2.predict(5)

    assert_allclose(kf1.x_hat-kf2.x_hat, np.zeros((6, 1)), rtol=.1)
    assert_allclose(kf1.P_hat-kf2.P_hat, np.zeros((6, 6)), rtol=.1)

    assert_allclose(kf1.x_hat, np.array([[50, 25, 0, 10, 5, 0]]).T, rtol=.1)
    assert_allclose(kf1.P_hat, np.array([
       [ 67.66,     0,     0, 17.5,    0,    0],
       [     0, 67.66,     0,    0, 17.5,    0],
       [     0,     0, 67.66,    0,    0, 17.5],
       [  17.5,     0,     0,    6,    0,    0],
       [     0,  17.5,     0,    0,    6,    0],
       [     0,     0,  17.5,    0,    0,    6]
    ]), rtol=.1)