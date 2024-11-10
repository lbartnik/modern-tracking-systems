import numpy as np
import scipy as sp

from tracking_v2.kalman.turn import CoordinatedTurn
from tracking_v2.sensor import GeometricSensor
from tracking_v2.target import SingleTurnTarget


def test_f():
    ct = CoordinatedTurn([[1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0]],
                         [1, .02])

    ct.x_hat = np.array([[1, 2, 3, 4, 0]]).T
    np.testing.assert_almost_equal(ct.f(.5), np.array([[1, .5, 0,  0, 0],
                                                       [0,  1, 0,  0, 0],
                                                       [0,  0, 1, .5, 0],
                                                       [0,  0, 0,  1, 0],
                                                       [0,  0, 0,  0, 1]]))

    ct.x_hat = np.array([[1, 2, 3, 4, np.pi/6]]).T
    np.testing.assert_almost_equal(ct.f(.5), np.array([[1,  .49, 0, -.06, 0],
                                                       [0,  .96, 0, -.25, 0],
                                                       [0,  .06, 1,  .49, 0],
                                                       [0,  .25, 0,  .96, 0],
                                                       [0,   0,  0,   0,  1]]), 2)


def test_f_x():
    ct = CoordinatedTurn([[1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0]],
                         [1, .02])
    
    ct.x_hat = np.array([[1, 2, 3, 4, 0]]).T
    np.testing.assert_almost_equal(ct.f_x(.5), np.array([[1, .5, 0,  0, -.5],
                                                         [0,  1, 0,  0, -2],
                                                         [0,  0, 1, .5, .25],
                                                         [0,  0, 0,  1, 1],
                                                         [0,  0, 0,  0, 1]]))


    ct.x_hat = np.array([[1, 2, 3, 4, np.pi/6]]).T
    np.testing.assert_almost_equal(ct.f_x(.5), np.array([[1,  .49, 0, -.06, -.53],
                                                         [0,  .96, 0, -.25, -2.19],
                                                         [0,  .06, 1,  .49, .15],
                                                         [0,  .25, 0,  .96, .44],
                                                         [0,   0,  0,   0,   1]]), 2)


def test_q():
    ct = CoordinatedTurn([[1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0]],
                         [2, .02])

    print(ct.Q(1.5))
    np.testing.assert_almost_equal(ct.Q(1.5), np.array([[5.0625,  6.75, 0,      0,    0],
                                                        [6.75,    9,    0,      0,    0],
                                                        [0,       0,    5.0625, 6.75, 0],
                                                        [0,       0,    6.75,   9,    0],
                                                        [0,       0,    0,      0,    0.0009]]), 4)


def test_ct():
    target = SingleTurnTarget(30, 1)
    sensor = GeometricSensor(seed=0)
    true_positions = target.true_states()

    mc_runs = 200
    mc_nees_scores = []

    for _ in range(mc_runs):
        ct = CoordinatedTurn([[1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0]],
                             [1, .02289])
        t = 0
        ct.initialize(true_positions[t, :2], np.eye(2))

        x_hat, P_hat = [], []
        for _ in range(399):
            t += 1
            m = sensor.generate_measurement(t, true_positions[t, :3])

            ct.predict(1)
            ct.update(m.z[:, :2], m.R[:2, :2])

            x_hat.append(np.copy(ct.x_hat))
            P_hat.append(np.copy(ct.P_hat))

        x_hat, P_hat = np.array(x_hat).squeeze(), np.array(P_hat).squeeze()
        
        # select only positions
        x_hat = x_hat[:, [0, 2]]
        P_hat = P_hat[:, [0, 2], :][:, :, [0, 2]]

        d = x_hat - true_positions[1:, :2]
        d = d.reshape(399, 2, 1)
        P_inv = np.linalg.inv(P_hat)

        mc_run_scores = np.matmul(np.matmul(d.transpose(0, 2, 1), P_inv), d).squeeze()
        mc_nees_scores.append(mc_run_scores)
    
    mc_nees_scores = np.array(mc_nees_scores).T
    mc_nees_scores = np.mean(mc_nees_scores, axis=1)

    state_dim = 2
    conf_int = sp.stats.chi2.ppf([0.025, 0.975], mc_runs * state_dim) / mc_runs

    within_conf_int = np.logical_and(conf_int[0] <= mc_nees_scores, mc_nees_scores <= conf_int[1])

    # skip the first 15 observations where errors might be higher due to incorrect
    # initialization of the filter; throughout the rest of the time, with a 95%
    # confidence interval, we expect close to 95% of observations to fall within
    # that interval; however, the Coordinated Turn filter is not linear and
    # overestimates the covariance, which results in fewer observations (~81%)
    # falling within the confidence interval
    np.testing.assert_almost_equal(np.mean(within_conf_int[15:]), .8098, 4)

    # this is illustrated next, where ~19% of the observations are under the lower
    # bound of the confidence interval, whereas we would expect 2.5% for a
    # statistically consistent filter; the fact that all ~19% of observations outside
    # of the confidence interval are in fact less than its lower bound shows that
    # the filter systematically overestimates the covariance (as opposed to over-
    # and underestimating it, in which case we would expect half of those ~19% to
    # be below and half above the confidence interval)
    too_low = np.mean(mc_nees_scores < conf_int[0])
    np.testing.assert_almost_equal(too_low, .1929, 4)

