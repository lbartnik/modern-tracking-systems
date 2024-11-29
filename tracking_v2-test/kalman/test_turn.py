import numpy as np
import scipy as sp

from tracking_v2.kalman.turn import CoordinatedTurn2D, CoordinatedTurn
from tracking_v2.sensor import GeometricSensor
from tracking_v2.target import SingleTurnTarget


# confirm that the state transition matrix is calculated correctly given the
# horizontal velocities and the turn rate
def test_ct2d_f():
    ct = CoordinatedTurn2D([[1, 0, 0, 0, 0],
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


# confirm that the covariance transition matrix is calculated correctly given the
# horizontal velocities and the turn rate
def test_ct2d_f_x():
    ct = CoordinatedTurn2D([[1, 0, 0, 0, 0],
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


# confirm that the process noise matrix is calculated correctly given the
# initial noise sigmas and delta time
def test_ct2d_q():
    ct = CoordinatedTurn2D([[1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0]],
                           [2, .02])

    print(ct.Q(1.5))
    np.testing.assert_almost_equal(ct.Q(1.5), np.array([[5.0625,  6.75, 0,      0,    0],
                                                        [6.75,    9,    0,      0,    0],
                                                        [0,       0,    5.0625, 6.75, 0],
                                                        [0,       0,    6.75,   9,    0],
                                                        [0,       0,    0,      0,    0.0009]]), 4)


# confirm that NEES remains within the predicted confidence interval
# certain fraction of the time
def test_ct2d_conf_int():
    target = SingleTurnTarget(30, 1)
    sensor = GeometricSensor(seed=0)
    true_positions = target.true_states()

    mc_runs = 200
    mc_nees_scores = []

    for _ in range(mc_runs):
        ct = CoordinatedTurn2D([[1, 0, 0, 0, 0],
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



# --- Coordinated Turn 3D ---

# confirm that the state transition matrix is calculated correctly given the
# horizontal velocities and the turn rate
def test_ct3d_f():
    ct = CoordinatedTurn([1, 1, 1, .02])

    # Omega = 0
    ct.x_hat = np.array([[1, 3, 5, 2, 4, 6, 0]]).T
    np.testing.assert_almost_equal(ct.f(.5), np.array([[1, 0, 0, .5,  0,  0, 0],
                                                       [0, 1, 0,  0, .5,  0, 0],
                                                       [0, 0, 1,  0,  0, .5, 0],
                                                       [0, 0, 0,  1,  0,  0, 0],
                                                       [0, 0, 0,  0,  1,  0, 0],
                                                       [0, 0, 0,  0,  0,  1, 0],
                                                       [0, 0, 0,  0,  0,  0, 1]]))

    # Omega = pi/6
    ct.x_hat = np.array([[1, 3, 5, 2, 4, 6, np.pi/6]]).T
    np.testing.assert_almost_equal(ct.f(.5), np.array([[1, 0, 0, .49, -.06,  0, 0],
                                                       [0, 1, 0, .06,  .49,  0, 0],
                                                       [0, 0, 1,   0,    0, .5, 0],
                                                       [0, 0, 0, .96, -.25,  0, 0],
                                                       [0, 0, 0, .25,  .96,  0, 0],
                                                       [0, 0, 0,   0,    0,  1, 0],
                                                       [0, 0, 0,   0,    0,  0, 1]]), 2)


# confirm that the covariance transition matrix is calculated correctly given the
# horizontal velocities and the turn rate
def test_ct3d_f_x():
    ct = CoordinatedTurn([1, 1, 1, .02])

    # Omega = 0
    ct.x_hat = np.array([[1, 3, 5, 2, 4, 6, 0]]).T
    np.testing.assert_almost_equal(ct.f_x(.5), np.array([[1, 0, 0, .5,  0,  0, -.5],
                                                         [0, 1, 0,  0, .5,  0, .25],
                                                         [0, 0, 1,  0,  0, .5,   0],
                                                         [0, 0, 0,  1,  0,  0,  -2],
                                                         [0, 0, 0,  0,  1,  0,   1],
                                                         [0, 0, 0,  0,  0,  1,   0],
                                                         [0, 0, 0,  0,  0,  0,   1]]))

    # Omega = pi/6
    ct.x_hat = np.array([[1, 3, 5, 2, 4, 6, np.pi/6]]).T
    np.testing.assert_almost_equal(ct.f_x(.5), np.array([[1, 0, 0, .49, -.06, 0, -.53 ],
                                                         [0, 1, 0, .06,  .49, 0,  .15 ],
                                                         [0, 0, 1,  0,    0, .5,   0  ],
                                                         [0, 0, 0, .96, -.25, 0, -2.19],
                                                         [0, 0, 0, .25,  .96, 0,  .44 ],
                                                         [0, 0, 0,  0,    0,  1,   0  ],
                                                         [0, 0, 0,  0,    0,  0,   1  ]]), 2)


# confirm that the process noise matrix is calculated correctly given the
# initial noise sigmas and delta time
def test_ct3d_q():
    ct = CoordinatedTurn([2, 3, 4, .02])

    vx = 4
    vy = 9
    vz = 16
    
    T  = 1.5
    T3 = 1.5 ** 3 / 2
    T4 = 1.5 ** 4 / 4

    np.testing.assert_almost_equal(ct.Q(1.5),
        np.array([[T4*vx, 0, 0, T3*vx, 0, 0,    0],
                  [0, T4*vy, 0, 0, T3*vy, 0,    0],
                  [0, 0, T4*vz, 0, 0, T3*vz,    0],
                  [T3*vx, 0, 0, T**2*vx, 0, 0,  0],
                  [0, T3*vy, 0, 0, T**2*vy, 0,  0],
                  [0, 0, T3*vz, 0, 0, T**2*vz,  0],
                  [0, 0, 0, 0, 0, 0, .0004 * T**2]]), 4)


# confirm that NEES remains within the predicted confidence interval
# certain fraction of the time
def test_ct3d_conf_int():
    target = SingleTurnTarget(30, 1)
    sensor = GeometricSensor(seed=0)
    true_positions = target.true_states()

    mc_runs = 200
    mc_nees_scores = []

    for _ in range(mc_runs):
        ct = CoordinatedTurn([1, 1, 1, .02289])
        t = 0
        ct.initialize(true_positions[t, :3], np.eye(3))

        x_hat, P_hat = [], []
        for _ in range(399):
            t += 1
            m = sensor.generate_measurement(t, true_positions[t, :3])

            ct.predict(1)
            ct.update(m.z, m.R)

            x_hat.append(np.copy(ct.x_hat))
            P_hat.append(np.copy(ct.P_hat))

        x_hat, P_hat = np.array(x_hat), np.array(P_hat)
        
        # select only positions
        x_hat = x_hat[:, :3, :]
        P_hat = P_hat[:, :3, :3]

        d = x_hat.squeeze() - true_positions[1:, :3]
        d = d.reshape(399, 3, 1)
        P_inv = np.linalg.inv(P_hat)

        mc_run_scores = np.matmul(np.matmul(d.transpose(0, 2, 1), P_inv), d).squeeze()
        mc_nees_scores.append(mc_run_scores)
    
    mc_nees_scores = np.array(mc_nees_scores).T
    mc_nees_scores = np.mean(mc_nees_scores, axis=1)

    state_dim = 3
    conf_int = sp.stats.chi2.ppf([0.025, 0.975], mc_runs * state_dim) / mc_runs

    within_conf_int = np.logical_and(conf_int[0] <= mc_nees_scores, mc_nees_scores <= conf_int[1])

    # skip the first 15 observations where errors might be higher due to incorrect
    # initialization of the filter; throughout the rest of the time, with a 95%
    # confidence interval, we expect close to 95% of observations to fall within
    # that interval; however, the Coordinated Turn filter is not linear and
    # overestimates the covariance, which results in fewer observations (~68%)
    # falling within the confidence interval
    np.testing.assert_almost_equal(np.mean(within_conf_int[15:]), .6796, 4)

    # this is illustrated next, where ~32% of the observations are under the lower
    # bound of the confidence interval, whereas we would expect 2.5% for a
    # statistically consistent filter; the fact that all ~32% of observations outside
    # of the confidence interval are in fact less than its lower bound shows that
    # the filter systematically overestimates the covariance (as opposed to over-
    # and underestimating it, in which case we would expect half of those ~32% to
    # be below and half above the confidence interval)
    too_low = np.mean(mc_nees_scores < conf_int[0])
    np.testing.assert_almost_equal(too_low, .3182, 4)
