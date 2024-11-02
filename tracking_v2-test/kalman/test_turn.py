import numpy as np
import scipy as sp

from tracking_v2.kalman.turn import CoordinatedTurn
from tracking_v2.sensor import GeometricSensor
from tracking_v2.target import SingleTurnTarget


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
    
    inside_conf_int = np.logical_and(conf_int[0] <= mc_nees_scores, mc_nees_scores <= conf_int[1]).mean()
    np.testing.assert_almost_equal(inside_conf_int, 0.79699, 5)
