import numpy as np

from tracking_v2.kalman import LinearKalmanFilter
from tracking_v2.module import SingleTargetTracker
from tracking_v2.motion import ConstantVelocityModel
from tracking_v2.sensor import GeometricSensor
from tracking_v2.target import ConstantVelocityTarget


def test_nees():
    target = ConstantVelocityTarget()
    monte_carlo_n = 100

    monte_carlo_x, monte_carlo_P = [], []
    for i in range(monte_carlo_n):
        tracker = SingleTargetTracker(LinearKalmanFilter(ConstantVelocityModel(noise_intensity=0),
                                                         [[1, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0]]))
        
        sensor = GeometricSensor(seed=i)

        for t, position in enumerate(target.true_states()[:, :3]):
            tracker.update(sensor.generate_measurement(t, position))

        _, x_hat, P_hat = list(zip(*tracker.track_updates))

        monte_carlo_x.append(np.array(x_hat))
        monte_carlo_P.append(np.array(P_hat))
    
    monte_carlo_x, monte_carlo_P = np.array(monte_carlo_x).squeeze(), np.array(monte_carlo_P)

    d = monte_carlo_x - target.true_states()
    d = d.reshape(monte_carlo_n, 400, 6, 1)
    P_inv = np.linalg.inv(monte_carlo_P)

    mc_nees_scores = np.matmul(np.matmul(d.transpose(0, 1, 3, 2), P_inv), d).squeeze()
    average_nees_scores = np.sum(mc_nees_scores, axis=0) / monte_carlo_n

    # state_dim, monte_carlo_n = 6, 100
    # scipy.stats.chi2.ppf([0.025, 0.975], monte_carlo_n * state_dim) / monte_carlo_n
    # array([5.3401855 , 6.69769152])
    critical_region_lower, critical_region_upper = [5.3401855 , 6.69769152]

    assert np.mean(average_nees_scores[75:] < critical_region_lower) <= 0.025
    assert np.mean(average_nees_scores[75:] > critical_region_upper) <= 0.025
