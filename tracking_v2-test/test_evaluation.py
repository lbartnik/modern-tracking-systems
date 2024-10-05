import numpy as np

from tracking_v2.evaluation import nees, validate_kalman_filter
from tracking_v2.kalman import KalmanFilter
from tracking_v2.module import SingleTargetTracker
from tracking_v2.motion import ConstantVelocityModel
from tracking_v2.sensor import GeometricSensor
from tracking_v2.target import ConstantVelocityTarget


def test_nees():
    truth = [[1, 1, 1], [2, 2, 2]]
    x_hat = [[2, 3, 4], [4, 5, 6]]
    P_hat = [np.diag([1, 4, 9]), np.diag([4, 9, 16])]
    
    score = nees(truth, x_hat, P_hat)
    np.array_equal(score, [1, 1])


def test_nees_multiple_runs():
    truth = [[1, 1, 1], [2, 2, 2]]
    x_hat = [[[2, 3, 4], [4, 5, 6]],
             [[3, 4, 5], [5, 6, 7]]]
    P_hat = [[np.diag([1, 4, 9]), np.diag([4, 9, 16])],
             [np.diag([4, 9, 16])/np.sqrt(2), np.diag([9, 16, 25])/np.sqrt(2)]]
    
    score = nees(truth, x_hat, P_hat)
    np.array_equal(score, [[1, 1], [2, 2]])



def test_validate_kalman_filter():
    target = ConstantVelocityTarget()

    trackers = []
    for i in range(100):
        tracker = SingleTargetTracker(KalmanFilter(ConstantVelocityModel(noise_intensity=0),
                                                   [[1, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0]]))
        sensor = GeometricSensor(seed=i)

        for t, position in enumerate(target.true_states()[:, :3]):
            tracker.update(sensor.generate_measurement(t, position))
        
        trackers.append(tracker)
    
    report = validate_kalman_filter(trackers, target)
    assert report.is_valid()
