import numpy as np

from tracking_v2.evaluation import evaluate_nees


def test_nees():
    truth = [[1, 1, 1], [2, 2, 2]]
    x_hat = [[2, 3, 4], [4, 5, 6]]
    P_hat = [np.diag([1, 4, 9]), np.diag([4, 9, 16])]
    
    score = evaluate_nees(truth, x_hat, P_hat)
    np.array_equal(score, [1, 1])


def test_nees_multiple_runs():
    truth = [[1, 1, 1], [2, 2, 2]]
    x_hat = [[[2, 3, 4], [4, 5, 6]],
             [[3, 4, 5], [5, 6, 7]]]
    P_hat = [[np.diag([1, 4, 9]), np.diag([4, 9, 16])],
             [np.diag([4, 9, 16])/np.sqrt(2), np.diag([9, 16, 25])/np.sqrt(2)]]
    
    score = evaluate_nees(truth, x_hat, P_hat)
    np.array_equal(score, [[1, 1], [2, 2]])

