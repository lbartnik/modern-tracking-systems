import numpy as np
from numpy.typing import ArrayLike


def nees(truth: ArrayLike, x_hat: ArrayLike, P_hat: ArrayLike):
    truth, x_hat, P_hat = np.array(truth), np.array(x_hat), np.array(P_hat)

    assert len(truth.shape) == 2
    assert len(x_hat.shape) == 2
    assert len(P_hat.shape) == 3

    assert truth.shape == x_hat.shape
    assert P_hat.shape[0] == x_hat.shape[0]
    assert P_hat.shape[1] == P_hat.shape[2]
    assert P_hat.shape[1] == x_hat.shape[1]

    d = (x_hat-truth).reshape(-1, x_hat.shape[1], 1)
    P_inv = np.linalg.inv(P_hat)
    
    return np.matmul(np.matmul(d.transpose(0, 2, 1), P_inv), d).squeeze()
