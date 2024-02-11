import numpy as np
import pandas as pd


def evaluate(target, kf, time=np.arange(0, 100), z_sigma=.1, seed=0):
    # calculate target positions over time
    time = np.array(time)
    positions = target.positions(time, seed=0)
    
    # transform positions into measurements
    z_var = z_sigma*z_sigma
    meas = _cartesian_measurements(positions, np.diag([z_var, z_var, z_var]))
    
    # initialize track state
    mean = np.full(6, 0)
    mean[:3] = meas[0,:]
    cov = np.diag(np.full(6, z_var))
    
    kf.initialize(mean, cov)

    # iterate
    err = []
    pos = []
    vel = []
    
    for dt, z, true_pos in zip(np.diff(time), meas, positions):
        kf.predict(dt)
        pos.append(kf.x[:3])
        vel.append(kf.x[3:6])
        err.append(kf.x[:3] - true_pos)
        kf.update(z)
    
    return positions, np.array(pos), np.array(vel), np.array(err)


def _cartesian_measurements(positions, noise_covariance):
    noise_mean = np.full(positions.shape[1], 0)
    noise = np.random.multivariate_normal(noise_mean, noise_covariance, size=positions.shape[0])
    return positions + noise


def monte_carlo(target, kf, time=np.arange(0, 100), z_sigma=.1, seeds=range(50)):
    if not isinstance(z_sigma, list):
        z_sigma = [z_sigma]

    res = []
    for s in z_sigma:
        for i, seed in enumerate(seeds):
            _, _, _, err = evaluate(target, kf, time, s, seed)
            res.append((s, i, seed, rmse(err)))
    
    return pd.DataFrame(res, columns=['z_sigma', 'no', 'seed', 'rmse'])


def rmse(err):
    return np.sqrt(np.power(err, 2).sum(axis=1).mean(axis=0))