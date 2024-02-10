import numpy as np


def evaluate(target, kf, time=np.arange(0, 100), z_sigma=.1):
    # calculate target positions over time
    time = np.array(time)
    positions = target.positions(time)
    
    # transform positions into measurements
    z_var = z_sigma*z_sigma
    meas = cartesian_measurements(positions, np.diag([z_var, z_var, z_var]))
    
    # initialize track state
    mean = np.full(6, 0)
    mean[:3] = meas[0,:]
    cov = np.diag(np.full(6, z_var))
    
    kf.initialize(mean, cov)

    # iterate
    err = []
    pos = []
    vel = []
    
    for dt, z in zip(np.diff(time), meas):
        kf.predict(dt)
        pos.append(kf.x[:3])
        vel.append(kf.x[3:6])
        err.append(kf.x[:3] - z)
        kf.update(z)
    
    return positions, np.array(pos), np.array(vel), np.array(err)


def cartesian_measurements(positions, noise_covariance):
    noise_mean = np.full(positions.shape[1], 0)
    noise = np.random.multivariate_normal(noise_mean, noise_covariance, size=positions.shape[0])
    return positions + noise
