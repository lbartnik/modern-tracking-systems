import numpy as np
from numpy.typing import ArrayLike


class SingerTarget:
    """Generates sample target trajectory using the Singer Acceleration Model.
    """

    def __init__(self, tau: float, sigma: float, T: float = 1):
        """Initialize trajectory generator.

        Args:
            tau (float): target maneuver time constant
            sigma (float): target maneuver standard deviation
            T (float, optional): Sampling time. Defaults to 1.
        """
        self.tau = tau
        self.sigma = sigma
        self.T = T

    def positions(self, time: ArrayLike, seed: float = 0) -> np.ndarray:
        rnd = np.random.default_rng(seed=seed)

        # sampling interval
        if len(np.unique(np.diff(time))) != 1:
            raise ValueError(f"Single acceleration model requires constant time step, got {np.diff(time)}")
        
        T = np.min(np.diff(time))

        # generate position, velocity and acceleration independently in each dimension
        dims = [_calculate_dimension(time, _acceleration(self.tau, self.sigma, T, len(time), rnd)) for i in range(3)]

        # reshuffle to keep positions, velocities, and accelerations grouped
        ret = [dims[0][:,0], dims[1][:,0], dims[2][:,0],
            dims[0][:,1], dims[1][:,1], dims[2][:,1],
            dims[0][:,2], dims[1][:,2], dims[2][:,2]]
        
        return (np.array(ret).T)[:,:3]


def _acceleration(tau, sigma, T, n=100, rnd=None):
    normal = np.random.normal if rnd is None else rnd.normal
    rho = np.exp(-T / tau)
    res = []
    
    a_k = 0
    for _ in range(n):
        a_k_1 = rho * a_k + np.sqrt(1 - rho*rho) * normal(0, sigma)
        res.append(a_k_1)
        a_k = a_k_1

    return np.array(res)


def _calculate_dimension(time, a):
    v_last = 0; v = []
    p_last = 0; p = []

    time = [0] + np.diff(time).tolist()
    
    for dt, a_last in zip(time, a):
        p_last += v_last * dt + a_last*dt*dt/2
        v_last += dt*a_last
        p.append(p_last)
        v.append(v_last)
    
    return np.array((np.array(p), np.array(v), a)).T


def sample_singer(tau: float, sigma: float, T: float, n=100, seed=None) -> np.ndarray:
    """Generate a sample trajectory using the Singer Acceleration Model.

    Args:
        tau (float): target maneuver time constant
        sigma (float): target maneuver standard deviation
        T (float, optional): Sampling time. Defaults to 1.
        n (int, optional): Number of samples. Defaults to 100.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        np.ndarray: Array of sampled positions, shape: (n, 3)
    """
    return SingerTarget(tau, sigma, T).positions(np.arange(n, step=T), seed)