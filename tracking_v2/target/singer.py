import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Optional, Union

from .target import Target


__all__ = ['SingerTarget']


class SingerTarget(Target):
    """Generates sample target trajectory using the Singer Acceleration Model.
    """

    def __init__(self, tau: float, sigma: float, T: float = 1, seed: Optional[float] = None):
        """Initialize trajectory generator.

        Args:
            tau (float): target maneuver time constant
            sigma (float): target maneuver standard deviation
            T (float, optional): Sampling time. Defaults to 1.
        """        
        self.tau = tau
        self.sigma = sigma
        self.T = T
        self.seed = seed

        if self.seed is None:
            self.name = f"singer_{self.tau}_{self.sigma}"
        else:
            self.name = f"singer_{self.tau}_{self.sigma}_{self.seed}"


    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: float = 0) -> np.ndarray:
        if self.seed is not None:
            seed = self.seed
        
        if isinstance(T, (int, float)):
            time = np.arange(0, n, T)
        else:
            T = 1 # TODO better take the most frequent value in np.diff(time)
            time = np.array(T)
        
        rnd = np.random.default_rng(seed=seed)

        # generate position, velocity and acceleration independently in each dimension
        dims = [_calculate_one_dimension(time, _acceleration(self.tau, self.sigma, T, len(time), rnd)) for _ in range(3)]

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


def _calculate_one_dimension(time, a):
    v_last = 0; v = []
    p_last = 0; p = []

    time = [0] + np.diff(time).tolist()
    
    for dt, a_last in zip(time, a):
        p_last += v_last * dt + a_last*dt*dt/2
        v_last += dt*a_last
        p.append(p_last)
        v.append(v_last)
    
    return np.array((np.array(p), np.array(v), a)).T
