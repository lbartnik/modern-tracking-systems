import numpy as np
from numpy.typing import ArrayLike


class SingerAccelerationModel:
    """Singer target maneuver model takes the target states to be position, velocity,
    and acceleration and assumes the target acceleration to be a first-order Markov
    process.
    """

    def __init__(self, tau: float, sigma: float):
        """Construct a Single acceleration model.

        Args:
            tau (float): target maneuver time constant
            sigma (float): target maneuver standard deviation
        """
        self.tau = tau
        self.sigma = sigma
        self.state_dim = 9
        self.spatial_dim = 3
    
    @property
    def name(self):
        """Motion model name"""
        return f"singer_{self.tau}_{self.sigma}"
        
    def F(self, dt):
        beta = 1/self.tau
        BT   = beta * dt
        rho  = np.exp(-BT)

        f_1 = 1/beta**2 * (-1 + BT + rho)
        f_2 = 1/beta * (1 - rho)
        f_3 = rho

        return np.array([[1, 0, 0, dt, 0, 0, f_1, 0, 0],
                         [0, 1, 0, 0, dt, 0, 0, f_1, 0],
                         [0, 0, 1, 0, 0, dt, 0, 0, f_1],
                         [0, 0, 0, 1, 0, 0, f_2, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, f_2, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, f_2],
                         [0, 0, 0, 0, 0, 0, f_3, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, f_3, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, f_3]])
        
    def Q(self, dt):
        beta = 1/self.tau
        BT   = beta * dt

        e_BT  = np.exp(-BT)
        e_2BT = np.exp(-2*BT)
        BT2   = BT**2
        BT3   = BT**3

        q_11 = .5/beta**5 * (1 - e_2BT + 2*BT + 2/3 * BT3 - 2*BT2 - 4*BT*e_BT)
        q_12 = .5/beta**4 * (e_2BT + 1 - 2*e_BT + 2*BT*e_BT - 2*BT + BT2)
        q_13 = .5/beta**3 * (1 - e_2BT - 2*BT*e_BT)
        q_22 = .5/beta**3 * (4*e_BT - 3 - e_2BT + 2*BT)
        q_23 = .5/beta**2 * (e_2BT + 1 - 2*e_BT)
        q_33 = .5/beta * (1 - e_2BT)

        q_21 = q_12
        q_31 = q_13
        q_32 = q_23

        return np.array([[q_11, 0, 0, q_12, 0, 0, q_13, 0, 0],
                         [0, q_11, 0, 0, q_12, 0, 0, q_13, 0],
                         [0, 0, q_11, 0, 0, q_12, 0, 0, q_13],
                         [q_21, 0, 0, q_22, 0, 0, q_23, 0, 0],
                         [0, q_21, 0, 0, q_22, 0, 0, q_23, 0],
                         [0, 0, q_21, 0, 0, q_22, 0, 0, q_23],
                         [q_31, 0, 0, q_32, 0, 0, q_33, 0, 0],
                         [0, q_31, 0, 0, q_32, 0, 0, q_33, 0],
                         [0, 0, q_31, 0, 0, q_32, 0, 0, q_33]]) \
                * 2 * self.sigma**2 / self.tau


def singer_acceleration_models(*args):
    return [SingerAccelerationModel(tau, sigma) for tau, sigma in args]
