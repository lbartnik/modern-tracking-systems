import numpy as np
from numpy.typing import ArrayLike


__all__ = ['RbeSpace', 'CartesianPositionVelocity']


class RbeSpace(object):
    def __init__(self):
        self.name = "RBE"
        self.spatial_dim = 3

    def from_cartesian(self, a: ArrayLike) -> ArrayLike:
        """Transform an array of Cartesian positions to RBE, with sensor located in (0,0,0).

        Args:
            a (ArrayLike): Array of Cartesian positions.

        Returns:
            ArrayLike: Array of RBE positions.
        """
        a = np.array(a)
        if len(a.shape) == 1:
            a = np.expand_dims(a, 0)

        assert a.shape[1] == self.spatial_dim
        
        sq = np.power(a, 2)
        r  = np.sqrt(sq.sum(axis=1))
        rh = np.sqrt(sq[:, :2].sum(axis=1))

        az = np.arctan2(a[:,1], a[:,0])
        el = np.arctan2(a[:,2], rh)

        rbe = np.array((r, az, el)).T
        return rbe

    def to_cartesian(self, a: ArrayLike) -> ArrayLike:
        """Transform an array of RBE positions to Cartesian, with sensor located in (0,0,0).

        Args:
            a (ArrayLike): Array of RBE positions.

        Returns:
            ArrayLike: Array of Cartesian positions.
        """
        a = np.array(a)
        if len(a.shape) == 1:
            a = np.expand_dims(a, 0)
        
        assert a.shape[1] == self.spatial_dim

        r, b, e = a[:,0], a[:,1], a[:,2]
        z = r * np.sin(e)
        x2y2 = r * np.cos(e)
        x = x2y2 * np.cos(b)
        y = x2y2 * np.sin(b)
        
        return np.array((x, y, z)).T

    def jacobian(self, x0: ArrayLike) -> ArrayLike:
        """Calculates the Jacobian matrix of the inverse of the measurement function, in
        point x0.

        Args:
            x0 (ArrayLike): Cartesian position for which to evaluate the Jacobian.
            state_dim (int): Optional dimensionality of KF state.

        Returns:
            ArrayLike: Jacobian evaluated in x0
        """
        x0 = np.array(x0).squeeze()
        assert x0.shape == (self.spatial_dim,)

        x, y, z = x0

        x2y2 = x**2 + y**2
        x2y2z2 = x2y2 + z**2
        
        r = np.sqrt(x2y2z2)
        t = np.sqrt(x2y2)
        u = t * x2y2z2

        # range, bearing, elevation
        return np.array([
            [x/r, y/r, z/r],
            [-y/x2y2, x/x2y2, 0],
            [-x*z/u, -y*z/u, t/x2y2z2]
        ])


class CartesianPositionVelocity(object):
    def __init__(self):
        self.name = "CartesianPositionVelocity"
        self.spatial_dim = 3
        self.state_dim = 6
