import numpy as np

from scipy.spatial.transform import Rotation

from numpy.typing import ArrayLike
from typing import Union

class Transform(object):
    def __init__(self, translation: ArrayLike, rotation: Rotation):
        self.translation = np.asarray(translation)
        self.rotation = rotation

    def __repr__(self) -> str:
        r, p, y = self.rotation.as_euler('xyz', degrees=True)
        return f"Transform(translation={self.translation.tolist()}, rotation=({y:.2f}, {p:.2f}, {r:.2f}))"
    
    def transform_vec(self, vec: ArrayLike) -> np.ndarray:
        vec = np.asarray(vec).flatten()

        if len(vec) == 3:
            return self.rotation.apply(vec) + self.translation
        elif len(vec) == 6:
            ans = np.zeros(6)
            ans[:3] = self.rotation.apply(vec[:3]) + self.translation
            ans[3:] = self.rotation.apply(vec[3:])
            return ans
        else:
            raise Exception(f"Unsupported vector shape {vec.shape}")
    
    def transform_cov(self, cov: ArrayLike) -> np.ndarray:
        cov = np.asarray(cov)
        r = self.rotation.as_matrix()
        
        if cov.shape == (3, 3):
            return r @ cov @ r.T
        elif cov.shape == (6, 6):
            ans = np.zeros_like(cov)
            ans[:3, :3] = r @ cov[:3, :3] @ r.T
            ans[3:, :3] = r @ cov[3:, :3] @ r.T
            ans[:3, 3:] = r @ cov[:3, 3:] @ r.T
            ans[3:, 3:] = r @ cov[3:, 3:] @ r.T
            return ans
        else:
            raise Exception(f"Unsupported covariance shape {cov.shape}")



class Frame(object):
    """Global frame is one with origin (0,0,0) and unity rotation. Local frame
    is anything else."""
    
    # Choose right vector (arbitrary but stable)
    # use world +Y as up, so right = up × forward
    UP = np.asarray([0, 0, 1])

    # base frame
    GlobalFrame = None

    def __init__(self, origin: ArrayLike, orientation: Union[ArrayLike, Rotation]):
        self.origin = np.asarray(origin)

        # active rotation from XYZ to frame
        if isinstance(orientation, Rotation):
            self.rotation = orientation
        else:
            self.rotation = Frame.to_rotation(orientation)

    def __repr__(self) -> str:
        r, p, y = self.rotation.as_euler('xyz', degrees=True)
        return f"Frame(origin={self.origin.tolist()}, rotation=({y:.2f}, {p:.2f}, {r:.2f}))"

    def frame_from_local_direction(self, local_direction: ArrayLike) -> 'Frame':
        """Global frame of the direction relative to this frame"""
        local_rotation = Frame.to_rotation(local_direction)
        return Frame(self.origin, self.rotation * local_rotation)

    def transform(self, to: 'Frame') -> Transform:
        """Calculate a Transform from this frame to the `to` frame."""
        r = to.rotation.inv() * self.rotation
        o = to.rotation.apply(self.origin - to.origin, inverse=True)
        return Transform(o, r)

    def to_global_vec(self, local_vec: ArrayLike) -> np.ndarray:
        return self.transform(Frame.GlobalFrame).transform_vec(local_vec)

    def to_local_vec(self, global_vec: ArrayLike) -> np.ndarray:
        return Frame.GlobalFrame.transform(self).transform_vec(global_vec)

    def to_global_cov(self, local_cov: ArrayLike) -> np.ndarray:
        return self.transform(Frame.GlobalFrame).transform_cov(local_cov)
    
    @staticmethod
    def to_rotation(vec: ArrayLike) -> Rotation:
        """
        Derives a quaternion for a coordinate system: 
        Forward=[1,0,0], Left=[0,1,0], Up=[0,0,1].
        The result will have zero roll relative to the XY plane.
        """
        vec = np.asarray(vec)
        
        # 1. Normalize the target forward vector (Local X)
        forward = vec / np.linalg.norm(vec)
        
        # 2. Derive the Local Left vector (Local Y)
        # Crossing Global Up with Forward gives a vector on the XY plane
        left = np.cross(Frame.UP, forward)
        
        # Handle singularity: if target is straight up/down, cross product is zero
        if np.linalg.norm(left) < 1e-6:
            # Default to global Left [0, 1, 0] to resolve ambiguity
            left = np.array([0, 1, 0])
        else:
            left /= np.linalg.norm(left)
        
        # 3. Derive the Re-orthogonalized Local Up vector (Local Z)
        up = np.cross(forward, left)
        
        # 4. Construct Rotation Matrix
        # Columns correspond to Local X (Forward), Local Y (Left), and Local Z (Up)
        matrix = np.column_stack((forward, left, up))
    
        # 5. Convert to Quaternion [w, x, y, z]
        return Rotation.from_matrix(matrix)

Frame.GlobalFrame = Frame([0, 0, 0], [1, 0, 0])
