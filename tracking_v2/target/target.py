import numpy as np


__all__ = ['Target']


class TargetIdGenerator(object):
    _next_measurement_id = 1

    @classmethod
    def generate_target_id(cls):
        cls._next_measurement_id += 1
        return cls._next_measurement_id
    
    @classmethod
    def reset(cls):
        cls._next_measurement_id = 1



class Target(object):
    target_id: int = TargetIdGenerator.generate_target_id()
    name: str
    spatial_dim: int
    seed: int
    rng: np.random.Generator
    T: float
    n: int
    cached_states: np.ndarray

    def cache(self, T: float = 1, n: int = 400):
        self.T = T
        self.n = n
        self.cached_time = np.arange(0, n) * T
        self.cached_states = self.true_states(T, n)

    def reset_seed(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def reset_rng(self, rng: np.random.Generator = None):
        self.seed = None
        self.rng = rng

    def __repr__(self):
        if self.seed is None:
            return self.name
        else:
            return f"{self.name}_{self.seed}"

    def true_state(self, t: float) -> np.ndarray:
        assert self.cached_states is not None, "States are not present in cache"
        
        index = self.cached_time == t
        if np.any(index):
            return self.cached_states[index]

        state = []    
        for i in range(self.cached_states.shape[-1]):
            interpolated = np.interp(t, self.cached_time, self.cached_states[:, i])
            state.append(interpolated)
        
        return np.asarray(state)
        

    def true_states(self, T: float = 1, n: int = 400) -> np.ndarray:
        """Generate target states.

        Args:
            T (Union[float, ArrayLike]): Sampling interval or array of specific timestamps.
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Raises:
            Exception: If not implemented by the subclass.

        Returns:
            np.ndarray: (n, 6) array of states.
        """
        raise Exception(f"Target {self.__class__.__name__} does not implement true_states()")
