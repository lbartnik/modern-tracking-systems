import numpy as np


class SinusTarget:
    def __init__(self, speed=20, range=10):
        self.speed = speed
        self.range = range

    def positions(self, t, seed=0):
        t = np.array(t)
        x = np.sin(t / 180 * np.pi * self.speed) * self.range
        y = np.full_like(t, 0)
        z = np.arange(0, len(t))
        return np.array((x, y, z)).T
