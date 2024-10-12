class MotionModel:
    state_dim: int
    name: str

    def F(self, dt: float):
        raise Exception(f"{self.__class__.__name__} does not implement F()")

    def Q(self, dt: float):
        raise Exception(f"{self.__class__.__name__} does not implement Q()")