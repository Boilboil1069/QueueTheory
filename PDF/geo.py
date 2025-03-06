from utils import numeric


class geo(numeric):
    def __init__(self, rate, f: callable, var, interval: tuple):
        super().__init__(f, var, interval)
