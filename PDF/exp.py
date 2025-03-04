from sympy import exp as sympy_exp
import numpy as np

from numeric import numeric


class Exp(numeric):
    def __init__(self, rate):
        """
        :param rate: rate of exponential distribution.
        """
        if not (isinstance(rate, float) | isinstance(rate, int)):
            raise TypeError("Rate must be int or float.")
        if rate <= 0:
            raise ValueError("Rate must be positive.")
        else:
            self.rate = float(rate)
            # 修改lambda使用sympy的exp
            self.f = lambda var: self.rate * sympy_exp(-self.rate * var)
            super().__init__(self.f, 'x', (0, np.inf))

    def expectation(self):
        """
        :return: The expectation of exponential distribution at rate.
        """
        return 1 / self.rate

    def variance(self):
        """
        :return: The variance of exponential distribution at rate.
        """
        return 1 / (self.rate ** 2)

    def generate(self, size: tuple[int, int], randomstate: int):
        """
        To generate the sequences of number to analysis.
        :param size: size of generation num default (1,1)
        :param randomstate: rand seed
        :return: size of sequences of number.
        """
        if isinstance(size, tuple):
            size = size
            if len(size) != 2:
                raise ValueError("Size must be tuple of length 2.")
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError("Size must be positive.")
        else:
            size = (1, 1)
        if randomstate:
            np.random.seed(randomstate)

        return np.random.exponential(1 / self.rate, size)

if __name__ == '__main__':
    exp = Exp(5)
    print(exp.expectation())
    print(exp.variance())
    print(exp.generate((10, 10), 1))