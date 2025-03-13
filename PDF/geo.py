import random
from utils import numeric


class geo(numeric):
    def __init__(self, p: float, var: str, interval: tuple):
        """
        Function: ((1-p)^(i-1))*p
        :param p: The probability of success
        :param var: The variable of integration
        :param interval: The upper and lower bounds of integration
        """
        if not (isinstance(p, float) | isinstance(p, int)):
            raise TypeError("Probability p must be int or float.")
        if p <= 0 or p > 1:
            raise ValueError("Probability p must be in the range (0, 1].")
        else:
            self.p = float(p)
            def geometric_pdf(i):
                return (1 - self.p) ** (i - 1) * self.p

            super().__init__(geometric_pdf, var, interval)

    def expectation(self):
        """
        :return: The expectation of geometric distribution.
        """
        return 1 / self.p

    def variance(self):
        """
        :return: The variance of geometric distribution.
        """
        return (1 - self.p) / (self.p ** 2)

    def generate(self, size: tuple[int, int], randomstate: int):
        """
        To generate the sequences of numbers following the geometric distribution.
        :param size: size of generation num, default (1,1)
        :param randomstate: rand seed
        :return: size of sequences of numbers following the geometric distribution.
        """
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError("Size must be tuple of length 2.")
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError("Size must be positive.")
        else:
            size = (1, 1)

        if randomstate:
            random.seed(randomstate)

        return [[self._generate_random() for _ in range(size[1])] for _ in range(size[0])]

    def _generate_random(self) -> int:
        """
        Helper method to generate a single random number following the geometric distribution.
        :return: A random number representing the number of trials until the first success.
        """
        count = 1
        while random.random() > self.p:
            count += 1
        return count

