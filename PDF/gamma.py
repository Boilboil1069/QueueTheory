import numpy as np
from sympy import exp as sympy_exp, gamma as sympy_gamma

from utils import numeric

class Gamma(numeric):
    def __init__(self,shape,scale):
        """
        :param shape:
        :param scale:
        """
        if shape<=0 or scale<=0:
            raise ValueError('shape and scale must be positive')

        self.shape = float(shape)
        self.scale = float(scale)
        self.f = lambda var: (self.scale ** self.shape) * var ** (self.shape - 1) * sympy_exp(-self.scale * var) / sympy_gamma(self.shape)
        super().__init__(self.f, 'x', (0, np.inf))

    def expectation(self):
        """
        :return:The expectation of gamma distribution.
        """
        return self.shape*self.scale

    def variance(self):
        """
        :return: The variance of gamma distribution.
        """
        return self.shape*(self.scale**2)

    def generate(self,size: tuple[int,int],randomstate:int):
        """
        :param size:
        :param randomstate:
        :return:
        """
        if randomstate:
            np.random.seed(randomstate)

        return np.random.gamma(self.shape,self.scale,size=size)

if __name__ == '__main__':
    g = Gamma(2,3)
    print(g.expectation())
    print(g.variance())
    print(g.generate((2,2),randomstate=1))