from sympy import integrate, exp

from . import numeric


class transform(numeric):
    def __init__(self, f: callable, var, interval: tuple):
        """
        :param f:  Probability density function.
        :param var: The variable of integration
        :param interval: The upper and lower of integration
        """
        super().__init__(f, var, interval)

    def generating_function(self, s):
        """
        Generating function: G(s) = E[s^X]
        :param s: Symbol variable
        :return: Expr
        """
        return integrate(s ** self.var * self.f,
                         (self.var, self.interval_lower, self.interval_upper))

    def laplace_stieltjes_transform(self, s):
        """
        Laplace-Stieltjes Transform L(s) = E[e^{-sX}]
        :param s: Symbol variable
        :return: Expr
        """
        return integrate(exp(-s * self.var) * self.f, (self.var, self.interval_lower, self.interval_upper))
