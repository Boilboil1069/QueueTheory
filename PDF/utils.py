from sympy import symbols, integrate, exp


class numeric:
    def __init__(self, f: callable, var, interval: tuple):
        """
        :param f:  Probability density function.
        :param var: The variable of integration
        :param interval: The upper and lower of integration
        """
        self.var = symbols(var)
        self.f = f(self.var)
        self.interval_upper = interval[1]
        self.interval_lower = interval[0]
    def display_func(self):
        """
        :return: Display the distribution function.
        """
        print(self.f)
    def expectation(self):
        """
        :return: The expectation of f
        """
        return integrate(self.var * self.f,(self.var,self.interval_lower, self.interval_upper))

    def variance(self):
        """
        :return: The variance of f
        """
        return integrate(((self.var - self.expectation())** 2) * self.f,(self.var,self.interval_lower, self.interval_upper))

    def moment(self, n: int, m: str = 'c'):
        """
        :param n: The order of moment
        :param m: The type of moment, z for zero-moment, c for central moment
        :return: The n-th moment of f
        """
        if m == 'z':
            return integrate((self.var ** n) * self.f,(self.var,self.interval_lower, self.interval_upper))
        elif m == 'c':
            return integrate(((self.var - self.expectation()) ** n) * self.f, (self.var, self.interval_lower, self.interval_upper))
        else:
            raise ValueError('m must be z or c')
        
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


if __name__ == '__main__':
    x = symbols('x')
    f_expr = x ** 2
    interval = (0, 1)
    numeric_test = numeric(lambda var: var ** 2, 'x', interval)
    print(numeric_test.expectation())
    print(numeric_test.variance())
    print(numeric_test.moment(2, 'z'))
    print(numeric_test.moment(2, 'c'))

    transform_test = transform(lambda var: var ** 2, 'x', (0, 1))
    s = symbols('s')
    print("\nTransform Test:")
    print("Generating function:", transform_test.generating_function(s))
    print("LST:", transform_test.laplace_stieltjes_transform(s))