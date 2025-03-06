import unittest
from sympy import symbols, integrate, exp
from utils.transform import transform

class TestTransform(unittest.TestCase):
    def setUp(self):
        self.exp_dist = transform(
            f=lambda x: exp(-x),
            var='x',
            interval=(0, float('inf'))
        )

    def test_generating_function(self):
        s = symbols('s')
        gf = self.exp_dist.generating_function(s)
        expected = integrate(s**x * exp(-x), (x, 0, float('inf')))
        self.assertEqual(str(gf), str(expected))

    def test_laplace_transform(self):
        s = symbols('s')
        lt = self.exp_dist.laplace_stieltjes_transform(s)
        expected = 1/(s + 1)
        self.assertEqual(str(lt.simplify()), str(expected.simplify()))

if __name__ == '__main__':
    unittest.main()