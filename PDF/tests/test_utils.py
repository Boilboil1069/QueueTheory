import unittest
from unittest.mock import patch
from sympy import symbols, integrate, exp
from PDF.utils import numeric, transform

class TestNumeric(unittest.TestCase):
    def setUp(self):
        self.uniform = numeric(
            f=lambda x: 0.5,
            var='x',
            interval=(0, 2)
        )

    def test_expectation(self):
        expected = 1.0
        self.assertAlmostEqual(self.uniform.expectation(), expected, delta=1e-6)

    def test_variance(self):
        expected = 1/3
        self.assertAlmostEqual(self.uniform.variance(), expected, delta=1e-6)

    def test_moment_z(self):
        moment = self.uniform.moment(2, 'z')
        expected = 4/3
        self.assertAlmostEqual(moment, expected, delta=1e-6)

    def test_moment_c(self):
        var = self.uniform.variance()
        moment = self.uniform.moment(2, 'c')
        self.assertAlmostEqual(moment, var, delta=1e-6)

    def test_invalid_moment_type(self):
        with self.assertRaises(ValueError):
            self.uniform.moment(2, 'invalid')

    @patch('builtins.print')
    def test_display_func(self, mock_print):
        self.uniform.display_func()
        mock_print.assert_called_with(self.uniform.f)

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
