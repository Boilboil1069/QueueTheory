import unittest
from unittest.mock import patch
import numpy as np
from PDF.gamma import Gamma  # 替换为实际模块路径

class TestGamma(unittest.TestCase):

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            Gamma(-1, 2)
        with self.assertRaises(ValueError):
            Gamma(2, -1)

        valid_gamma = Gamma(2, 3)
        self.assertEqual(valid_gamma.shape, 2.0)
        self.assertEqual(valid_gamma.scale, 3.0)

    def test_expectation(self):
        test_cases = [
            (2, 3, 6),
            (5, 0.5, 2.5)
        ]
        for shape, scale, expected in test_cases:
            with self.subTest(shape=shape, scale=scale):
                gamma = Gamma(shape, scale)
                self.assertAlmostEqual(gamma.expectation(), expected)

    def test_variance(self):
        gamma = Gamma(2, 3)
        self.assertAlmostEqual(gamma.variance(), 18.0)
        gamma = Gamma(5, 0.5)
        self.assertAlmostEqual(gamma.variance(), 1.25)

    @patch('numpy.random.gamma')
    def test_generate_mock(self, mock_gamma):
        """验证调用参数正确性"""
        gamma = Gamma(2, 3)
        test_size = (2, 2)
        gamma.generate(test_size, randomstate=42)
        mock_gamma.assert_called_with(2, 3, size=test_size)

    def test_generate_consistency(self):
        """验证随机种子效果"""
        gamma = Gamma(2, 3)
        result1 = gamma.generate((2,2), randomstate=42)
        result2 = gamma.generate((2,2), randomstate=42)
        np.testing.assert_array_equal(result1, result2)

    def test_generate_dimension(self):
        """验证输出维度"""
        gamma = Gamma(2, 3)
        output = gamma.generate((3,4), randomstate=None)
        self.assertEqual(output.shape, (3,4))

if __name__ == '__main__':
    unittest.main()
