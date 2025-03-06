import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sympy import exp as sympy_exp

# 被测模块需根据实际路径调整
from PDF.exp import Exp

class TestExp(unittest.TestCase):
    """Test cases for Exp class"""

    @patch('your_module.numeric')
    def test_init_validation(self, mock_numeric):
        """测试初始化参数校验 | Test __init__ validation"""
        # Type error case
        with self.assertRaises(TypeError):
            Exp("invalid_rate")

        # Value error case
        with self.assertRaises(ValueError):
            Exp(-5)

        # Valid case
        obj = Exp(5)
        self.assertEqual(obj.rate, 5.0)
        self.assertEqual(obj.f(0), 5 * sympy_exp(0))

    def test_expectation_variance(self):
        """测试期望/方差计算 | Test expectation/variance calculation"""
        exp = Exp(5)
        self.assertAlmostEqual(exp.expectation(), 0.2)
        self.assertAlmostEqual(exp.variance(), 0.04)

    @patch('numpy.random.exponential')
    @patch('numpy.random.seed')
    def test_generate(self, mock_seed, mock_exp):
        """测试随机数生成 | Test random generation"""
        exp = Exp(5)

        # Case 1: Invalid size
        with self.assertRaises(ValueError):
            exp.generate("invalid_size", 1)

        # Case 2: Valid generation
        mock_exp.return_value = np.ones((10,10))
        result = exp.generate((10,10), 1)
        mock_exp.assert_called_with(0.2, (10,10))
        self.assertEqual(result.shape, (10,10))

        # Case 3: Random seed setting
        exp.generate((2,2), 42)
        mock_seed.assert_called_with(42)

if __name__ == '__main__':
    unittest.main()
