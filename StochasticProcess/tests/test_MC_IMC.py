import unittest
from unittest.mock import patch
import numpy as np
from StochasticProcess.MarkovChain import MarkovChain, InfiniteMarkovChain

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        self.valid_P = np.array([[0.9, 0.1], [0.5, 0.5]])
        self.states = ['A', 'B']

    def test_init_validation(self):
        # 有效矩阵测试
        mc = MarkovChain(self.valid_P, self.states)
        self.assertIsInstance(mc, MarkovChain)

        # 无效矩阵测试
        with self.assertRaises(ValueError):
            MarkovChain(np.array([[0.8, 0.3], [0.5, 0.5]]), self.states)  # 行和不为1

    @patch('numpy.random.choice')
    def test_state_transition(self, mock_choice):
        mock_choice.return_value = 1  # 强制跳转到状态B
        mc = MarkovChain(self.valid_P, self.states)
        seq = mc.generate_sequence(2, 'A')
        self.assertEqual(seq, ['A', 'B', 'B'])

    def test_stationary_distribution(self):
        # 已知稳态测试
        mc = MarkovChain(np.array([[0.5, 0.5], [0.5, 0.5]]), ['A', 'B'])
        pi = mc.stationary_distribution(method='power')
        self.assertAlmostEqual(pi['A'], 0.5, delta=0.01)

class TestInfiniteMarkovChain(unittest.TestCase):
    def test_kernel_validation(self):
        # 合规核函数测试
        def valid_kernel(x):
            return lambda y: 1.0  # 均匀分布
        chain = InfiniteMarkovChain(valid_kernel)
        chain.kernel_check(0.5)  # 不应抛出异常

        # 错误核函数测试
        def invalid_kernel(x):
            return lambda y: 2.0  # 积分=2
        with self.assertRaises(ValueError):
            InfiniteMarkovChain(invalid_kernel).kernel_check(0.5)

    @patch('numpy.random.random')
    def test_sequence_generation(self, mock_random):
        mock_random.return_value = 0.7
        chain = InfiniteMarkovChain(lambda x: lambda y: 1.0)
        seq = chain.generate_sequence(3, 0.5)
        self.assertEqual(len(seq), 4)  # 初始状态+3步
        self.assertEqual(seq[1], 0.7)  # mock验证

if __name__ == '__main__':
    unittest.main()
