import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from StochasticProcess import SemiMarkovProcess

class TestSemiMarkovProcess(unittest.TestCase):
    def setUp(self):
        """通用测试配置"""
        self.valid_states = ['A', 'B']
        self.valid_trans = {
            'A': {'A': 0.3, 'B': 0.7},
            'B': {'A': 1.0}
        }
        self.valid_holding = {
            'A': lambda: 2.0,
            'B': lambda: 1.5
        }

    # region 初始化验证测试
    def test_valid_initialization(self):
        """测试有效输入初始化"""
        smp = SemiMarkovProcess(self.valid_states,
                              self.valid_trans,
                              self.valid_holding)
        self.assertEqual(smp.states, self.valid_states)

    def test_missing_transition_state(self):
        """测试缺失转移状态异常"""
        invalid_trans = {'A': self.valid_trans['A']}
        with self.assertRaisesRegex(ValueError, "Missing transition probabilities"):
            SemiMarkovProcess(self.valid_states, invalid_trans, self.valid_holding)

    def test_invalid_holding_dist(self):
        """测试无效停留时间分布"""
        invalid_holding = {'A': lambda: -1, 'B': lambda: 1}
        with self.assertRaisesRegex(ValueError, "Invalid holding time distribution"):
            SemiMarkovProcess(self.valid_states, self.valid_trans, invalid_holding)
    # endregion

    # region 轨迹生成测试
    @patch('numpy.random.choice')
    def test_trajectory_generation(self, mock_choice):
        """测试多步轨迹生成"""
        mock_choice.side_effect = ['B', 'A']  # 控制状态转移路径
        holding_mock = {
            'A': lambda: 1.0,
            'B': lambda: 1.0
        }
        smp = SemiMarkovProcess(['A', 'B'], self.valid_trans, holding_mock)

        trajectory = smp.generate_trajectory(3.0, 'A')

        expected = [(0.0, 'A'), (1.0, 'B'), (2.0, 'A')]
        self.assertEqual(trajectory, expected)

    def test_early_termination(self):
        """测试时间溢出终止"""
        smp = SemiMarkovProcess(self.valid_states,
                              self.valid_trans,
                              {'A': lambda: 5.0, 'B': lambda: 1.0})
        trajectory = smp.generate_trajectory(3.0, 'A')
        self.assertEqual(len(trajectory), 1)  # 仅初始状态
    # endregion

    # region 嵌入式链测试
    def test_embedded_chain_conversion(self):
        """测试转移矩阵转换正确性"""
        smp = SemiMarkovProcess(self.valid_states, self.valid_trans, self.valid_holding)
        chain = smp.embedded_chain()

        expected_matrix = np.array([
            [0.3, 0.7],
            [1.0, 0.0]
        ])
        np.testing.assert_array_equal(chain.P, expected_matrix)
    # endregion

    # region 时间分布测试
    def test_time_average_calculation(self):
        """测试时间平均分布计算"""
        smp = SemiMarkovProcess(self.valid_states, self.valid_trans, self.valid_holding)
        test_trajectory = [
            (0.0, 'A'),
            (2.0, 'B'),  # A停留2单位
            (3.5, 'A'),  # B停留1.5单位
            (5.5, 'B')   # A停留2单位（总时间5.5）
        ]
        distribution = smp.time_average_distribution(test_trajectory)

        expected = {'A': 4 / 5.5, 'B': 1.5 / 5.5}
        for state in expected:
            self.assertAlmostEqual(distribution[state], expected[state])
    # endregion

    # region 绘图测试
    @patch('matplotlib.pyplot.figure')
    def test_plot_generation(self, mock_fig):
        """测试绘图函数调用"""
        smp = SemiMarkovProcess(self.valid_states, self.valid_trans, self.valid_holding)
        trajectory = [(0.0, 'A'), (1.0, 'B')]

        try:
            smp.plot_trajectory(trajectory, save_path='test.png')
            mock_fig.assert_called_once()  # 验证绘图初始化
        except ImportError:
            self.skipTest("Matplotlib not available")
    # endregion

if __name__ == '__main__':
    unittest.main()
