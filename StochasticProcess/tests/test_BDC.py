# test_BDC.py
import pytest
import numpy as np
from StochasticProcess.BDC import BirthDeathChain

class TestBirthDeathChain:
    def test_valid_initialization(self):
        """测试有效参数初始化"""
        birth = [0.2, 0.3]
        death = [0.1, 0.4]
        chain = BirthDeathChain(birth, death)
        assert chain.N == 3
        assert chain.states == ['0', '1', '2']

    def test_invalid_rate_length(self):
        """测试参数长度校验"""
        with pytest.raises(ValueError):
            BirthDeathChain([0.2], [0.3, 0.4])

    def test_transition_matrix_structure(self):
        """测试转移矩阵结构正确性"""
        birth = [0.2, 0.3]
        death = [0.1, 0.4]
        chain = BirthDeathChain(birth, death)
        P = chain.P
        
        # 验证边界状态
        assert P[0,0] == 0.8   # 1 - 0.2
        assert P[0,1] == 0.2
        assert P[2,1] == 0.4  # death[-1]
        assert P[2,2] == 0.6  # 1 - 0.4
        
        # 验证中间状态
        assert np.isclose(P[1,0], 0.1)  # death[0]
        assert np.isclose(P[1,2], 0.3)  # birth[1]
        assert np.isclose(P[1,1], 0.6)  # 1 - (0.1+0.3)

    def test_stationary_distribution(self):
        """测试稳态分布计算"""
        # 对称案例
        birth = [0.3, 0.2]
        death = [0.2, 0.3]
        chain = BirthDeathChain(birth, death)
        
        analytic = chain.analytic_stationary()
        numeric = chain.stationary_distribution()
        
        # 验证解析解和数值解一致性
        for state in chain.states:
            assert np.isclose(analytic[state], numeric[state], rtol=1e-6)

    def test_edge_cases(self):
        """测试边界条件"""
        # 单状态链
        with pytest.raises(ValueError):
            BirthDeathChain([], [])

        # 零概率情况
        chain = BirthDeathChain([0.0], [0.0])
        assert chain.P[0,0] == 1.0

if __name__ == "__main__":
    pytest.main()
