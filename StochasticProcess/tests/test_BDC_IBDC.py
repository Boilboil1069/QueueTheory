# test_BDC.py
import unittest
import pytest
import numpy as np
from math import isclose
from StochasticProcess.BDC import BirthDeathChain, InfiniteBirthDeathChain

class TestBDC:
    def test_valid_initialization(self):
        birth = [0.2, 0.3]
        death = [0.1, 0.4]
        chain = BirthDeathChain(birth, death)
        assert chain.N == 3
        assert chain.states == ['0', '1', '2']

    def test_invalid_rate_length(self):
        with pytest.raises(ValueError):
            BirthDeathChain([0.2], [0.3, 0.4])

    def test_transition_matrix_structure(self):
        birth = [0.2, 0.3]
        death = [0.1, 0.4]
        chain = BirthDeathChain(birth, death)
        P = chain.P

        assert P[0,0] == 0.8
        assert P[0,1] == 0.2
        assert P[2,1] == 0.4
        assert P[2,2] == 0.6
        
        # 验证中间状态
        assert np.isclose(P[1,0], 0.1)
        assert np.isclose(P[1,2], 0.3)
        assert np.isclose(P[1,1], 0.6)

    def test_stationary_distribution(self):
        birth = [0.3, 0.2]
        death = [0.2, 0.3]
        chain = BirthDeathChain(birth, death)
        
        analytic = chain.analytic_stationary()
        numeric = chain.stationary_distribution()

        for state in chain.states:
            assert np.isclose(analytic[state], numeric[state], rtol=1e-5, atol=1e-6)

    def test_edge_cases(self):
        with pytest.raises(ValueError):
            BirthDeathChain([], [])

        chain = BirthDeathChain([0.0], [0.0])
        assert chain.P[0,0] == 1.0

class TestInfiniteBDC(unittest.TestCase):
    def setUp(self):
        """Test case with constant rates λ=0.4, μ=0.6"""
        self.chain = InfiniteBirthDeathChain(
            birth_fn=lambda i: 0.4,
            death_fn=lambda i: 0.6 if i>0 else 0
        )

    def test_transition_probs(self):
        """Verify transition probabilities calculation"""
        # Test state 0
        probs0 = self.chain.transition_probs(0)
        self.assertAlmostEqual(probs0[0], 0.6)
        self.assertAlmostEqual(probs0[1], 0.4)

        # Test state 5
        probs5 = self.chain.transition_probs(5)
        self.assertAlmostEqual(probs5[4], 0.6)
        self.assertAlmostEqual(probs5[5], 0.0)
        self.assertAlmostEqual(probs5[6], 0.4)

    def test_stationary_distribution(self):
        """Verify stationary distribution properties"""
        pi = self.chain.analytic_stationary()

        # Sum to 1 check
        self.assertTrue(isclose(sum(pi.values()), 1.0, rel_tol=1e-6))

        # Geometric distribution verification
        r = 0.4 / 0.6
        for i, prob in pi.items():
            expected = (1 - r) * (r**i)
            self.assertTrue(isclose(prob, expected, rel_tol=1e-4))

    def test_boundary_handling(self):
        """Test state 0 death rate handling"""
        with self.assertRaises(ValueError):
            InfiniteBirthDeathChain(
                    birth_fn=lambda i: 0.5,
                    death_fn=lambda i: 0.5  # Invalid for i=0
            )

if __name__ == "__main__":
    # pytest.main()
    unittest.main()
