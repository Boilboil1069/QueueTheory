import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm

from StochasticProcess.MarkovProcess import ContinuousTimeMarkovProcess,InfiniteContinuousTimeMarkovProcess

class TestContinuousTimeMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.valid_Q = np.array([[-2, 1, 1],
                                 [2, -4, 2],
                                 [0, 3, -3]], dtype=np.float64)
        self.states = ["A", "B", "C"]
        self.ctmp = ContinuousTimeMarkovProcess(self.valid_Q, self.states)

    def test_init_validation(self):
        invalid_Q = np.array([[1, -1], [-1, 1]])
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovProcess(invalid_Q, ["X", "Y"])

    @patch('numpy.random.exponential')
    @patch('numpy.random.choice')
    def test_trajectory_generation(self, mock_choice, mock_exp):
        mock_exp.return_value = 0.5
        mock_choice.side_effect = [1, 2]

        trajectory = self.ctmp.generate_trajectory(2.0, "A")
        self.assertEqual(len(trajectory), 3)
        self.assertAlmostEqual(trajectory[-1][0], 1.0)

    def test_embedded_chain(self):
        embedded = self.ctmp.embedded_chain()
        self.assertTrue(np.allclose(embedded.P.sum(axis=1), 1))

    def test_transition_probability(self):
        P = self.ctmp.transition_probability(0.0)
        self.assertTrue(np.allclose(P, np.eye(3)))

    def test_stationary_distribution(self):
        sym_Q = np.array([[-1, 1], [1, -1]])
        ctmp = ContinuousTimeMarkovProcess(sym_Q, ["X", "Y"])
        pi = ctmp.stationary_distribution()
        self.assertAlmostEqual(pi["X"], 0.5)

class TestInfiniteCTMP(unittest.TestCase):
    def test_jump_process(self):
        def mock_rates(i):
            return {i+1: 1.0} if i < 2 else {}

        process = InfiniteContinuousTimeMarkovProcess(mock_rates)
        with patch('numpy.random.exponential', return_value=0.1):
            seq = process.jump_process(0, max_jumps=2)
            self.assertEqual(seq, [(0.0,0), (0.1,1), (0.2,2)])

if __name__ == "__main__":
    unittest.main()
