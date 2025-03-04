from MarcovC import MarkovChain
import unittest
import numpy as np

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        self.P1 = np.array([[0.8, 0.2],
                            [0.3, 0.7]])
        self.states1 = ['A', 'B']

        self.P2 = np.array([[0, 1],
                            [1, 0]])
        self.states2 = ['X', 'Y']

        self.P3 = np.array([[0.5, 0.5, 0],
                            [0.25, 0.5, 0.25],
                            [0, 0.5, 0.5]])
        self.states3 = ['S1', 'S2', 'S3']

    def test_stationary_distribution(self):
        mc1 = MarkovChain(self.P1, self.states1)
        pi1 = mc1.stationary_distribution(method='power')
        self.assertTrue(mc1.verify_stationary(pi1))

        expected_pi1 = {'A': 0.6, 'B': 0.4}
        for s in self.states1:
            self.assertAlmostEqual(pi1[s], expected_pi1[s], delta=1e-3)

    def test_method_consistency(self):
        mc = MarkovChain(self.P3, self.states3)
        pi_power = mc.stationary_distribution(method='power')
        pi_eigen = mc.stationary_distribution(method='eigen')
        pi_linear = mc.stationary_distribution(method='linear')

        # 比较三种方法结果差异
        for s in self.states3:
            diff = abs(pi_power[s] - pi_eigen[s])
            self.assertLess(diff, 1e-6)
            diff = abs(pi_eigen[s] - pi_linear[s])
            self.assertLess(diff, 1e-6)

    def test_sequence_generation(self):
        mc = MarkovChain(self.P2, self.states2)
        sequence = mc.generate_sequence(1000, 'X')

        # 验证周期性
        for i, s in enumerate(sequence):
            if i % 2 == 0:
                self.assertEqual(s, 'X')
            else:
                self.assertEqual(s, 'Y')

    def test_invalid_transition_matrix(self):
        with self.assertRaises(ValueError):
            invalid_P = np.array([[0.5, 0.6],
                                  [0.3, 0.7]])
            MarkovChain(invalid_P, self.states1)

        with self.assertRaises(ValueError):
            invalid_P = np.array([[1.1, -0.1],
                                  [0.5, 0.5]])
            MarkovChain(invalid_P, self.states1)

    def test_transition_count(self):
        mc = MarkovChain(self.P1, self.states1)
        sequence = mc.generate_sequence(10000, 'A')
        count = mc.transition_count(sequence)

        actual_P = count / count.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual_P, self.P1, atol=0.02)

if __name__ == '__main__':
    unittest.main()
