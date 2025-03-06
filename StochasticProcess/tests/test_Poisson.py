import unittest
from unittest.mock import patch
import numpy as np
from StochasticProcess.Poisson import PoissonProcess  # Assume code is in demo.py

class TestPoissonProcess(unittest.TestCase):
    def setUp(self):
        self.rate = 2.0
        self.obj = PoissonProcess(self.rate)

    # --- Init Tests ---
    def test_init_with_invalid_rate(self):
        with self.assertRaises(ValueError):
            PoissonProcess(0)

    def test_init_with_valid_rate(self):
        self.assertEqual(self.obj.rate, self.rate)

    # --- Generate Events Tests ---
    def test_generate_events_zero_time(self):
        events = self.obj.generate_events(0)
        self.assertEqual(len(events), 0)

    @patch('numpy.random.exponential')
    def test_generate_events_ordering(self, mock_exp):
        mock_exp.side_effect = [0.3, 0.2, 0.4]  # Force specific intervals
        events = self.obj.generate_events(1.0)
        self.assertTrue(np.all(np.diff(events) > 0))  # Check ascending

    # --- Event Count Tests ---
    @patch('numpy.random.poisson')
    def test_event_count_call(self, mock_poisson):
        t = 5
        self.obj.event_count(t)
        mock_poisson.assert_called_with(self.rate * t)

    # --- Statistical Tests ---
    def test_expectation_calculation(self):
        t = 10
        self.assertEqual(self.obj.expectation(t), self.rate * t)

    def test_variance_calculation(self):
        t = 10
        self.assertEqual(self.obj.variance(t), self.rate * t)

    # --- Time Between Events Tests ---
    @patch('numpy.random.exponential')
    def test_time_between_events_size(self, mock_exp):
        size = 5
        self.obj.time_between_events(size)
        mock_exp.assert_called_with(1/self.rate, size)

if __name__ == '__main__':
    unittest.main()
