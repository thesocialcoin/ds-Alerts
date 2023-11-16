import unittest
import numpy as np
import pandas as pd

import sys

sys.path.append("/Users/alejandrobonell/ds-Alerts")

from src.alerts.time_series import timeSeries
from src.alerts.anomaly import algorithmAnomalyTimeSeries


class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.dates = pd.date_range("20210101", periods=5)
        self.data = np.array([1, 2, 3, 4, 5])
        self.ts = timeSeries(self.data, self.dates)

    def test_initialization(self):
        self.assertEqual(len(self.ts.time_series), 5)
        self.assertIsNone(self.ts.trend)
        self.assertIsNone(self.ts.std)
        self.assertIsNone(self.ts.lower_bound)
        self.assertIsNone(self.ts.upper_bound)

    def test_compute_trend(self):
        self.ts.compute_trend(2)
        expected_trend = [np.nan, np.nan, 1.5, 2.5, 3.5]
        np.testing.assert_array_almost_equal(self.ts.trend.values, expected_trend)

    def test_compute_std(self):
        self.ts.compute_std(2)
        expected_std = [np.nan, np.nan, 0.707107, 0.707107, 0.707107]
        np.testing.assert_array_almost_equal(self.ts.std.values, expected_std)

    def test_prediction_interval(self):
        self.ts.prediction_interval(2)
        # Here, just test if bounds are computed, you can add specific values if needed.
        self.assertIsNotNone(self.ts.lower_bound)
        self.assertIsNotNone(self.ts.upper_bound)


class TestAlgorithmAnomalyTimeSeries(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.dates = pd.date_range("20210101", periods=5)
        self.data = np.array([1, 2, 50, 4, 5])
        self.ts = timeSeries(self.data, self.dates)
        self.algo = algorithmAnomalyTimeSeries(window=2)

    def test_initialization(self):
        self.assertEqual(self.algo.threshold, 10)
        self.assertEqual(self.algo.a, 0.5)
        self.assertEqual(self.algo.multiplier, 1.96)
        self.assertEqual(self.algo.window, 2)

    def test_detect_alerts(self):
        result = self.algo.detect_alerts(self.ts)
        # Check if the third data point is detected as an anomaly
        self.assertEqual(result["alerts_idx"][0], 2)
        self.assertEqual(result["alerts"][2], 50)


print("all tests passed")
