# ####################### Code Implemented by Alejandro Bonell ########################
# Inspired by: Anomaly detection in univariate time series incorporating active learning
# Link: https://www.sciencedirect.com/science/article/pii/S2772415822000323

import copy
from typing import Dict, List

from .time_series import TimeSeries


class AnomalyDetector:
    """
    A class associated to the TimeSeries class more focused on the anomaly properties of it
    """

    WINDOW = 30

    def __init__(
        self,
        window: int = WINDOW,
        threshold: float = 10.0,
        a: float = 0.5,
        multiplier: float = 1.96,
    ):
        # hyperparameters Algorithm
        self.threshold = threshold
        self.a = a
        self.multiplier = multiplier
        self.window = window

    def detect_alerts(self, time_series: TimeSeries):
        """
        Method to detect the alerts of the whole time series!
        """

        mod_ts = copy.deepcopy(time_series)
        alerts = []
        threshold = max(self.threshold, time_series.data.median())

        # FIRST PART: Compute the upper bound & lower bound
        pred_interval = mod_ts.prediction_interval(self.window)
        upper_bound = pred_interval["upper_bound"]
        lower_bound = pred_interval["lower_bound"]

        for i, item in enumerate(mod_ts.data[self.window :], self.window):
            # SECOND PART: Detect the alert
            if item > upper_bound[i] and item > threshold:
                # THIRD PART: Store the alert
                alerts.append({item.index: item})

                # FOURTH PART a): UPDATE the time series
                item = item - self.a * (item - upper_bound[i])

            if item < lower_bound[i]:
                # FOURTH PART b) : UPDATE the time series
                item = item + self.a * (lower_bound[i] - item)

        return alerts
