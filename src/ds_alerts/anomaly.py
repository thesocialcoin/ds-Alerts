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

    def detect_alerts(self, ts: TimeSeries) -> List[Dict[str, int]]:
        """
        Method to detect the alerts of the whole time series!
        """

        mod_ts = copy.deepcopy(ts)
        alerts = []
        threshold = max(self.threshold, ts.data.median())

        # FIRST PART: Compute the upper bound & lower bound
        pred_interval = mod_ts.prediction_interval(self.window)
        upper_bound = pred_interval["upper_bound"]
        lower_bound = pred_interval["lower_bound"]

        for i, (index, value) in enumerate(
            mod_ts.data[self.window :].items(), self.window
        ):
            # SECOND PART: Detect the alert
            if value > upper_bound[i] and value > threshold:
                # THIRD PART: Store the alert
                alerts.append({"date": index, "volume": value})

                # FOURTH PART a): UPDATE the time series
                value = value - self.a * (value - upper_bound[i])

            if value < lower_bound[i]:
                # FOURTH PART b) : UPDATE the time series
                value = value + self.a * (lower_bound[i] - value)

        return alerts
