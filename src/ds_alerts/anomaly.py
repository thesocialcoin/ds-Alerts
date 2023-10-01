# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import Dict, List
import sys
import os
from os import listdir
from os.path import join
from ds_alerts.time_series import TimeSeries

class AnomalyDetector:
    """
    Class focused on the anomaly properties of a TimeSeries
    """

    def __init__(self, a: float = 0.5, multiplier: float = 1.96, window_size: int = 30):
        
        self.a = a
        self.multiplier = multiplier
        self.window_size = window_size


    def detect_alerts(self, ts: TimeSeries, a: float = 0.5) -> List[Dict[str, int]]:
        """
        Method to detect the alerts of the whole time series!
        """

        # Adjust hiperparameters

        # Initialize alerts
        alerts = []

        pred_interval = ts.prediction_interval(a=self.a, multiplier=self.multiplier, \
                                               window=self.window_size)

        for i, (date, value) in enumerate(
            zip(ts.dates[self.window_size:], ts.values[self.window_size:]), self.window_size
        ):
            # Detect and store the alert
            if value > pred_interval["upper_bound"][i]:
                alerts.append({"date": date, "volume": value})

        return alerts

"""


class AnomalyDetector:


    def detect_alerts(self, ts: TimeSeries) -> List[Dict[str, int]]:
      
        alerts = []

        # Get prediction interval and the window of the TS
        pred_interval = ts.prediction_interval()
        window = ts.WINDOW

        for i, (date, value) in enumerate(
            zip(ts.dates[window:], ts.values[window:]), window
        ):
            # Detect and store the alert
            if value > pred_interval["upper_bound"][i]:
                alerts.append({"date": date, "volume": value})

        return alerts

"""