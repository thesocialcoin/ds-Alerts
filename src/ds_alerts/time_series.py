# ####################### Code Implemented by Alejandro Bonell ########################
# Inspired by: Anomaly detection in univariate time series incorporating active learning
# Link: https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import List

import pandas as pd


class TimeSeries:
    """
    Class to represent a time series and to calculate some typical properties of it
    """

    def __init__(self, ts: List[int], dates: List[str]):
        """
        Initializes the TS with raw time data values and its corresponding dates
        """
        self.data = pd.Series(ts, index=dates)

    def compute_trend(self, window: int):
        """
        Computes the trend of the time series
        """
        return self.data.rolling(window=window, closed="left").mean()

    def compute_std(self, window: int):
        """
        Computes the standard deviation of the time series
        """
        return self.data.rolling(window=window, closed="left").std()

    def prediction_interval(self, window: int, multiplier: float = 1.96):
        """
        Calculates the prediction intervals for the TS or the anomaly adjusted TS
        """
        trend = self.compute_trend(window)
        std = self.compute_std(window)

        lower_bound = []
        upper_bound = []

        for t, s in zip(trend, std):
            lower_item = t - (multiplier * s)
            lower_item = lower_item if lower_item > 0 else 0
            lower_bound.append(lower_item)

            upper_item = t + (multiplier * s)
            upper_bound.append(upper_item)

        return {
            "dates": self.data.index.tolist(),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
