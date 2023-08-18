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

        self.data = pd.Series(ts, index=pd.to_datetime(dates))

        self.trend = None
        self.std = None

        self.lower_bound = None
        self.upper_bound = None

    def compute_trend(self, window: int):
        """
        Computes the trend of the time series
        """
        self.trend = self.data.rolling(window=window, closed="left").mean()

    def compute_std(self, window: int):
        """
        Computes the standard deviation of the time series
        """
        self.std = self.data.rolling(window=window, closed="left").std()

    def prediction_interval(self, window: int, multiplier: float = 1.96):
        """
        Calculates the prediction intervals for the TS or the anomaly adjusted TS
        """

        if self.trend is None:
            self.compute_trend(window)

        if self.std is None:
            self.compute_std(window)

        self.lower_bound = self.trend - (multiplier * self.std)
        self.upper_bound = self.trend + (multiplier * self.std)
