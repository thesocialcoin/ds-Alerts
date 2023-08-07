import sys
import os
import pandas as pd
import math

sys.path.append(os.getcwd())

import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

#from src.conf import (WINDOW_SIZE)
import matplotlib.pyplot as plt
import seaborn as sns


######################## Code Implemented by Alejandro Bonell ########################
# Inspired by: Anomaly detection in univariate time series incorporating active learning
# Link: https://www.sciencedirect.com/science/article/pii/S2772415822000323
# Version 0.1


############################ TimeSeries ############################

class timeSeries:
    """
    A class to represent a Time Series and to calculate some typical properties of 
    Time series
    """

    WINDOW_SIZE_STD = 20  
    WINDOW_SIZE_MEAN = 20

    def __init__(self, ts:np.array, dates:np.array):
        """
        Initializes the time series with raw time data values and its corresponding dates
        Args:
            - ts, numpy array: the time series values
            - dates, numpy array: the corresponding dates
        """
        
        self.time_series = pd.Series(ts, index=pd.to_datetime(dates))

        self.trend = None
        self.std = None

        self.lower_bound = None
        self.upper_bound = None
    
    def compute_trend(self, window):
        """
        Computes the trend of the time series
        """
        self.trend = self.time_series.rolling(window=window, closed='left').mean()
    
    def compute_std(self, window):
        """
        Computes the standard deviation of the time series
        """
        self.std = self.time_series.rolling(window=window, closed='left').std()

    def prediction_interval(self, window, multiplier=1.96):
        """
        Calculates the confidence intervals for the time series or the anomaly adjusted time series.
        
        Args:
        anomaly (bool): If True, compute confidence intervals for the anomaly adjusted time series.
        """

        if self.trend is None:
            self.compute_trend(window)
        
        if self.std is None:
            self.compute_std(window)

        self.lower_bound = self.trend - (multiplier * self.std)
        self.upper_bound = self.trend + (multiplier * self.std)



