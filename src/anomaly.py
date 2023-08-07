
import math
import pandas as pd
import copy
import numpy as np
import sys

sys.path.append('/Users/alejandrobonell/ds-Alerts')
from src.time_series import timeSeries

class algorithmAnomalyTimeSeries():
    """
        A subclass of Time Series more focused on the anomaly properties of the time series
    """
    def __init__(self, window:int, absolute_threshold:float=10, a:float=0.5, multiplier:float=1.96):

        #hyperparameters Algorithm
        self.threshold = absolute_threshold
        self.a = a
        self.multiplier = multiplier
        self.window = window

    def detect_alerts(self, time_series:timeSeries):
        """
        Method to detect the alerts of the whole time series!
        """
        
        modified_time_series = copy.deepcopy(time_series)
        alerts_idx = []
        alerts = pd.Series([np.nan]*len(time_series.time_series), index=time_series.time_series.index)
        threshold = max(self.threshold, time_series.time_series.median())


        for i in range(len(modified_time_series.time_series)):

            # FIRST PART: Compute the upper bound & lower bound
            modified_time_series.compute_trend(self.window)
            modified_time_series.compute_std(self.window)

            ### Let's check the confidence intervals are available
            if i == 0 or math.isnan(modified_time_series.trend[i]):
                continue

            upper_bound = modified_time_series.trend + (self.multiplier * modified_time_series.std)
            lower_bound = modified_time_series.trend - (self.multiplier * modified_time_series.std)

            #SECOND PART: Detect the alert

            if modified_time_series.time_series[i] > upper_bound[i] \
                  and modified_time_series.time_series[i] > threshold:
            
                # THIRD PART: Store the alert
                alerts_idx.append(i)
                alerts[i] = time_series.time_series[i]

                # FOURTH PART a): UPDATE the time series
                modified_time_series.time_series[i] = modified_time_series.time_series[i] - self.a*(modified_time_series.time_series[i] - upper_bound[i])


            if modified_time_series.time_series[i] < lower_bound[i]:

                # FOURTH PART b) : UPDATE the time series
                modified_time_series.time_series[i] = modified_time_series.time_series[i] + self.a*(lower_bound[i] - modified_time_series.time_series[i])
            


        return {'alerts':alerts, 'alerts_idx':alerts_idx, 'modified_time_series':modified_time_series, 'original_time_series':time_series}
    