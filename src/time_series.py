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


WINDOW_SIZE_STD = 30  
WINDOW_SIZE_MEAN = 30




############################## TimeSeries ############################## 

class timeSeries:

    def __init__(self, ts:np.array, dates:np.array):
        
        self.time_series = pd.Series(ts, index=pd.to_datetime(dates))
        self.time_series_anomaly = self.time_series.copy()

        self.trend = None
        self.seasonality = None
        self.residual = None

        self.deseasonalized_time_series = None
        self.multiplier = 1.645

        self.lower_bound = None
        self.upper_bound = None

    def decompose_time_series(self):

        decompose_time_series = seasonal_decompose(self.time_series, model='additive')

        self.trend = decompose_time_series.trend
        self.seasonality = decompose_time_series.seasonal
        self.residual = decompose_time_series.resid

    def remove_seasonality(self):

        if self.seasonality is None:
            self.decompose_time_series()

        self.deseasonalized_time_series = self.time_series - self.seasonality

    def confidence_intervals(self, anomaly=False):

        if self.deseasonalized_time_series is None:
            self.remove_seasonality()
        
        if not anomaly:
            trend_window = self.time_series.rolling(window=WINDOW_SIZE_MEAN).median()
            rolling_std = self.time_series.rolling(window=WINDOW_SIZE_STD).std()

        else:
            trend_window = self.time_series_anomaly.rolling(window=WINDOW_SIZE_MEAN).median()
            rolling_std = self.time_series_anomaly.rolling(window=WINDOW_SIZE_STD).std()



        self.lower_bound = trend_window - (self.multiplier * rolling_std)
        self.upper_bound = trend_window + (self.multiplier * rolling_std)

    def plot(self):

        if self.trend is None:
            self.decompose_time_series()

        if self.lower_bound is None:
            self.confidence_intervals()

        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_series, label='Original', linewidth=2)
        plt.plot(self.trend, color='black', label='Trend', linewidth=2)
        plt.fill_between(self.time_series.index, self.lower_bound, self.upper_bound, color='b', alpha=.1, label='Confidence Intervals')
        plt.title('Time series with trend and confidence intervals', fontsize=20)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Value', fontsize=15)
        plt.legend(fontsize=13)
        sns.despine()
        plt.show()
            


############################## TimeSeries ############################## 

class AnomalyTS(timeSeries):

    def __init__(self, ts, dates, threshold:float, a:float=0.5):
        super().__init__(ts, dates)
        self.threshold = threshold
        self.a = a
        self.alerts = None
    
    def detect_alerts(self):

        if self.lower_bound is None:
            self.confidence_intervals()

        alerts_idx = []
        alert = False
        self.alerts = pd.Series([np.nan]*len(self.time_series), index=self.time_series.index)

        for i in range(len(self.time_series)):
            
            if i==0 or math.isnan(self.lower_bound[i]):
                continue

            if not alert and self.time_series[i] > self.upper_bound[i] and self.time_series[i] > self.threshold \
                     and self.time_series[i]>self.time_series[i-1]:
                
                alert = True
                alert_point = self.time_series[i] - self.a*(self.time_series[i] - self.upper_bound[i]) 

            if alert and self.time_series[i] >= alert_point:
                
                self.time_series_anomaly[i] = alert_point
                self.alerts[i] = self.time_series[i]
                alerts_idx.append(self.time_series.index[i])
            
            self.confidence_intervals(anomaly=True)

        return alerts_idx

    
    def plot(self):

        if self.trend is None:
            self.decompose_time_series()

        if self.lower_bound is None:
            self.confidence_intervals()

        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_series, label='Original', linewidth=2)
        #plt.plot(self.time_series_anomaly, color='red', label='Anomaly-adjusted', linewidth=2)
        plt.plot(self.trend, color='black', label='Trend', linewidth=2)
        plt.fill_between(self.time_series.index, self.lower_bound, self.upper_bound, color='b', alpha=.1, label='Confidence Intervals')
        plt.axhline(y=self.threshold, color='g', linestyle='--', label='Threshold')  # Threshold line
        plt.scatter(self.alerts.index, self.alerts, color='r', label='Alerts')  # Plot alerts
        plt.title('Time series with trend, confidence intervals, and threshold', fontsize=20)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Value', fontsize=15)
        plt.legend(fontsize=13)
        sns.despine()
        plt.show()
            



            
            #Start an alert  

