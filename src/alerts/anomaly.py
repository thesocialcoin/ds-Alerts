# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import List
from alerts.time_series import AnomalyTS
from alerts.dataclasses import Event


class AnomalyDetector:
    """
    Class focused on the anomaly properties of a TimeSeries
    """

    def detect_alerts(self, ts: AnomalyTS) -> List[Event]:
        """
        Method to detect the alerts of the whole time series!
        """
        alerts = []

        # Get prediction interval and the window of the TS
        pred_interval = ts.prediction_interval()
        dates = [e.date for e in pred_interval]
        values = [e.value for e in pred_interval]
        upper_bound_limits = [e.upper_bound for e in pred_interval]
        window = ts.WINDOW

        for i, (date, value) in enumerate(
            zip(dates[window:], values[window:]), window
        ):
            # Detect and store the alert
            if value > upper_bound_limits[i]:
                alerts.append(Event(date, value))

        return alerts
