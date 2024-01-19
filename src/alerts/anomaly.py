# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import List
from alerts.time_series import AnomalyTS
from alerts.dataclasses import Event
from datetime import timedelta


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

        for i, (date, value) in enumerate(zip(dates[window:], values[window:]), window):
            # Detect and store the alert
            if value > upper_bound_limits[i]:
                alerts.append(Event(date, value))

        return alerts

    def detect_alerts_groups(self, ts: AnomalyTS, span: int = 5) -> List[List[Event]]:
        time_proximity = timedelta(days=span)
        alerts = self.detect_alerts(ts)
        alerts_length = len(alerts)
        grouping_alerts = []
        i = 0

        while i < alerts_length:
            start_anomaly = alerts[i]
            anomaly = alerts[i]
            group = [start_anomaly]
            continues_anomalies = alerts[i + 1 :]

            if len(continues_anomalies) == 0:
                i += 1
                pass

            for next_anomaly in continues_anomalies:
                time_difference = abs(
                    (anomaly.date - next_anomaly.date)
                    .astype("timedelta64[D]")
                    .astype(int)
                )
                i += 1
                if time_difference > time_proximity.days:
                    break
                else:
                    anomaly = next_anomaly
                    group.append(next_anomaly)

            grouping_alerts.append(group)

        return grouping_alerts
