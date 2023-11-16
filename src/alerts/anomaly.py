# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import Dict, List

from alerts.time_series import TimeSeries


class AnomalyDetector:
    """
    Class focused on the anomaly properties of a TimeSeries
    """

    def detect_alerts(self, ts: TimeSeries) -> List[Dict[str, int]]:
        """
        Method to detect the alerts of the whole time series!
        """
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
