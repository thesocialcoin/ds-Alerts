# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alerts.anomaly import AnomalyDetector


class AnomalyTS():

    QUANTILE = 1.625
    WINDOW = 30

    def __init__(self,
                 T: List[Tuple[Any, int]],
                 parameters: Dict[str, Any]):
        """Time series anomaly

        Args:
            T (List[Tuple[int, int]]): represents data point containing a
        timestamp and an associating value.
            parameters (Dict[str, Any]): set of hyper-parameters that help to
                compute the prediction intervals.
        """

        self.dates, self.values = zip(*T)
        self.dates = list(self.dates)
        self.values = list(self.values)

        self.window = (
            parameters['window'] if 'window' in parameters else self.WINDOW
        )

        self.static_metric = (
            parameters['static_metric']
            if 'static_metric' in parameters else 'mean'
        )

        self.transf = parameters['transf'] if 'transf' in parameters else None

        assert self.static_metric in ['mean', 'median']
        assert self.transf is None or self.transf == 'sqrt'

    def normalize_time_series(self) -> List[Tuple[Any, int]]:
        normalized_values = self._flatten_transform_sqrt(self.values)
        return list(zip(self.dates, normalized_values))

    def _flatten_transform_sqrt(self, values: List[int]) -> List[int]:
        np_values = np.array(values)
        flatten_values = np.sqrt(np_values)
        return flatten_values

    def _inverse_transform_sqrt(self, values: List[int]) -> List[int]:
        np_values = np.array(values)
        inverse_values = np.square(np_values)
        return inverse_values

    def _compute_metrics(
            self, pre_window_values: List[Any],
            static_metric: str,
            transf: Optional[str] = None) -> Dict[str, Any]:

        assert static_metric in ['mean', 'median']
        assert self.transf is None or self.transf == 'sqrt'

        values = pre_window_values.copy()

        if transf and transf == 'sqrt':
            values = self._flatten_transform_sqrt(values)
            f = self._flatten_transform_sqrt
        else:
            f = lambda x :  x

        values = np.array(values)

        if static_metric == 'mean':
            mean_value = np.mean(values)
            std_value = np.std(values)

            lower_bound = mean_value - self.QUANTILE * std_value
            upper_bound = mean_value + self.QUANTILE * std_value
        else:
            median = self._median(values)
            mad = self._mad(values)

            # Calculate bounds
            lower_bound = median - (self.QUANTILE * 1.4826 * mad)
            upper_bound = median + (self.QUANTILE * 1.4826 * mad)

        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'apply': f
        }

    def _mad(self, values: List[int]) -> float:
        median = self._median(values)
        return self._median([abs(x - median) for x in values])

    def _median(self, values: List[int]) -> float:
        length = len(values)
        ordered_values = sorted(values)
        if length % 2:
            # case even length
            lower = ordered_values[int(length / 2) - 1]
            higher = ordered_values[int(length / 2)]
            return (higher + lower) / 2
        else:
            # case odd length
            return ordered_values[int(length / 2)]

    def prediction_interval(self) -> Dict[str, Any]:

        def transform_value(value):
            if self.transf and self.transf == 'sqrt':
                value = self._flatten_transform_sqrt([value])[0]
            return value

        prediction_interval = {
            "dates": [date for date in self.dates[:self.window]],
            "values": [
                transform_value(value)
                for value in self.values[:self.window]
            ],
            "lower_bound": [None for _ in self.dates[:self.window]],
            "upper_bound": [None for _ in self.dates[:self.window]]
        }

        post_window_dates = self.dates[self.window:]
        post_window_values = self.values[self.window:]
        targets = zip(post_window_dates, post_window_values)

        for i, (date, value) in enumerate(targets):
            # Calculate statistic values for this window

            pre_window_values = self.values[i: self.window + i]

            metrics = self._compute_metrics(
                pre_window_values,
                self.static_metric,
                self.transf
            )
            lower_bound = metrics['lower_bound']
            upper_bound = metrics['upper_bound']
            apply = metrics['apply']

            if lower_bound and lower_bound < 0:
                lower_bound = 0.0

            value = apply([value])[0]

            prediction_interval["lower_bound"].append(lower_bound)
            prediction_interval["upper_bound"].append(upper_bound)
            prediction_interval["values"].append(value)
            prediction_interval["dates"].append(date)

        return prediction_interval

    def plot(self, show_anomalies=False) -> None:

        plt.subplots(figsize=(12, 4))

        pred_interval = pd.DataFrame(self.prediction_interval())
        simplified_preds = pred_interval.copy()
        # simplified_preds = pred_interval[~pred_interval['lower_bound'].isna()]

        plt.plot(simplified_preds['dates'],
                 simplified_preds['values'], label='value')

        plt.plot(simplified_preds['dates'],
                 simplified_preds['lower_bound'], label='lower_bound')

        plt.plot(simplified_preds['dates'],
                 simplified_preds['upper_bound'], label='upper_bound')

        if show_anomalies:
            detector = AnomalyDetector()
            anomalies = detector.detect_alerts(self)

            x = [anomalie['date'] for anomalie in anomalies]
            y = [anomalie['volume'] for anomalie in anomalies]

            plt.scatter(x, y,
                        marker='o',
                        alpha=0.5,
                        color='red')

        plt.xlabel("Days")
        plt.ylabel("Magnitude")
        plt.title(
            "Static method: {} - Transformation: {} - Anomalies: {}"
            .format(self.static_metric,
                    self.transf,
                    len(x) if show_anomalies else 0)
        )

        plt.legend()
        plt.show()


class TimeSeries:
    """
    Class to represent a time series and to calculate some
    typical properties of it
    """

    # Algorithm hiperparameters
    MULTIPLIER = 1.625
    THRESHOLD = 10.0
    WINDOW = 30

    def __init__(self, values: List[int], dates: List[str]):
        self.values = values
        self.dates = dates

    def prediction_interval(
        self,
        multiplier: float = MULTIPLIER,
        threshold: float = THRESHOLD,
        window: int = WINDOW,
    ) -> Dict[str, Any]:
        # Initial before window values
        prediction_interval = {
            "dates": [date for date in self.dates[:window]],
            "lower_bound": [None for _ in self.dates[:window]],
            "upper_bound": [None for _ in self.dates[:window]],
        }

        # Original copies from window for calculation

        for i, (value, date) in enumerate(
            zip(self.values[window:], self.dates[window:]), window
        ):
            # Calculate statistic values for this window
            median = self._median(self.values[i - window : i])
            mad = self._mad(self.values[i - window : i])

            # Calculate bounds
            upper_bound = median + (multiplier * 1.4826 * mad)
            lower_bound = median - (multiplier * 1.4826 * mad)

            if lower_bound and lower_bound < 0:
                lower_bound = 0.0

            # Add bounds to calculation
            prediction_interval["dates"].append(date)
            prediction_interval["lower_bound"].append(lower_bound)
            prediction_interval["upper_bound"].append(upper_bound)

        return prediction_interval

    def _mean(self, values: List[int]) -> float:
        return sum(values) / len(values)

    def _median(self, values: List[int]) -> float:
        length = len(values)
        ordered_values = sorted(values)
        if length % 2:
            # case even length
            lower = ordered_values[int(length / 2) - 1]
            higher = ordered_values[int(length / 2)]
            return (higher + lower) / 2
        else:
            # case odd length
            return ordered_values[int(length / 2)]

    def _std(self, values: List[int]) -> float:
        mean = self._mean(values)
        return (sum([((x - mean) ** 2) for x in values]) / len(values)) ** 0.5

    def _mad(self, values: List[int]) -> float:
        median = self._median(values)
        return self._median([abs(x - median) for x in values])
