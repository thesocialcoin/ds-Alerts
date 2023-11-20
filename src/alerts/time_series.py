# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alerts.anomaly import AnomalyDetector
from alerts.dataclasses import TimeSeries, PredictionInterval, Event, LimitInterval

from abc import abstractmethod
from abc import ABC


def identity(x: Any) -> Any:
    return x


class AnomalyTS(ABC):

    @abstractmethod
    def prediction_interval() -> List[PredictionInterval]:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass


class AnomalyMeanTS(AnomalyTS):

    Z_SCORE = 1.625
    WINDOW = 30

    def __init__(self,
                 ts: TimeSeries,
                 z_score: float = Z_SCORE,
                 window: int = WINDOW,
                 flat_ts: bool = False):
        self.z_score = z_score
        self.window = window
        self.flat_ts = flat_ts
        self.ts = ts

    def _init_prediction_interval(self) -> List[PredictionInterval]:
        """Init the prediction intervals

        Returns:
            List[PredictionInterval]: The list contains
                the events pre window frame
        """
        prediction_interval = list()
        for event in self.ts.events[:self.window]:
            prediction_interval.append(
                PredictionInterval(
                    t=event.t,
                    v=np.sqrt(event.v) if self.flat_ts else event.v,
                    lb=None,
                    ub=None
                )
            )
        return prediction_interval

    def _compute_limits(
            self,
            current_window_frame: List[Event]) -> LimitInterval:

        values = [e.v for e in current_window_frame]
        if self.flat_ts:
            values = np.sqrt(self.ts.values)

        mean_value = np.mean(values)
        std_value = np.std(values)
        lower_bound = mean_value - self.z_score * std_value
        upper_bound = mean_value + self.z_score * std_value

        limits = LimitInterval(
            lb=lower_bound,
            ub=upper_bound
        )

        return limits

    def prediction_interval(self) -> List[PredictionInterval]:

        prediction_interval = self._init_prediction_interval
        future_events = self.ts.events[self.window:]

        for i, (date, value) in enumerate(future_events):

            current_window_frame = self.ts.events[i: self.window + i]

            limits = self._compute_limits(current_window_frame)
            lower_bound = 0 if limits.lower_bound < 0 else limits.upper_bound
            upper_bound = limits.upper_bound

            prediction_interval["lower_bound"].append(lower_bound)
            prediction_interval["upper_bound"].append(upper_bound)
            prediction_interval["values"].append(
                np.sqrt(value) if self.flat_ts else value
            )
            prediction_interval["dates"].append(date)

        return prediction_interval

    def plot(self, show_anomalies=False) -> None:

        plt.subplots(figsize=(12, 4))
        prediction_interval = self.prediction_interval()

        df = pd.DataFrame([p.__dict__ for p in prediction_interval])

        plt.plot(df['time'], df['value'], label='value')
        plt.plot(df['time'], df['lower_bound'], label='lower_bound')
        plt.plot(df['time'], df['upper_bound'], label='upper_bound')

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






class AnomalyTS2():

    Z_SCORE = 1.625
    WINDOW = 30

    def __init__(self, ):


        self.dates, self.values = zip(*T)
        self.dates = list(self.dates)
        self.values = list(self.values)

        self.window = (
            parameters['window'] if 'window' in parameters else self.WINDOW
        )

        self.quantile = (
            parameters['quantile']
            if 'quantile' in parameters else self.Z_SCORE
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

            lower_bound = mean_value - self.quantile * std_value
            upper_bound = mean_value + self.quantile * std_value
        else:
            median = self._median(values)
            mad = self._mad(values)

            # Calculate bounds
            lower_bound = median - (self.quantile * 1.4826 * mad)
            upper_bound = median + (self.quantile * 1.4826 * mad)

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






class AnomalyMedianTS(AnomalyTS):
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
