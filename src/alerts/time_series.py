# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


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
        self.window = (
            parameters['window'] if 'window' in parameters else self.WINDOW
        )

    def normalize_time_series(self) -> List[Tuple[Any, int]]:
        normalized_values = self._flatten_transform(self.values)
        return list(zip(self.dates, normalized_values))

    def _flatten_transform(self, values: List[int]) -> List[int]:
        np_values = np.array(values)
        flatten_values = np.sqrt(np_values)
        return flatten_values

    def _inverse_transform(self, values: List[int]) -> List[int]:
        np_values = np.array(values)
        inverse_values = np.square(np_values)
        return inverse_values

    def prediction_interval(self) -> Dict[str, Any]:
        prediction_interval = {
            "dates": [date for date in self.dates[:self.window]],
            "values": [value for value in self.values[:self.window]],
            "lb": [None for _ in self.dates[:self.window]],
            "ub": [None for _ in self.dates[:self.window]],
            "flatten_values": [
                self._flatten_transform(value)
                for value in self.values[:self.window]
            ],
            "flatten_lb": [None for _ in self.dates[:self.window]],
            "flatten_ub": [None for _ in self.dates[:self.window]],
        }

        post_window_dates = self.dates[self.window:]
        post_window_values = self.values[self.window:]
        targets = enumerate(zip(post_window_dates, post_window_values))

        for i, (date, value) in targets:
            # Calculate statistic values for this window

            pre_window_values = self.values[i: self.window + i]
            flatten_pre_window_values = self. \
                _flatten_transform(pre_window_values)

            flatten_value = self._flatten_transform([value])[0]
            flatten_mean_value = np.mean(flatten_pre_window_values)
            flatten_std_value = np.std(flatten_pre_window_values)

            mean_value = np.mean(pre_window_values)
            std_value = np.std(pre_window_values)

            flatten_lb = flatten_mean_value - self.QUANTILE * flatten_std_value
            flatten_ub = flatten_mean_value + self.QUANTILE * flatten_std_value

            lb = mean_value - self.QUANTILE * std_value
            ub = mean_value + self.QUANTILE * std_value

            if lb and lb < 0:
                lb = 0.0

            if flatten_lb and flatten_lb < 0:
                flatten_lb = 0.0

            prediction_interval["dates"].append(date)
            prediction_interval["values"].append(value)
            prediction_interval["lb"].append(lb)
            prediction_interval["ub"].append(ub)
            prediction_interval["flatten_values"].append(flatten_value)
            prediction_interval["flatten_lb"].append(flatten_lb)
            prediction_interval["flatten_ub"].append(flatten_ub)

        return prediction_interval

    def plot(self) -> None:
        plt.subplots(figsize=(12, 5))
        plt.plot(self.dates, self.values)


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
