# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import Any, Dict, List


class TimeSeries:
    """
    Class to represent a time series and to calculate some typical properties of it
    """

    # Algorithm hiperparameters
    A = 0.5
    MULTIPLIER = 1.96
    THRESHOLD = 10.0
    WINDOW = 30

    def __init__(self, values: List[int], dates: List[str]):
        self.values = values
        self.dates = dates

    def prediction_interval(
        self,
        a: float = A,
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
        mod_val = [*self.values]

        for i, (value, date) in enumerate(
            zip(mod_val[window:], self.dates[window:]), window
        ):
            # Calculate statistic values for this window
            mean = self._mean(mod_val[i - window : i])
            median = self._median(mod_val[i - window : i])
            std = self._std(mod_val[i - window : i])

            # Calculate bounds
            limit = max(threshold, median)
            upper_bound = mean + (multiplier * std)
            lower_bound = mean - (multiplier * std)
            if lower_bound and lower_bound < 0:
                lower_bound = 0.0

            # Modify original values
            if value > upper_bound and value > limit:
                mod_val[i] = value - a * (value - upper_bound)

            elif value < lower_bound:
                mod_val[i] = value + a * (lower_bound - value)

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
