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
    MULTIPLIER = 1.625
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