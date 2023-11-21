# ################ Code Implemented by Alejandro Bonell ################
# Inspired by: Anomaly detection in univariate time series incorporating
# active learning.
# https://www.sciencedirect.com/science/article/pii/S2772415822000323

from typing import List
import numpy as np
from alerts.dataclasses import (TimeSeries,
                                PredictionInterval,
                                Event,
                                LimitInterval)

from abc import abstractmethod
from abc import ABC


def to_time_series(ds, vs) -> TimeSeries:

    if len(ds) != len(vs):
        raise Exception(
            ("To create a time series, "
             "bouth iterables must have the same lenght")
        )

    events = [Event(d, vs[i]) for i, d in enumerate(ds)]
    time_series = TimeSeries(events)

    return time_series


class AnomalyTS(ABC):

    @abstractmethod
    def prediction_interval() -> List[PredictionInterval]:
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
        self.ts = ts
        if flat_ts:
            self.ts = TimeSeries(
                [Event(event.time, np.sqrt(event.value))
                 for event in self.ts.events]
            )

    def _compute_limits(
            self,
            current_window_frame: List[Event]) -> LimitInterval:

        values = [e.value for e in current_window_frame]
        mean_value = np.mean(values)
        std_value = np.std(values)
        lower_bound = mean_value - self.z_score * std_value
        upper_bound = mean_value + self.z_score * std_value

        limits = LimitInterval(
            lb=0 if lower_bound < 0 else lower_bound,
            ub=upper_bound
        )

        return limits

    def prediction_interval(self) -> List[PredictionInterval]:

        intervals = self._init_prediction_interval()
        future_events = self.ts.events[self.window:]

        for i, event in enumerate(future_events):
            current_window_frame = self.ts.events[i: self.window + i]
            limits = self._compute_limits(current_window_frame)

            interval = PredictionInterval(
                t=event.time,
                v=event.value,
                lb=limits.lower_bound,
                ub=limits.upper_bound
            )

            intervals.append(interval)

        return intervals

    def _init_prediction_interval(self) -> List[PredictionInterval]:
        """Init the prediction intervals

        Returns:
            List[PredictionInterval]: The list contains
                the events pre window frame
        """
        intervals = list()
        for event in self.ts.events[:self.window]:
            intervals.append(
                PredictionInterval(
                    t=event.time,
                    v=event.value,
                    lb=None,
                    ub=None
                )
            )
        return intervals


class AnomalyMedianTS(AnomalyTS):

    MAD_FACTOR = 1.4826
    Z_SCORE = 1.625
    WINDOW = 30

    def __init__(self,
                 ts: TimeSeries,
                 z_score: float = Z_SCORE,
                 mad_factor: float = MAD_FACTOR,
                 window: int = WINDOW,
                 flat_ts: bool = False):
        self.z_score = z_score
        self.window = window
        self.mad_factor = mad_factor
        self.ts = ts
        if flat_ts:
            self.ts = TimeSeries(
                [Event(event.time, np.sqrt(event.value))
                 for event in self.ts.events]
            )

    def _compute_limits(
            self,
            current_window_frame: List[Event]) -> LimitInterval:

        values = [e.value for e in current_window_frame]
        median_value = np.median(values)
        mad_value = self._mad(values)

        # Calculate bounds
        lower_bound = median_value - (self.z_score * mad_value)
        upper_bound = median_value + (self.z_score * mad_value)

        limits = LimitInterval(
            lb=0 if lower_bound < 0 else lower_bound,
            ub=upper_bound
        )

        return limits

    def _mad(self, values: List[float]) -> float:
        median = np.median(values)
        mad = np.median([abs(v - median) for v in values]) * self.mad_factor
        return mad

    def prediction_interval(self) -> List[PredictionInterval]:

        intervals = self._init_prediction_interval()
        future_events = self.ts.events[self.window:]

        for i, event in enumerate(future_events):
            current_window_frame = self.ts.events[i: self.window + i]
            limits = self._compute_limits(current_window_frame)

            interval = PredictionInterval(
                t=event.time,
                v=event.value,
                lb=limits.lower_bound,
                ub=limits.upper_bound
            )

            intervals.append(interval)

        return intervals

    def _init_prediction_interval(self) -> List[PredictionInterval]:
        """Init the prediction intervals

        Returns:
            List[PredictionInterval]: The list contains
                the events pre window frame
        """
        intervals = list()
        for event in self.ts.events[:self.window]:
            intervals.append(
                PredictionInterval(
                    t=event.time,
                    v=event.value,
                    lb=None,
                    ub=None
                )
            )
        return intervals
