from dataclasses import dataclass
from typing import List, Any


@dataclass
class LimitInterval:
    def __init__(self, lb: float, ub: float):
        self.lower_bound = lb
        self.upper_bound = ub


@dataclass
class PredictionInterval:
    def __init__(self, t, v: float, lb: float, ub: float):
        self.date = t
        self.value = v
        self.lower_bound = lb
        self.upper_bound = ub


@dataclass
class Event:
    def __init__(self, t, v: float):
        self.date = t
        self.value = v


@dataclass
class TimeSeries:
    def to_events(self, dates: List[Any], values: List[float]) -> List[Event]:
        if len(dates) != len(values):
            raise Exception(
                (
                    "To create a time series, "
                    "bouth iterables must have the same lenght"
                )
            )

        events = [Event(d, values[i]) for i, d in enumerate(dates)]

        return events

    def __init__(self, dates: List[Any], values: List[float]):
        events = self.to_events(dates, values)
        self.events = events
