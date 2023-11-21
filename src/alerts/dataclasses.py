from dataclasses import dataclass
from typing import List


@dataclass
class LimitInterval:
    def __init__(self, lb: float, ub: float):
        self.lower_bound = lb
        self.upper_bound = ub


@dataclass
class PredictionInterval:
    def __init__(self, t, v: float, lb: float, ub: float):
        self.time = t
        self.value = v
        self.lower_bound = lb
        self.upper_bound = ub


@dataclass
class Event:
    def __init__(self, t, v):
        self.time = t
        self.value = v


@dataclass
class TimeSeries:

    def __init__(self, events: List[Event]):
        self.events = events
