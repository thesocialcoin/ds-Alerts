from dataclasses import dataclass
from typing import List

@dataclass
class LimitInterval:
    def __init__(self, lb, ub):
        self.lower_bound = lb
        self.upper_bound = ub


@dataclass
class PredictionInterval:
    def __init__(self, t, v, lb, ub):
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

    MAD_FACTOR = 1.4826
    Z_SCORE = 1.625
    WINDOW = 30

    def __init__(self, events: List[Event]):

        self.events = events
        self.dates = [e.time for e in self.events]
        self.values = [e.value for e in self.events]

        pass
