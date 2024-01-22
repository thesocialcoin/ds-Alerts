from alerts.anomaly import AnomalyDetector
from alerts.anomaly import AnomalyTS
from typing import List
import numpy as np


def maximaxer(ts: AnomalyTS, others: List[AnomalyTS]) -> None:
    for other_ts in others:
        detector = AnomalyDetector()
        group_events = detector.detect_alerts_groups(other_ts)
        group_events = [group for group in group_events if len(group) >= 1]

        dates = [event.date for event in other_ts.ts.events]
        # steps = np.arange(0, len(dates), dtype=int)

        for group in group_events:
            start_date = group[0].date
            end_date = group[-1].date
            matches = np.where(np.array(dates) >= start_date, dates <= end_date, dates)
            print(matches)
