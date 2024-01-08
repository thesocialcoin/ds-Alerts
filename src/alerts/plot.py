import matplotlib.pyplot as plt
import pandas as pd
from alerts.anomaly import AnomalyDetector
from alerts.anomaly import AnomalyTS


def plot(ts: AnomalyTS,
         show_anomalies=False,
         title: str = "Time series & prediction intervals") -> None:

    plt.subplots(figsize=(12, 4))

    intervals = ts.prediction_interval()

    df = pd.DataFrame([p.__dict__ for p in intervals])

    plt.plot(df["date"], df["value"], label="value")
    plt.plot(df["date"], df["lower_bound"], label="lower_bound")
    plt.plot(df["date"], df["upper_bound"], label="upper_bound")

    if show_anomalies:
        detector = AnomalyDetector()
        events = detector.detect_alerts(ts)

        x = [e.date for e in events]
        y = [e.value for e in events]

        plt.scatter(x, y,
                    marker="o",
                    alpha=0.5,
                    color="red")

    plt.xlabel("Days")
    plt.ylabel("Magnitude")
    plt.title(title + (
        " - Anomalies {}".format(len(x))
        if show_anomalies else ""
    ))

    plt.legend()
    plt.show()
