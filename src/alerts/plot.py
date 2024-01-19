import matplotlib.pyplot as plt
import pandas as pd
from alerts.anomaly import AnomalyDetector
from alerts.anomaly import AnomalyTS


def plot(
    ts: AnomalyTS,
    show_anomalies=False,
    title: str = "Time series & prediction intervals",
) -> None:
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

        plt.scatter(x, y, marker="o", alpha=0.5, color="red")

    plt.xlabel("Days")
    plt.ylabel("Magnitude")
    plt.title(title + (" - Anomalies {}".format(len(x)) if show_anomalies else ""))

    plt.legend()
    plt.show()


def plot_groups(
    ts: AnomalyTS,
    show_anomalies=False,
    title: str = "Time series & prediction intervals",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    intervals = ts.prediction_interval()
    df = pd.DataFrame([p.__dict__ for p in intervals])

    ax.plot(df["date"], df["value"], label="value")
    ax.plot(df["date"], df["lower_bound"], label="lower_bound")
    ax.plot(df["date"], df["upper_bound"], label="upper_bound")

    if show_anomalies:
        detector = AnomalyDetector()
        events = detector.detect_alerts_groups(ts)

        x = [evs.date for e in events for evs in e]
        y = [evs.value for e in events for evs in e]

        start_idx = 0
        for evs in events:
            length = len(evs)
            end_idx = start_idx + length

            ax.scatter(
                x[start_idx:end_idx],
                y[start_idx:end_idx],
                marker="o",
                alpha=0.5,
                color="red",
            )
            ax.axvspan(x[start_idx], x[end_idx - 1], color="blue", alpha=0.2, lw=0)

            start_idx = end_idx

    ax.set_xlabel("Days")
    ax.set_ylabel("Magnitude")
    ax.set_title(title + (" - Anomalies {}".format(len(x)) if show_anomalies else ""))

    ax.legend()
    plt.show()
