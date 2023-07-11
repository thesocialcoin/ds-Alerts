import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import date, timedelta


GROUP_KEYS = ["project_dataset_id", "category_id", "segment", "intent"]
STORY_DELTA = 8  # number of previous weeks/days we will compare the current date with
STORY_STD_MARGIN = 2  # multiplicity of STORY_DELTA to avoid standardization issues
STORY_INCREASE_THRESHOLD = 50  # threshold to determine if an increase is a story
STORY_VOLUME_THRESHOLD = 50  # threshold to determine if there are enough documents to have a story
STORY_NORM_DATASET_THRESHOLD = 0.5  # threshold to determine if, relative to the dataset, there are enough documents
MAX_N_OUTPUT_TERMS = 10

# To calculate trends, it will use a maximum of 1000 terms or
# N_COMPUTATION_TERMS_RATIO * number of total documents
N_COMPUTATION_TERMS_RATIO = 0.5
MAX_COMPUTATION_TERMS = 1000


class NotAnomalyDetected(Exception):
    """Raised when no anomaly detected"""
    pass

class AnomalyTS:
    None

class Anomaly:
    def __init__(self, time_series):
        self.time_series = time_series
        self.ts = np.array(time_series.get("volume"))
        self.ts_scaled = []

    def get_baseline(self):
        """
        Objective: from time_series, identify the anomaly and convert it in a Story
        """

        if len(self.ts) > 1:
            scaler = StandardScaler()
            # we fit scaler on the values except the last one that we transform according to the fitted scaler
            self.ts_scaled = scaler.fit_transform(self.ts[:-1].reshape(-1, 1)).reshape(-1)
            self.ts_scaled = np.concatenate((self.ts_scaled, scaler.transform(self.ts[-1].reshape(-1, 1)).reshape(-1)))

            if self.is_candidate_an_anomaly():
                #self.define_anomaly()
                #return self.story
                return 'is an anomaly'
            else:
                raise NotAnomalyDetected()
        else:
            raise NotAnomalyDetected()

    def is_candidate_an_anomaly(self):
        return (
            self.is_significant(self.ts, STORY_VOLUME_THRESHOLD)
            #and self.is_significant(self.ts_n_dataset, STORY_NORM_DATASET_THRESHOLD)
            and self.is_increasing(self.ts_scaled)
            #and self.is_increasing(self.ts_n)
            and self.alert_th_mean_n_days(self.ts)
            and self.alert_th_mean_n_days(self.ts_scaled)
            #and self.alert_th_mean_n_days(self.ts_n)
        )

    def is_significant(self, ts: np.ndarray, threshold: int):
        return ts[-1] >= threshold

    def is_increasing(self, ts: np.ndarray, min_length: int = STORY_DELTA):
        """
        Objective: getting all the series. If the series does not grow, or has less than d data, we get rid of it.
        """
        return len(ts) >= min_length and ts[-1] > ts[-2]

    def alert_th_mean_n_days(self, ts: np.ndarray, d: int = STORY_DELTA):
        """
        Objective: define anomaly according to last d days
                   if the increasing rate is higher than a threshold th we flag otherwise we don't
        """
        mean = np.mean(ts[-d:-1]) if len(ts[-d:-1]) > 0 else 0
        return mean != 0 and ((ts[-1] - mean) / abs(mean)) * 100 > STORY_INCREASE_THRESHOLD

