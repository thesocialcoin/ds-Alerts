# Investigation


## Iteration 1


Condition to be a story:

There are 3 different functions used to determine an anomaly candidate in **Stories**
1. is_significant(self, ts: np.ndarray, threshold: int):
>This function checks whether the last value of the time series ts is greater than or equal to a certain threshold. np.ndarray is the time series data, and threshold is the value that the last data point of the time series should surpass to be considered significant. If the last data point of the time series is greater than or equal to the threshold, the function returns True, otherwise False.

2. is_increasing(self, ts: np.ndarray, min_length: int = STORY_DELTA):
>This function checks two things. First, whether the length of the time series ts is at least min_length. Second, it checks whether the time series is increasing, i.e., whether the last value in the time series is greater than the second-to-last value. If both conditions are met, the function returns True; otherwise, it returns False. STORY_DELTA is presumably a constant defined elsewhere in the code that acts as a default minimum length for the time series.

3. alert_th_mean_n_days(self, ts: np.ndarray, d: int = STORY_DELTA):
>This function defines an anomaly based on the last d days of the time series ts. It calculates the mean of the previous d days (excluding the current day) and then computes the percentage difference between the last day and the mean. If this difference exceeds a certain threshold, STORY_INCREASE_THRESHOLD (defined elsewhere in the code), the function will return True, indicating an anomaly. If there are no data points for the past d days, the mean is considered as 0, and the function will return False since the increasing rate would be undefined in this case.


**Functions:**

self.ts: This is the raw time series data for "volume". The time series data is converted into a NumPy array for efficient numerical computation. "Volume" seems to be the total number of records or "docs" per day or week, depending on the period type.

self.ts_n: This is the time series data for "normalized". The normalization process often involves scaling the data to fall within a certain range, which makes the data easier to process and less sensitive to extreme values. The "normalized" time series appears to represent the "normalized" number of "docs" per day or week, where "docs_norm" is scaled by the number of days in the week.

self.ts_n_dataset: This is the time series data for "normalized_dataset". The exact meaning of this variable would depend on what "docs_norm_dataset" represents in the context of your code, but it's safe to assume that this is another normalized version of the "docs" time series.

self.ts_scaled: This is the standardized version of the raw volume time series data self.ts. Standardization is a scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the data is 0 and the standard deviation is 1. Standardizing data can be useful when your data has varying scales and the algorithm you are using does make assumptions about your data having a Gaussian (bell curve) distribution, such as linear regression, logistic regression, and linear discriminant analysis. In this case, the data is standardized using the StandardScaler class from scikit-learn, which standardizes the features by removing the mean and scaling to unit variance.