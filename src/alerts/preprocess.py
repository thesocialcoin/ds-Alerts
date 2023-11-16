import pandas as pd


def data2timeSeries(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the daily frequency of occurrences
        for each feature in the given DataFrame.

    This function assumes that the input DataFrame contains a "date" column.
        The "date" column
    is converted to a datetime format and set as the index of a new DataFrame.
    Subsequently, each feature is resampled on a daily frequency,
    and the total count of occurrences is computed,
    generating a DataFrame representing the daily frequency of each feature.

    Args:
        data (pd.DataFrame): A DataFrame containing
            features and a "date" column.

    Returns:
        pd.DataFrame: A new DataFrame with the date set
            as the index and features resampled
            on a daily frequency, indicating the
            daily count of occurrences for each feature.
    """
    assert "date" in data.columns

    ts = data.copy()
    ts.loc[:, "date"] = pd.to_datetime(data.loc[:, "date"])
    ts = ts.set_index("date") \
        .resample("D") \
        .count()

    return ts
