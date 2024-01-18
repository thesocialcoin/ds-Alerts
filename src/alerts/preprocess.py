import pandas as pd


def data2timeseries(data: pd.DataFrame) -> pd.DataFrame:
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
    assert "text" in data.columns

    ts = data.copy()
    ts["date"] = ts["date"].apply(pd.to_datetime)
    ts = ts.set_index("date") \
        .resample("D") \
        .count()

    agg_ts = pd.DataFrame(columns=["date", "timestamp", "value"])
    agg_ts["value"] = ts["text"].values
    agg_ts["date"] = ts.index.values
    agg_ts["timestamp"] = agg_ts["date"].astype(int)
    agg_ts["timestamp"] = agg_ts["timestamp"].div(10**9)

    return agg_ts