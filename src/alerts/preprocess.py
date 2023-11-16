import pandas as pd


def data2timeSeries(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[:, "date"] = pd.to_datetime(data.loc[:, "date"])

    ts = data.set_index("date").resample("D").count()
    data = data.reset_index()
    return ts
