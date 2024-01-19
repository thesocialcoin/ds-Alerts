from typing import List
import pandas as pd


def dataframe_segmentation(df: pd.DataFrame, split_columns: List[str]):
    """Function to split dataframe grouping by the different
    values defined in the colection of columns to split

    Args:
        df (pd.DataFrame)
        split_columns (List[str])
    """

    # Identify unique values in specified columns
    unique_values = df[split_columns].drop_duplicates()

    # Create a dictionary to store split DataFrames
    split_dataframes = {}

    # Iterate over unique values and create separate DataFrames
    for _, row in unique_values.iterrows():
        # Generate the mask dynamically based on the values in the specified columns
        mask = (df[split_columns] == row[split_columns]).all(axis=1)
        split_dataframes["_".join(f"{col}_{row[col]}" for col in split_columns)] = df[
            mask
        ].copy()

    return split_dataframes
