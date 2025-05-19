import pandas as pd


def compute_global_axes(data_list, x_column, y_column):
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    for k, df in data_list.items():
        x_min = min(x_min, df[x_column].dropna().min())
        x_max = max(x_max, df[x_column].dropna().max())
        y_min = min(y_min, df[y_column].dropna().min())
        y_max = max(y_max, df[y_column].dropna().max())
        
    return x_min, x_max, y_min, y_max


def collect_unique_column_values(nested_dict, column_name, seen_values=None):
    """
    Recursively traverse a nested dictionary structure and collect all unique values
    for a given column from pandas DataFrames at the leaves.

    Parameters:
    - nested_dict: dict
        Nested dictionary with DataFrames at the leaves.
    - column_name: str
        The column name to extract values from.
    - seen_values: set
        Internal use for recursive value collection.

    Returns:
    - set
        Unique values found in the specified column.
    """
    if seen_values is None:
        seen_values = set()

    if isinstance(nested_dict, dict):
        for value in nested_dict.values():
            collect_unique_column_values(value, column_name, seen_values)
    elif isinstance(nested_dict, pd.DataFrame):
        if column_name in nested_dict.columns:
            seen_values.update(nested_dict[column_name].dropna().unique())
        else:
            # Column doesn't exist in this DataFrame
            pass  # optionally raise or log a warning here

    return sorted(seen_values)