import pandas as pd
from typing import List, Literal, Union


def grouped_rank(
    df: pd.DataFrame,
    group_columns: Union[str, List[str]] = ("frame", "chamber"),
    value_column: str = "x",
    rank_column: str = "rank",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank rows within each group based on a value column.

    Args:
        df (pd.DataFrame): input tracking DataFrame
        group_columns (str or list[str]): columns to group by. Defaults to ("frame", "chamber")
        value_column (str): column to rank within each group. Defaults to "x"
        rank_column (str): name of the output rank column. Defaults to "rank"
        ascending (bool): if False, rank 1 = highest value; if True, rank 1 = lowest value.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with an integer rank column added
    """
    cols = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    df = df.copy()
    df[rank_column] = (
        df.groupby(cols)[value_column]
        .rank(method="first", ascending=ascending)
        .astype(int)
    )
    return df


def grouped_metric(
    df: pd.DataFrame,
    group_columns: Union[str, List[str]] = ("frame", "chamber"),
    value_column: str = "y",
    mode: Literal["min", "max", "mean", "median", "sd"] = "mean",
    metric_column: str = None,
) -> pd.DataFrame:
    """
    Compute a summary metric per group and broadcast it as a new column.

    Args:
        df (pd.DataFrame): input tracking DataFrame
        group_columns (str or list[str]): columns to group by. Defaults to ("frame", "chamber")
        value_column (str): column to aggregate. Defaults to "y"
        mode (str): aggregation type — one of "min", "max", "mean", "median", "sd".
            Defaults to "mean"
        metric_column (str, optional): name of the output column.
            Defaults to f"{value_column}_{mode}"

    Returns:
        pd.DataFrame: DataFrame with the metric column added
    """
    _agg_map: dict = {
        "min": "min",
        "max": "max",
        "mean": "mean",
        "median": "median",
        "sd": "std",
    }
    if mode not in _agg_map:
        raise ValueError(f"mode must be one of {list(_agg_map)}, got '{mode}'")

    if metric_column is None:
        metric_column = f"{value_column}_{mode}"

    cols = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    df = df.copy()
    df[metric_column] = df.groupby(cols)[value_column].transform(_agg_map[mode])
    return df
