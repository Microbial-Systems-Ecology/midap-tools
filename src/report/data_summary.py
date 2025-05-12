import pandas as pd

def data_summary(series: pd.Series) -> dict:
    """
    Generates a summary of the specified pd.Series

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to summarize.

    Returns:
        dict: Summary dictionary containing statistics.
    """
    if series.empty:
        return {}
    
    quantiles = series.quantile([0.25, 0.5, 0.75])

    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std_dev": series.std(),
        "min": series.min(),
        "max": series.max(),
        "25th_percentile": quantiles[0.25],
        "50th_percentile": quantiles[0.5],
        "75th_percentile": quantiles[0.75],
        "unique_values": series.nunique(),
    }
    
    return summary