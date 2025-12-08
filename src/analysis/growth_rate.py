import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_growth_rate(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          value_column: str, 
                          frame_column: str = "frame", 
                          growth_rate_column: str = "growth_rate",
                          centric: bool = True) -> pd.DataFrame:
    """
    Determines the growth rate of value_column over a specified integration window.

    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to calculate growth
        id_column (str): entity identifier column (e.g., "trackID")
        value_column (str): column representing the metric to track (e.g., "area")
        frame_column (str): column representing frame index. Defaults to "frame"
        growth_rate_column (str): name for the new growth rate column. Defaults to "growth_rate"
        centric (bool): if True, use a symmetric window around the current frame

    Returns:
        pd.DataFrame: dataframe with growth rate column added
    """
    df = df.copy()
    df[growth_rate_column] = 0.0

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values(frame_column)
        values = group[value_column].to_numpy()
        growth_rates = np.full(len(values), np.nan)

        for i in range(len(values)):
            if centric:
            # symmetric window around each index
                half_window = integration_window // 2
                start = i - half_window
                end = i + half_window + 1
                if start >= 0 and end <= len(values):
                    deltas = np.diff(values[start:end])
                    growth_rates[i] = np.mean(deltas)
            else:
            # forward-looking window
                if i + integration_window < len(values):
                    future_deltas = np.subtract(values[i + 1:i + 1 + integration_window], values[i:i + integration_window])
                    growth_rates[i] = np.mean(future_deltas)

        df.loc[group.index, growth_rate_column] = growth_rates

    return df

def calculate_growth_rate_r2(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          value_column: str, 
                          frame_column: str = "frame", 
                          growth_rate_column: str = "growth_rate",
                          centric: bool = False,
                          r_squared_column: str = "growth_rsquared") -> pd.DataFrame:
    """
    Determines the growth rate and R-squared of value_column over a specified integration window. Warning, this function is considerably slower due to r2 calculations.

    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to calculate growth and R-squared
        id_column (str): entity identifier column (e.g., "trackID")
        value_column (str): column representing the metric to track (e.g., "area")
        frame_column (str): column representing frame index. Defaults to "frame"
        growth_rate_column (str): name for the new growth rate column. Defaults to "growth_rate"
        r_squared_column (str): name for the new R-squared column. Defaults to "growth_rsquared"

    Returns:
        pd.DataFrame: dataframe with growth rate and R-squared columns added
    """
    df = df.copy()
    df[growth_rate_column] = 0.0
    df[r_squared_column] = 0.0

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values(frame_column)
        values = group[value_column].to_numpy()
        frames = group[frame_column].to_numpy()
        n = len(values)
        growth_rates = np.zeros(n)
        r_squared = np.zeros(n)

        if centric:
            # symmetric window around each index
            half = integration_window // 2

            for i in range(n):
                start = i - half
                end = i + half + 1
                if start < 0 or end > n:
                    continue
                fv = values[start:end]
                ff = frames[start:end].reshape(-1, 1)
                model = LinearRegression()
                model.fit(ff, fv)
                growth_rates[i] = model.coef_[0]
                r_squared[i] = model.score(ff, fv)

        else:
            # forward-looking window
            for i in range(n):
                end = i + integration_window
                if end > n:
                    continue
                fv = values[i:end]
                ff = frames[i:end].reshape(-1, 1)
                model = LinearRegression()
                model.fit(ff, fv)
                growth_rates[i] = model.coef_[0]
                r_squared[i] = model.score(ff, fv)

        df.loc[group.index, growth_rate_column] = growth_rates
        df.loc[group.index, r_squared_column] = r_squared

    return df

def growth_rate_polyfit(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          value_column: str, 
                          frame_column: str = "frame", 
                          growth_rate_column: str = "growth_rate",
                          centric: bool = True) -> pd.DataFrame:
    """
    Determines the growth rate and R-squared of value_column over a specified integration window. Warning, this function is considerably slower due to r2 calculations.

    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to calculate growth and R-squared
        id_column (str): entity identifier column (e.g., "trackID")
        value_column (str): column representing the metric to track (e.g., "area")
        frame_column (str): column representing frame index. Defaults to "frame"
        growth_rate_column (str): name for the new growth rate column. Defaults to "growth_rate"

    Returns:
        pd.DataFrame: dataframe with growth rate and R-squared columns added
    """
    df = df.copy()
    df[growth_rate_column] = 0.0

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values(frame_column)
        values = group[value_column].to_numpy()
        frames = group[frame_column].to_numpy()
        n = len(values)
        growth_rates = np.full(n, np.nan)

        if centric:
            # symmetric window around each index
            half = integration_window // 2
            for i in range(n):
                start = i - half
                end = i + half + 1
                if start < 0 or end > n:
                    continue

                f = frames[start:end]
                v = values[start:end]
                if len(f) >= 2:
                    p = np.polyfit(f, v, 1)
                    growth_rates[i] = p[0]
        else:
            # forward-looking window
            for i in range(n):
                end = i + integration_window
                if end > n:
                    continue
                f = frames[i:end]
                v = values[i:end]
                if len(f) >= 2:
                    p = np.polyfit(f, v, 1)
                    growth_rates[i] = p[0]

        df.loc[group.index, growth_rate_column] = growth_rates

    return df


def calculate_min_max_deltas(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          value_column: str, 
                          frame_column: str = "frame", 
                          growth_rate_column: str = "delta_",
                          centric: bool = False) -> pd.DataFrame:
    """
    Determines the min and max delta within id_column and an integration window 

    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to calculate growth
        id_column (str): entity identifier column (e.g., "trackID")
        value_column (str): column representing the metric to track (e.g., "area")
        frame_column (str): column representing frame index. Defaults to "frame"
        growth_rate_column (str): prefix for the min and max colums. Defaults to "delta_"
        centric (bool): if True, use a symmetric window around the current frame

    Returns:
        pd.DataFrame: dataframe with min and max delta columns added
    """
    df = df.copy()
    max_column = growth_rate_column + "max"
    min_column = growth_rate_column + "min"
    df[max_column] = 0.0
    df[min_column] = 0.0

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values(frame_column)
        values = group[value_column].to_numpy()
        mins = np.full(len(values), np.nan)
        maxes = np.full(len(values), np.nan)

        for i in range(len(values)):
            if centric:
                half_window = integration_window // 2
                start = i - half_window
                end = i + half_window + 1
                if start >= 0 and end <= len(values):
                    deltas = np.diff(values[start:end])
                    mins[i] = np.min(deltas)
                    maxes[i] = np.max(deltas)
            else:
                if i + integration_window < len(values):
                    future_deltas = np.subtract(values[i + 1:i + 1 + integration_window], values[i:i + integration_window])
                    mins[i] = np.min(future_deltas)
                    maxes[i] = np.max(future_deltas)

        df.loc[group.index, max_column] = maxes
        df.loc[group.index, min_column] = mins

    return df

def calculate_min_max_deltas_nona(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          value_column: str, 
                          frame_column: str = "frame", 
                          growth_rate_column: str = "delta_",
                          centric: bool = False) -> pd.DataFrame:
    """
    Determines the min and max delta within id_column and an integration window.
    This version of the function will create no Nans 
    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to calculate growth
        id_column (str): entity identifier column (e.g., "trackID")
        value_column (str): column representing the metric to track (e.g., "area")
        frame_column (str): column representing frame index. Defaults to "frame"
        growth_rate_column (str): prefix for the min and max colums. Defaults to "delta_"
        centric (bool): if True, use a symmetric window around the current frame

    Returns:
        pd.DataFrame: dataframe with min and max delta columns added
    """
    df = df.copy()
    max_column = growth_rate_column + "max"
    min_column = growth_rate_column + "min"
    df[max_column] = 0.0
    df[min_column] = 0.0

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values(frame_column)
        values = group[value_column].to_numpy()
        mins = np.full(len(values), np.nan)
        maxes = np.full(len(values), np.nan)

        for i in range(len(values)):
            if centric:
                half_window = integration_window // 2
                start = i - half_window
                end = i + half_window + 1
                start = max(start, 0)
                end = min(end, len(values))
                deltas = np.diff(values[start:end])
                mins[i] = np.min(deltas)
                maxes[i] = np.max(deltas)
            else:
                if i + integration_window < len(values):
                    future_deltas = np.subtract(values[i + 1:i + 1 + integration_window], values[i:i + integration_window])
                    mins[i] = np.min(future_deltas)
                    maxes[i] = np.max(future_deltas)

        df.loc[group.index, max_column] = maxes
        df.loc[group.index, min_column] = mins

    return df
