import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_growth_rate(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          value_column: str, 
                          frame_column: str = "frame", 
                          growth_rate_column: str = "growth_rate",
                          centric: bool = False) -> pd.DataFrame:
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
                half_window = integration_window // 2
                start = i - half_window
                end = i + half_window + 1
                if start >= 0 and end <= len(values):
                    deltas = np.diff(values[start:end])
                    growth_rates[i] = np.mean(deltas)
            else:
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
        growth_rates = np.zeros(len(values))
        r_squared = np.zeros(len(values))

        for i in range(len(values)):
            if i + integration_window < len(values):
                future_values = values[i + 1:i + 1 + integration_window]
                future_frames = frames[i + 1:i + 1 + integration_window].reshape(-1, 1)

                # Linear regression for R-squared calculation
                model = LinearRegression()
                model.fit(future_frames, future_values)
                slope = model.coef_[0]
                r2 = model.score(future_frames, future_values)

                growth_rates[i] = slope
                r_squared[i] = r2
            else:
                growth_rates[i] = np.nan
                r_squared[i] = np.nan

        df.loc[group.index, growth_rate_column] = growth_rates
        df.loc[group.index, r_squared_column] = r_squared

    return df