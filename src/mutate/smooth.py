import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from skmisc.loess import loess


def smooth_svagol(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          x_column: str, 
                          y_column: str = "frame", 
                          smoothed_postfix: str = "_smoothed",
                          poly_degree = 2) -> pd.DataFrame:
    """
    Method that uses Savitzky-Golay filter to smooth the data. Very fast polynomal fit within time window. Assumes that y_column (i.e frames) are equaly spaced!
    
    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to smooth data
        id_column (str): entity identifier column (e.g., "trackID")
        x_column (str): column representing the metric to track (e.g., "area")
        y_column (str): column representing the y data across which we smooth each identifiers value_column. Defaults to "frame"
        smoothed_postfix (str): value of the post fix added to x_column name to define the name of the new column with smoothed data. defaults to "_smoothed"
        poly_degree (int): what order polygon should be fitted? defaults to 2

    Returns:
        pd.DataFrame: dataframe with growth rate column added
    """
    df = df.copy()
    col = x_column + smoothed_postfix

    def process(group):
        group = group.sort_values(y_column)
        x = group[x_column].values
        w = min(integration_window, len(x))
        smoothed = savgol_filter(x, window_length=w, polyorder=poly_degree, mode="interp")
        group[col] = smoothed
        return group

    return df.groupby(id_column, group_keys=False).apply(process)


def smooth_linear(df: pd.DataFrame, 
                    integration_window: int, 
                    id_column: str, 
                    x_column: str, 
                    y_column: str = "frame", 
                    smoothed_postfix: str = "_smoothed"):
    """
    Method that uses linear filter (np.convolution) to correct the data. Comparable to a rolling window average.
    
    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to smooth data
        id_column (str): entity identifier column (e.g., "trackID")
        x_column (str): column representing the metric to track (e.g., "area")
        y_column (str): column representing the y data across which we smooth each identifiers value_column. Defaults to "frame"
        smoothed_postfix (str): value of the post fix added to x_column name to define the name of the new column with smoothed data. defaults to "_smoothed"

    Returns:
        pd.DataFrame: dataframe with growth rate column added
    """
    df = df.copy()
    col_out = x_column + smoothed_postfix

    half = integration_window // 2
    window = np.ones(integration_window) / integration_window

    def process(group):
        group = group.sort_values(y_column)
        x = group[x_column].to_numpy()
        padded = np.pad(x, (half, half), mode='edge')
        smoothed = np.convolve(padded, window, mode='valid')
        group[col_out] = smoothed
        return group

    return df.groupby(id_column, group_keys=False).apply(process)


def smooth_loess_optimized(df: pd.DataFrame,
                           integration_window: int,
                           id_column: str,
                           x_column: str,
                           y_column: str = "frame",
                           smoothed_postfix: str = "_smoothed",
                           poly_degree = 2) -> pd.DataFrame:
    """
    Optimized LOESS smoothing using scikit-misc (vectorized implementation). Supports degree >= 2 
    
    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to smooth data
        id_column (str): entity identifier column (e.g., "trackID")
        x_column (str): column representing the metric to track (e.g., "area")
        y_column (str): column representing the y data across which we smooth each identifiers value_column. Defaults to "frame"
        smoothed_postfix (str): value of the post fix added to x_column name to define the name of the new column with smoothed data. defaults to "_smoothed"
        poly_degree (int): what order polygon should be fitted? defaults to 2

    Returns:
        pd.DataFrame: dataframe with growth rate column added
    """
    df = df.copy()
    out_col = x_column + smoothed_postfix
    df[out_col] = np.nan

    df = df.sort_values([id_column, y_column])

    for track_id, group in df.groupby(id_column):
        x = group[x_column].to_numpy()
        y = group[y_column].to_numpy()
        n = len(x)

        span = integration_window / n #best approximation to a window size we can get from the vectorized versions
        span = min(1.0, span)

        try:
            model = loess(y, x, degree=poly_degree, span=span)  #note, y and x are flipped in loess implementation!
            model.fit()
            smoothed = model.predict(y, stderror=False).values
            df.loc[group.index, out_col] = smoothed
            
        except Exception as e:
            continue

    return df


def smooth_lowess_fast(df: pd.DataFrame,
                           integration_window: int,
                           id_column: str,
                           x_column: str,
                           y_column: str = "frame",
                           smoothed_postfix: str = "_smoothed") -> pd.DataFrame:
    """
    Optimized LOWESS smoothing using statsmodels version (vectorized implementation). Works with first order degree.
    
    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to smooth data
        id_column (str): entity identifier column (e.g., "trackID")
        x_column (str): column representing the metric to track (e.g., "area")
        y_column (str): column representing the y data across which we smooth each identifiers value_column. Defaults to "frame"
        smoothed_postfix (str): value of the post fix added to x_column name to define the name of the new column with smoothed data. defaults to "_smoothed"

    Returns:
        pd.DataFrame: dataframe with growth rate column added
    """
    df = df.copy()
    col = x_column + smoothed_postfix

    def process(group):
        group = group.sort_values(y_column)
        x = group[x_column].values
        y = group[y_column].values
        n = len(x)
        
        frac = integration_window / n #best approximation to a window size we can get from the vectorized versions.
        frac = min(1.0, frac)

        sm = lowess(x, y, frac=frac, return_sorted=False)
        group[col] = sm
        return group

    return df.groupby(id_column, group_keys=False).apply(process)