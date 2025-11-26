import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from skmisc.loess import loess

def smooth_data_standard(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          x_column: str, 
                          y_column: str = "frame", 
                          smoothed_postfix: str = "_smoothed") -> pd.DataFrame:
    """
    Determines the growth rate of value_column over a specified integration window.

    Args:
        df (pd.DataFrame): track output dataframe
        integration_window (int): number of frames over which to smooth data
        id_column (str): entity identifier column (e.g., "trackID")
        x_column (str): column representing the metric to track (e.g., "area")
        y_column (str): column representing the y data across which we smooth each identifiers value_column. Defaults to "frame"
        growth_rate_column (str): name for the new growth rate column. Defaults to "growth_rate"
        centric (bool): if True, use a symmetric window around the current frame

    Returns:
        pd.DataFrame: dataframe with growth rate column added
    """
    df = df.copy()
    smoothed_column = x_column + smoothed_postfix
    df[smoothed_column] = np.nan
    smooth_window_halfwidth = integration_window // 2

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values(y_column)
        x = group[x_column].to_numpy()
        y = group[y_column].to_numpy()
        smoothed_values = np.full(len(x), np.nan)
        
        for i in range(x.shape[0]):
            idx0 = max(0, i - smooth_window_halfwidth)
            idx1 = min(group.shape[0], i + smooth_window_halfwidth)
            curr_idx = smooth_window_halfwidth + min(0, i - smooth_window_halfwidth)
            smoothed_values[i] = lowess(x[idx0:idx1], y[idx0:idx1], frac=1, return_sorted=False)[curr_idx] 
        df.loc[group.index, smoothed_column] = smoothed_values

    return df



def smooth_savgol(df: pd.DataFrame, 
                          integration_window: int, 
                          id_column: str, 
                          x_column: str, 
                          y_column: str = "frame", 
                          smoothed_postfix: str = "_smoothed",
                          poly_degree = 2) -> pd.DataFrame:

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


def smooth_data_fast(df: pd.DataFrame, 
                    integration_window: int, 
                    id_column: str, 
                    x_column: str, 
                    y_column: str = "frame", 
                    smoothed_postfix: str = "_smoothed"):

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


def smooth_loess(df: pd.DataFrame,
                               integration_window: int,
                               id_column: str,
                               x_column: str,
                               y_column: str = "frame",
                               smoothed_postfix: str = "_smoothed",
                               poly_degree = 2) -> pd.DataFrame:

    if integration_window < 5:
        raise ValueError("LOESS with degree=2 is numerically unstable below ~5 points.")

    df = df.copy()
    out_col = x_column + smoothed_postfix
    df[out_col] = np.nan

    half = integration_window // 2

    for track_id, group in df.groupby(id_column):
        group = group.sort_values(y_column)
        idx = group.index.values

        x = group[y_column].to_numpy()
        y = group[x_column].to_numpy()
        n = len(x)

        smoothed = np.empty(n)
        smoothed[:] = np.nan

        for i in range(n):
            start = max(0, i - half)
            end   = min(n, i + half)

            xs = x[start:end]
            ys = y[start:end]

            x0 = x[i]
            xs_n = xs - x0

            try:
                model = loess(xs_n, ys, degree=poly_degree, normalize=False)
                model.fit()
                pred = model.predict(np.array([0.0]), stderror=False)
                smoothed[i] = pred.values[0]
            except:
                smoothed[i] = np.nan

        df.loc[idx, out_col] = smoothed

    return df


def smooth_loess_optimized(df: pd.DataFrame,
                           integration_window: int,
                           id_column: str,
                           x_column: str,
                           y_column: str = "frame",
                           smoothed_postfix: str = "_smoothed",
                           poly_degree = 2) -> pd.DataFrame:
    """
    Optimized LOESS smoothing using scikit-misc (vectorized).
    Supports degree=2 and handles variable window sizes efficiently.
    """
    df = df.copy()
    out_col = x_column + smoothed_postfix
    df[out_col] = np.nan

    # Sort is critical for LOESS/Time-series
    df = df.sort_values([id_column, y_column])

    for track_id, group in df.groupby(id_column):
        x = group[y_column].to_numpy() # Time/Frame
        y = group[x_column].to_numpy() # Value
        n = len(x)

        span = integration_window / n
        span = min(1.0, span)

        try:
            model = loess(x, y, degree=poly_degree, span=span, normalize=False)
            model.fit()
            smoothed = model.predict(x, stderror=False).values
            df.loc[group.index, out_col] = smoothed
            
        except Exception as e:
            # Fallback for extremely numerically unstable segments (rare with skmisc)
            continue

    return df


def smooth_lowess_fast(df: pd.DataFrame,
                           integration_window: int,
                           id_column: str,
                           x_column: str,
                           y_column: str = "frame",
                           smoothed_postfix: str = "_smoothed") -> pd.DataFrame:
    df = df.copy()
    col = x_column + smoothed_postfix

    def process(group):
        group = group.sort_values(y_column)
        x = group[x_column].values
        y = group[y_column].values
        n = len(x)
        
        frac = integration_window / n
        frac = min(1.0, frac)

        sm = lowess(x, y, frac=frac, return_sorted=False)
        group[col] = sm
        return group

    return df.groupby(id_column, group_keys=False).apply(process)