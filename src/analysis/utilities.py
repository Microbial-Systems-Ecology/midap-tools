import os
import glob
import numpy as np
import pandas as pd
import h5py
import cv2
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import KDTree
from typing import Union, List
from scipy import stats
from scipy.stats import linregress
import statsmodels.formula.api as smf



def load_tracking_data(path, group):
    res_path = os.path.join(path,group, "track_output","*.csv")
    tracking_file = glob.glob(res_path)[0]
    data = pd.read_csv(tracking_file)
    return data

def sort_folder_names(folders):
    def extract_number(name):
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else float('inf')  # Push non-numbered names to end

    return sorted(folders, key=extract_number)

def load_segmentations_h5(path, group, binary = True):
    res_path = os.path.join(path,group, "track_output","segmentations*.h5")
    tracking_file = glob.glob(res_path)[0]
    with h5py.File(tracking_file, 'r') as f:
        data = f["segmentations"][:]
    if binary:
        data = data > 0
        
    return data

def filter_tracks(df : pd.DataFrame, column = "trackID", min_occurences = 2) -> pd.DataFrame:
    """
    Filters a track output file (loaded as pd.DataFrame) for low occurence 
    Args:
        df (pd.DataFrame): a track output dataframe
        column (str, optional): the column to filter. Defaults to "TrackID".
        min_occurences (int, optional): the number of minimal occurences of the entitiy in column to be retained. Defaults to 2.

    Returns:
        pd.DataFrame: filtered track output file
    """
    df = df.copy()
    counts = df[column].value_counts()
    valid_ids = counts[counts >= min_occurences].index
    return df[df[column].isin(valid_ids)].reset_index(drop=True)
    
def calculate_growth_rate(df: pd.DataFrame, id_column = "trackID", value_column = "major_axis_length", integration_window = 5) -> pd.DataFrame:
    """
    from a track output file, determines the growth rate of entities in value_column based on value of value_column over a integration window

    Args:
        df (pd.DataFrame): track output dataframe
        id_column (str, optional): which column will be used to determine what the entities are?. Defaults to "trackID".
        value_column (str, optional): which value will be used to determine the size of the cells. Defaults to "major_axis_length".
        integration_window (int, optional): over how many frames should the data be integrated. Defaults to 5.

    Returns:
        pd.DataFrame: a track output dataframe with the growth rate column added at each frame
    """
    df = df.copy()
    df['growth_rate'] = 0.0

    grouped = df.groupby(id_column)

    for track_id, group in grouped:
        group = group.sort_values('frame')
        values = group[value_column].to_numpy()
        growth_rates = np.zeros(len(values))

        for i in range(len(values)):
            if i + integration_window < len(values):
                future_avg = np.mean(values[i + 1:i + 1 + integration_window])
                growth_rates[i] = (future_avg - values[i]) / integration_window
            else:
                growth_rates[i] = 0.0  # Cannot compute growth at the end

        df.loc[group.index, 'growth_rate'] = growth_rates

    return df
    



def plot_growth_rate(
    df,
    frame_column="frame",
    growth_column="growth_rate",
    group_name=None,
    rec=False
):
    """
    Plots the average growth rate per frame.
    Accepts either a single DataFrame or a dictionary of DataFrames.

    Args:
        df (pd.DataFrame or dict of pd.DataFrame): Data to plot.
        frame_column (str): Name of the frame column.
        growth_column (str): Name of the growth rate column.
        group_name (str, optional): Label for the legend or title.
        rec (bool): Internal flag for recursive plotting. Should not be set by user.
    """
    if isinstance(df, dict):
        plt.figure(figsize=(10, 5))
        for name, sub_df in df.items():
            plot_growth_rate(
                sub_df,
                frame_column=frame_column,
                growth_column=growth_column,
                group_name=name,
                rec=True
            )
        plt.title("Average Growth Rate per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Average Growth Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    # Single DataFrame case
    avg_growth = df.groupby(frame_column)[growth_column].mean()
    label = group_name or "Average Growth Rate"

    if not rec:
        plt.figure(figsize=(10, 5))

    plt.plot(avg_growth.index, avg_growth.values, marker='o', label=label)

    if not rec:
        title = "Average Growth Rate per Frame"
        if group_name is not None:
            title = f"{group_name}: {title}"
        plt.title(title)
        plt.xlabel("Frame")
        plt.ylabel("Average Growth Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    


def plot_growth_rate_with_ribbon(
    df,
    frame_column="frame",
    growth_column="growth_rate",
    group_name=None,
    title = None,
    rec=False
):
    """
    Plots the mean growth rate per frame with a ribbon showing standard error of the mean (SEM).
    Can accept a single DataFrame or a dictionary of DataFrames to overlay multiple groups.

    Args:
        df (pd.DataFrame or dict of pd.DataFrame): Input data.
        frame_column (str): Name of the frame column.
        growth_column (str): Name of the growth rate column.
        group_name (str, optional): Label shown in the plot title (single DataFrame) or legend (dict).
        rec (bool): Internal flag for recursive plotting. Should not be set by user.
    """
    if isinstance(df, dict):
        plt.figure(figsize=(10, 5))
        for name, sub_df in df.items():
            plot_growth_rate_with_ribbon(
                sub_df,
                frame_column=frame_column,
                growth_column=growth_column,
                group_name=name,
                rec=True
            )
        if title is None:
            title = "Mean Growth Rate per Frame with SEM Ribbon"
        plt.title(title)
        plt.xlabel("Frame")
        plt.ylabel("Growth Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    # Single DataFrame plotting
    grouped = df.groupby(frame_column)[growth_column]
    mean_growth = grouped.mean()
    sem_growth = grouped.sem()

    frames = mean_growth.index
    means = mean_growth.values
    sems = sem_growth.values

    label = group_name or "Mean Growth Rate"
    
    if not rec:
        plt.figure(figsize=(10, 5))
    plt.plot(frames, means, label=label)
    plt.fill_between(frames, means - sems, means + sems, alpha=0.3)

    if not rec:
        if title is None:
            title = "Mean Growth Rate per Frame with SEM Ribbon"
        if group_name is not None:
            title = f"{group_name}: {title}"
        plt.title(title)
        plt.xlabel("Frame")
        plt.ylabel("Growth Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
  


def compute_densities(data, distance_threshold=50):
    # First, collect a lookup of all coordinates by frame
    frames = set()
    for df in data.values():
        frames.update(df['frame'].unique())
    
    # Loop over all frames
    for frame in frames:
        # Get all points in this frame from all channels
        frame_points = {}
        for key, df in data.items():
            points = df[df['frame'] == frame][['x', 'y']].values
            frame_points[key] = points
        
        # Build trees for all channels
        trees = {key: KDTree(pts) for key, pts in frame_points.items()}
        
        # Now calculate densities per dataframe
        for target_key, target_df in data.items():
            target_mask = target_df['frame'] == frame
            target_points = target_df.loc[target_mask, ['x', 'y']].values
            
            for source_key, tree in trees.items():
                # Count neighbors within distance_threshold (excluding the point itself)
                counts = tree.query_ball_point(target_points, r=distance_threshold)
                density = np.array([len(c) - 1 for c in counts])  # subtract 1 if same channel
                colname = f'density_{source_key}'
                
                # If this is the first frame, create column; otherwise, append
                if colname not in target_df.columns:
                    data[target_key][colname] = 0
                
                data[target_key].loc[target_mask, colname] = density
                
    # === Normalization Step ===
    # Collect all density column names
    density_cols = set()
    for df in data.values():
        density_cols.update(col for col in df.columns if col.startswith('density_'))
    
    # Calculate mean for each density column across all dataframes
    mean_densities = {}
    for col in density_cols:
        all_values = []
        for df in data.values():
            if col in df.columns:
                all_values.extend(df[col].values)
        mean_densities[col] = np.mean(all_values)

    # Compute the sum of the mean densities
    sum_mean_densities = sum(mean_densities.values())

    # Normalize each density column by the sum of mean densities
    for df in data.values():
        for col in density_cols:
            if col in df.columns:
                df[col] = df[col] / sum_mean_densities
                
    return data
      

def smooth_values_within_id(df: pd.DataFrame, id_column: str = "trackID", value_col: str = "growth_rate", interval: int = 5) -> pd.DataFrame:
    """
    Smooth the value column within each group using a moving average.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - id_column (str): Column name to group by.
    - value_col (str): Column name of the values to be smoothed.
    - interval (int): Integration interval (window size for smoothing).

    Returns:
    - pd.DataFrame: DataFrame with an additional 'smoothed' column.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Apply moving average within each group
    df[f'{value_col}_smoothed'] = df.groupby(id_column)[value_col] \
                       .transform(lambda x: x.rolling(window=interval, min_periods=1, center=True).mean())
    return df


def plot_correlation_with_fit(df: pd.DataFrame, x: str, y: str, title = None):
    """
    Plot x vs y with a linear regression line and display slope, intercept, and p-value.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - x (str): Name of the x-axis column.
    - y (str): Name of the y-axis column.
    """
    # Drop NA values
    data = df[[x, y]].dropna()

    x_vals = data[x].values
    y_vals = data[y].values

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)

    # Generate fit line
    fit_line = slope * x_vals + intercept

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, label='Data points', alpha=0.7)
    plt.plot(x_vals, fit_line, color='red', label='Linear fit')
    plt.xlabel(x)
    plt.ylabel(y)
    if title is None:
        title = f"Correlation between {x} and {y}"
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # Annotate with regression info
    plt.annotate(
        f"Slope: {slope:.3f}\nIntercept: {intercept:.3f}\nP-value: {p_value:.3e}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def show_slice(arr3d, axis=0, index=0, cmap='gray'):
    # Validate axis
    if axis < 0 or axis > 2:
        raise ValueError("Axis must be 0, 1, or 2.")

    # Take the appropriate 2D slice
    if axis == 0:
        img = arr3d[index, :, :]
    elif axis == 1:
        img = arr3d[:, index, :]
    else:  # axis == 2
        img = arr3d[:, :, index]

    # Show image
    plt.imshow(img, cmap=cmap)
    plt.title(f"Slice along axis {axis} at index {index}")
    plt.axis('off')
    plt.show()
    
    
    
def compute_densities_segmentation(data, masks, distance_threshold=50):
    # Precompute circular mask
    yy, xx = np.meshgrid(np.arange(-distance_threshold, distance_threshold + 1),
                         np.arange(-distance_threshold, distance_threshold + 1))
    circle_mask = (xx**2 + yy**2) <= distance_threshold**2
    dx, dy = circle_mask.shape

    for target_key, target_df in data.items():
        for source_key, mask in masks.items():
            densities = []

            for _, row in target_df.iterrows():
                frame = int(row['frame'])
                x = int(round(row['x']))
                y = int(round(row['y']))

                x0, y0 = x - dx // 2, y - dy // 2
                x1, y1 = x0 + dx, y0 + dy

                # Clip to bounds
                x0_clip, y0_clip = max(0, x0), max(0, y0)
                x1_clip, y1_clip = min(mask.shape[1], x1), min(mask.shape[2], y1)

                # Adjust circle mask slice to match the clipped region
                cx0, cy0 = x0_clip - x0, y0_clip - y0
                cx1, cy1 = cx0 + (x1_clip - x0_clip), cy0 + (y1_clip - y0_clip)
                circle_submask = circle_mask[cx0:cx1, cy0:cy1]

                region = mask[frame, x0_clip:x1_clip, y0_clip:y1_clip]
                valid_area = circle_submask.sum()
                overlap = np.logical_and(region, circle_submask).sum()

                ratio = overlap / valid_area if valid_area > 0 else 0.0
                densities.append(ratio)

            colname = f'density_{source_key}'
            target_df[colname] = densities

    return data



def run_stats_lmer(df, dependable_variable ,terms, interactions=None, group="trackID"):
    df = df.copy()

    # Create interaction terms from lists of column names
    interaction_terms = []
    if interactions is not None:
        for interaction in interactions:
            if not isinstance(interaction, list):
                raise ValueError("Each interaction must be a list of column names.")
            interaction_name = "interaction_" + "X".join(interaction)
            interaction_terms.append(interaction_name)
            df[interaction_name] = df.loc[:, interaction].prod(axis=1)

    # Combine all fixed effects
    fixed_effects = terms + interaction_terms
    formula = f"{dependable_variable} ~ " + " + ".join(fixed_effects)

    # Fit full model
    model = smf.mixedlm(formula, df, groups=df[group])
    result = model.fit()

    # Fit null model
    model_null = smf.mixedlm(f"{dependable_variable} ~ 1", df, groups=df[group])
    result_null = model_null.fit()

    # Likelihood ratio test
    llf_full = result.llf
    llf_null = result_null.llf

    lr_stat = 2 * (llf_full - llf_null)
    df_diff = result.df_modelwc - result_null.df_modelwc
    p_value = stats.chi2.sf(lr_stat, df_diff)

    # Output
    print(f"Likelihood Ratio Statistic: {lr_stat:.3f}")
    print(f"Degrees of Freedom: {df_diff}")
    print(f"p-value: {p_value:.4f}")
    print(result.summary())
    print("\nP-values:")
    print(result.pvalues)

    return result


def plot_frame_cv2_jupyter_dict(array_dict, frame_index=0, colors=None, figsize=(8, 8), title = None):
    """
    Displays overlay of boolean masks from a dict of 3D arrays (Jupyter-friendly).
    
    array_dict: dict with keys as names and values as 3D boolean numpy arrays
    frame_index: index along axis 0 to extract 2D slices
    colors: optional list of BGR tuples
    """
    if not array_dict:
        raise ValueError("Input dictionary is empty.")
    
    names = list(array_dict.keys())
    arrays_3d = list(array_dict.values())

    shape = arrays_3d[0][frame_index].shape
    overlay = np.zeros((*shape, 3), dtype=np.uint8)  # BGR image

    if colors is None:
        colors = [
            (0, 255, 255),     # Cyan (CFP-like)
            (255, 255, 0),     # Yellow (YFP-like)
            (255, 0, 255),     # Magenta (mCherry-like)
            (0, 255, 127),     # Spring Green
            (255, 105, 180),   # Hot Pink
            (0, 191, 255),     # Deep Sky Blue
            (124, 252, 0),     # Lawn Green
            (255, 20, 147),    # Deep Pink
            (173, 255, 47),    # Green Yellow
            (240, 128, 128),   # Light Coral
        ]

    legend_patches = []

    for i, (name, arr) in enumerate(array_dict.items()):
        if arr.ndim != 3 or arr.dtype != bool:
            raise ValueError(f"Array for '{name}' is not a 3D boolean array.")
        if frame_index >= arr.shape[0]:
            raise IndexError(f"Frame index {frame_index} out of bounds for array '{name}'.")

        mask = arr[frame_index]
        color = colors[i % len(colors)]

        for c in range(3):  # B, G, R
            overlay[:, :, c][mask] = color[c]

        rgb_color = (color[2]/255, color[1]/255, color[0]/255)
        legend_patches.append(Patch(color=rgb_color, label=name))

    # Convert BGR to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(overlay_rgb)
    if title is None:
        title = "Overlay of Channels"
    plt.title(f"{title} at Frame {frame_index}")
    plt.axis('off')
    plt.legend(handles=legend_patches, loc='upper right')
    plt.show()
    
