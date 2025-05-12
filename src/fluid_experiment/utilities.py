import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import linregress
import statsmodels.formula.api as smf


def sort_folder_names(folders):
    def extract_number(name):
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else float('inf')  # Push non-numbered names to end

    return sorted(folders, key=extract_number)

      

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
    
