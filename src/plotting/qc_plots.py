import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.patches import Patch
import matplotlib

def plot_qc_xy_correlation(data: pd.DataFrame, 
                           id_column: str = "trackID", 
                           value_column: str = "major_axis_length", 
                           frame_column: str = "frame", 
                           n: int = 5, 
                           random_seed: int = 42,
                           title = None):
    """
    Creates a QC plot showing XY correlation for `n` random examples grouped by `id_column`.
    Each plot includes a linear regression line with the R² value displayed.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        id_column (str): The column used to group the data. Defaults to "trackID".
        value_column (str): The column representing the Y-axis values. Defaults to "major_axis_length".
        frame_column (str): The column representing the X-axis values. Defaults to "frame".
        n (int): The number of random examples to plot. Defaults to 5.
        random_seed (int): The random seed for reproducibility. Defaults to 42.

    Returns:
        None: Displays the QC plots.
    """
    # Ensure reproducibility
    np.random.seed(random_seed)
    
    # Group data by the id_column
    grouped = data.groupby(id_column)
    
    # Randomly select `n` groups
    selected_groups = np.random.choice(list(grouped.groups.keys()), size=min(n, len(grouped.groups)), replace=False)
    
    # Set up the plot grid
    n_cols = 4
    n_rows = int(np.ceil(len(selected_groups) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()  # Flatten axes for easy iteration
    
    for ax, group_id in zip(axes, selected_groups):
        group_data = grouped.get_group(group_id)
        
        # Extract X and Y values
        x = group_data[frame_column].values.reshape(-1, 1)
        y = group_data[value_column].values
        
        # Perform linear regression
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)
        
        # Plot the data points
        ax.scatter(group_data[frame_column], group_data[value_column], label="Data", color="blue", alpha=0.7)
        
        # Plot the regression line
        ax.plot(group_data[frame_column], y_pred, color="red", label=f"Linear Fit (R²={r2:.2f})")
        
        # Set plot labels and title
        if title is None:
            title = f"QC-plot for {id_column}"
        
        ax.set_title(f"{title}: {group_id}")
        ax.set_xlabel(frame_column)
        ax.set_ylabel(value_column)
        ax.legend()
    
    # Hide unused subplots
    for ax in axes[len(selected_groups):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    
def plot_frame_cv2_jupyter_dict(array_dict, 
                                frame_index=0, 
                                colors=None, 
                                figsize=(8, 8), 
                                title = None):
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
    
def plot_xy_correlation(
    df: Union[pd.DataFrame, dict],
    x: str,
    y: str,
    title: str = None,
    rec: bool = False
):
    """
    Plots XY scatter plots and fits a linear regression model.
    Supports both single DataFrame and dict of DataFrames.

    Args:
        df (pd.DataFrame or dict of pd.DataFrame): Data source(s).
        x (str): Name of X-axis column.
        y (str): Name of Y-axis column.
        title (str): Optional plot title.
        rec (bool): Internal recursion flag for nested plotting.
    """
    if isinstance(df, dict):
        names = list(df.keys())
        cols = len(names)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 6, 5), sharey=False)

        if cols == 1:
            axes = [axes]

        for idx, name in enumerate(names):
            plt.sca(axes[idx])
            plot_xy_correlation(df[name], x, y, title=f'{name}: {x} vs {y}', rec=True)

        fig.suptitle(title or f'Correlation Plots: {x} vs {y}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return

    # Base case: single DataFrame
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' and/or '{y}' not found in DataFrame.")

    x_vals = df[x].dropna()
    y_vals = df[y].dropna()
    common_index = x_vals.index.intersection(y_vals.index)

    x_clean = x_vals.loc[common_index].values.reshape(-1, 1)
    y_clean = y_vals.loc[common_index].values

    model = LinearRegression()
    model.fit(x_clean, y_clean)
    y_pred = model.predict(x_clean)
    r2 = r2_score(y_clean, y_pred)

    if not rec:
        plt.figure(figsize=(6, 4))

    plt.scatter(x_clean, y_clean, alpha=0.6, label='Data')
    plt.plot(x_clean, y_pred, color='red', linewidth=2, label='Fit')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f'{y} vs {x}')
    plt.legend()
    plt.grid(True)

    coeff_text = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\n$R^2$ = {r2:.3f}'
    plt.annotate(coeff_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    if not rec:
        plt.tight_layout()
        plt.show()



def plot_xy_correlation_stacked(
    df: Union[pd.DataFrame, dict],
    x: str,
    y: str,
    title: str = None,
    xlim = None,
    ylim = None,
) -> plt.Figure:
    """
    Creates a plot of XY scatter data with linear regression fits.
    If input is a dict of DataFrames, each is plotted in the same axes with a distinct color.

    Args:
        df (pd.DataFrame or dict of pd.DataFrame): Input data.
        x (str): X-axis column name.
        y (str): Y-axis column name.
        title (str): Optional plot title.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    if isinstance(df, pd.DataFrame):
        df = {"Dataset": df}  # Normalize to dict format

    if not isinstance(df, dict):
        raise TypeError("Input must be a DataFrame or a dict of DataFrames.")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    summary_texts = []
    
    scores = {}

    for idx, (name, data) in enumerate(df.items()):
        scores[name] = {}
        if x not in data.columns or y not in data.columns:
            raise ValueError(f"Columns '{x}' and/or '{y}' not found in dataset '{name}'.")

        x_vals = data[x].dropna()
        y_vals = data[y].dropna()
        common_index = x_vals.index.intersection(y_vals.index)

        x_clean = x_vals.loc[common_index].values.reshape(-1, 1)
        y_clean = y_vals.loc[common_index].values

        model = LinearRegression()
        model.fit(x_clean, y_clean)
        y_pred = model.predict(x_clean)
        r2 = r2_score(y_clean, y_pred)

        color = colors[idx % len(colors)]
        ax.scatter(x_clean, y_clean, alpha=0.6, label=f'{name} Data', color=color)
        ax.plot(x_clean, y_pred, color=color, linewidth=2, label=f'{name} Fit')
        scores[name]["slope"] = model.coef_[0]
        scores[name]["intercept"] = model.intercept_
        scores[name]["r2"] = r2

        summary_texts.append(
            f"{name}:\n  y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}\n  $R^2$ = {r2:.3f}"
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f'{y} vs {x}')
    ax.legend()
    ax.grid(True)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    full_summary = "\n\n".join(summary_texts)
    # Adjust text position to fit within extended figure width
    fig.text(0.85, 0.5, full_summary, fontsize=10, va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
            transform=fig.transFigure)

    # Adjust layout so that the plotting area doesn't overlap with text
    fig.tight_layout(rect=[0, 0, 0.8, 1])  # Reserve right 20% for stats
    return fig, scores


def plot_spatial_maps(array_dict: dict,
                      df_dict: dict,
                      property: str,
                      frame_number: int = 0,
                      title: str = None,
                      silent: bool = True):
    """
    Plots spatial maps of cell property. Adapted from code received from Simon van Vilet

    Parameters:
        array_dict (dict np.ndarray):  Numpy ND array [t,y,x] with label image stack
        df_dict (dict of pd.DataFrame): pandas data frame of lineage object
        property (str): key of cell property contained in lineage object
        frame_number (int, optional): frame number to show, in case of 3D label stack, defaults to 0
        title (str, optional): title of the plot, defaults to None
        silent (bool, optional): if True, suppresses warnings about missing cells in the frame, defaults to True

    Returns:
        Creates a matplotlib figure with spatial maps of cell property at given frame and a colorbar
    """
    colMap = matplotlib.colormaps["viridis"].copy()
    colMap.set_bad(color='black')
    n_col = len(df_dict)

    fig, axs = plt.subplots(1, n_col, figsize=(n_col * 5, 5), constrained_layout=True)
    if n_col == 1:
        axs = [axs]

    im = None  # to store the image handle for colorbar reference

    for i, (k, label_stack, df) in enumerate(zip(df_dict.keys(), array_dict.values(), df_dict.values())):
        labels = label_stack[frame_number, :, :]
        spatial_map = np.full(labels.shape, np.nan)

        for cnb in np.unique(labels):
            if cnb == 0:
                continue
            try:
                spatial_map[labels == cnb] = df.loc[
                    (df['frame'] == frame_number) & (df['trackID'] == cnb), property
                ].item()
            except Exception:
                if not silent:
                    print(f"skipping cell {cnb} in frame {frame_number}")

        im = axs[i].imshow(spatial_map, cmap=colMap)
        axs[i].set_title(k)

    # Add a colorbar to the right of the plot grid
    if im:
        cbar = fig.colorbar(im, ax=axs, location='right', shrink=0.8, label=property)

    fig.suptitle(title or f'Spatial Maps of {property} at Frame {frame_number}')
    plt.show()
