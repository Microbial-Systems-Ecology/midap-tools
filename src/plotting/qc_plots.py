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
                           title=None,
                           overlay_column: str = None,
                           free_overlay_axes = True):
    """
    Creates a QC plot showing XY correlation for `n` random examples grouped by `id_column`.
    Each plot includes a linear regression line with the R² value displayed.

    If `overlay_column` is provided, it is plotted on a secondary Y-axis (right side)
    with its own regression fit.

    Args:
        data (pd.DataFrame): Input DataFrame.
        id_column (str): Column used for grouping.
        value_column (str): Primary Y-axis values.
        frame_column (str): X-axis values.
        n (int): Number of random examples.
        random_seed (int): Random seed for reproducibility.
        title (str): Optional title.
        overlay_column (str): Optional secondary Y-axis values.
        free_overlay_axes (bool): should the overlay data be shown with its own axis range?
    """

    # Validate columns
    required = {id_column, value_column, frame_column}
    if overlay_column:
        required.add(overlay_column)

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    np.random.seed(random_seed)

    grouped = data.groupby(id_column)
    selected_groups = np.random.choice(
        list(grouped.groups.keys()),
        size=min(n, len(grouped.groups)),
        replace=False
    )

    # Layout
    n_cols = 4
    n_rows = int(np.ceil(len(selected_groups) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    if title is None:
        title = f"QC-plot for {id_column}"

    for ax, group_id in zip(axes, selected_groups):
        group_data = grouped.get_group(group_id)

        x_raw = group_data[frame_column].values
        y_raw = group_data[value_column].values

        mask = np.isfinite(x_raw) & np.isfinite(y_raw)

        if mask.sum() >= 2:
            x = x_raw[mask].reshape(-1, 1)
            y = y_raw[mask]

            model = LinearRegression()
            model.fit(x, y)

            # Predict over full x-range for plotting
            y_pred = model.predict(x_raw.reshape(-1, 1))
            r2 = r2_score(y, model.predict(x))
        else:
            y_pred = np.full_like(x_raw, np.nan, dtype=float)
            r2 = np.nan

        ax.scatter(x_raw, y_raw, color="blue", alpha=0.7, label=value_column)
        ax.plot(group_data[frame_column], y_pred, color="red",
                label=f"{value_column} fit (R²={r2:.2f})")

        ax.set_xlabel(frame_column)
        ax.set_ylabel(value_column)

        # --- OVERLAY AXIS (optional) ---
        if overlay_column:
            if free_overlay_axes:
                ax2 = ax.twinx()          # independent axis
            else:
                ax2 = ax
            x2_raw = group_data[frame_column].values
            y2_raw = group_data[overlay_column].values

            mask2 = np.isfinite(x2_raw) & np.isfinite(y2_raw)

            if mask2.sum() >= 2:
                x2 = x2_raw[mask2].reshape(-1, 1)
                y2 = y2_raw[mask2]

                model2 = LinearRegression()
                model2.fit(x2, y2)

                y2_pred = model2.predict(x2_raw.reshape(-1, 1))
                r2_2 = r2_score(y2, model2.predict(x2))
            else:
                y2_pred = np.full_like(x2_raw, np.nan, dtype=float)
                r2_2 = np.nan

            ax2.scatter(x2_raw, y2_raw,
                        color="green", alpha=0.6, marker="x",
                        label=overlay_column)
            ax2.plot(group_data[frame_column], y2_pred, color="darkgreen",
                     linestyle="--",
                     label=f"{overlay_column} fit (R²={r2_2:.2f})")

            ax2.set_ylabel(overlay_column)

            # Merge legends from both axes
            if free_overlay_axes:
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(handles1 + handles2, labels1 + labels2,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.35),
                        ncol=1,
                        fontsize=8,
                        frameon=False)

        if overlay_column is None or not free_overlay_axes:
            ax.legend(loc="upper center",
                    bbox_to_anchor=(0.5, -0.35),
                    ncol=1,
                    fontsize=8,
                    frameon=False)

        ax.set_title(f"{title}: {group_id}")

    # Hide empty axes
    for ax in axes[len(selected_groups):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
def plot_frame_cv2_jupyter_dict(array_dict, 
                                frame_index=0, 
                                colors=None, 
                                figsize=(8, 8), 
                                title=None,
                                show_distance: int = None,
                                show: bool = True):
    """
    Displays overlay of boolean masks from a dict of 3D arrays (Jupyter-friendly).

    Args:
        array_dict: dict with keys as names and values as 3D boolean numpy arrays
        frame_index: index along axis 0 to extract 2D slices
        colors: optional list of BGR tuples
        figsize: size of the figure in inches
        title: plot title
        show_distance: if set, a red circle with this px radius is drawn in the center
        show (bool): if False, the function will return a figure object that can be used for other operations 
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

    # Draw center circle if requested
    if show_distance is not None and show_distance > 0:
        center_y = shape[0] // 2
        center_x = shape[1] // 2
        cv2.circle(overlay, (center_x, center_y), show_distance, (0, 0, 255), thickness=3)  # Red circle

    # Convert BGR to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plot
    fig = plt.figure(figsize=figsize)
    plt.imshow(overlay_rgb)
    if title is None:
        title = "Overlay of Channels"
    plt.title(f"{title} at Frame {frame_index}")
    plt.axis('off')
    plt.legend(handles=legend_patches, loc='upper right')
    if show:
        plt.show()
    else:
        return fig
    
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
                      silent: bool = True,
                      show: bool = True):
    """
    Plots spatial maps of cell property. Adapted from code received from Simon van Vilet

    Parameters:
        array_dict (dict np.ndarray):  Numpy ND array [t,y,x] with label image stack
        df_dict (dict of pd.DataFrame): pandas data frame of lineage object
        property (str): key of cell property contained in lineage object
        frame_number (int, optional): frame number to show, in case of 3D label stack, defaults to 0
        title (str, optional): title of the plot, defaults to None
        silent (bool, optional): if True, suppresses warnings about missing cells in the frame, defaults to True
        show (bool): if False, the function will return a figure object that can be used for other operations 

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
    if show:
        plt.show()
    else:
        return fig


def plot_spatial_maps_overlayed(array_dict: dict,
                                 df_dict: dict,
                                 property: str,
                                 color_map: dict,
                                 frame_number: int = 0,
                                 value_ranges: dict = None,
                                 title: str = None,
                                 silent: bool = True,
                                 show: bool = True):
    """
    Plots overlayed spatial maps of a cell property, compositing multiple channels
    each rendered with their own color gradient onto a single black background.
    Per-channel gradient colorbars are stacked vertically on the right.

    Parameters:
        array_dict (dict of np.ndarray): Numpy ND array [t,y,x] with label image stacks, keyed by channel.
        df_dict (dict of pd.DataFrame): Pandas DataFrames with tracking data, keyed by channel.
        property (str): Key of cell property contained in the DataFrames.
        color_map (dict): Mapping of channel name to base RGB color tuple, e.g. {'GFP': (0,1,0)}.
                          Colors should be in [0,1] range.
        frame_number (int, optional): Frame number to display. Defaults to 0.
        value_ranges (dict, optional): Mapping of channel name to (min, max) tuple for fixed scaling.
                                       If None, per-frame min/max is used (not recommended for GIFs).
        title (str, optional): Title of the plot. Defaults to None.
        silent (bool, optional): If True, suppresses warnings about missing cells. Defaults to True.
        show (bool): If False, returns the figure object instead of displaying it.

    Returns:
        matplotlib.figure.Figure or None: Figure object if show=False, else None.
    """
    # Determine the image shape from the first available channel
    shape = None
    for c, label_stack in array_dict.items():
        shape = label_stack[frame_number, :, :].shape
        break
    if shape is None:
        raise ValueError("array_dict is empty; cannot determine image shape.")

    active_channels = [c for c in df_dict.keys() if c in array_dict and c in color_map]
    n_cbars = len(active_channels)

    # Layout: image axes (left) + one narrow column of stacked colorbar axes (right)
    # Each channel gets its own subplot row in the right column via GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(8.5, 6), facecolor='black')
    outer_gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 0.18],
        wspace=0.08,
        left=0.02, right=0.93,
        top=0.92, bottom=0.05
    )

    ax_img = fig.add_subplot(outer_gs[0])

    # Stack n_cbars colorbar axes vertically in the right column with small gaps
    inner_gs = outer_gs[1].subgridspec(n_cbars, 1, hspace=0.6)
    cbar_axes = [fig.add_subplot(inner_gs[j]) for j in range(n_cbars)]

    # Composite RGB image
    composite = np.zeros((*shape, 3), dtype=np.float64)
    floor_intensity = 0.05
    channel_data = {}

    for c in active_channels:
        label_stack = array_dict[c]
        df = df_dict[c]
        labels = label_stack[frame_number, :, :]
        base_color = np.array(color_map[c], dtype=np.float64)

        spatial_map = np.full(shape, np.nan)
        for cnb in np.unique(labels):
            if cnb == 0:
                continue
            try:
                spatial_map[labels == cnb] = df.loc[
                    (df['frame'] == frame_number) & (df['trackID'] == cnb), property
                ].item()
            except Exception:
                if not silent:
                    print(f"skipping cell {cnb} in frame {frame_number} for channel {c}")

        if value_ranges is not None and c in value_ranges:
            v_min, v_max = value_ranges[c]
        else:
            valid = spatial_map[~np.isnan(spatial_map)]
            if len(valid) == 0:
                channel_data[c] = {"base_color": base_color, "v_min": 0.0, "v_max": 1.0}
                continue
            v_min, v_max = float(valid.min()), float(valid.max())

        denom = v_max - v_min if v_max != v_min else 1.0
        norm_map = np.clip((spatial_map - v_min) / denom, 0.0, 1.0)
        intensity = np.where(
            np.isnan(spatial_map),
            0.0,
            floor_intensity + norm_map * (1.0 - floor_intensity)
        )
        composite += intensity[:, :, np.newaxis] * base_color[np.newaxis, np.newaxis, :]
        channel_data[c] = {"base_color": base_color, "v_min": v_min, "v_max": v_max}

    composite = np.clip(composite, 0.0, 1.0)

    ax_img.imshow(composite)
    ax_img.set_title(title or f'Overlayed Spatial Maps of {property} at Frame {frame_number}',
                     color='white', fontsize=9)
    ax_img.axis('off')
    ax_img.set_facecolor('black')

    # Draw each colorbar in its own stacked axes slot
    gradient = np.linspace(0, 1, 256).reshape(256, 1)

    for i, c in enumerate(active_channels):
        ax_cb = cbar_axes[i]
        info = channel_data.get(c)
        if info is None:
            ax_cb.axis('off')
            continue

        base_color = info["base_color"]
        v_min = info["v_min"]
        v_max = info["v_max"]

        # Build gradient image: low at bottom, high at top
        cbar_img = np.zeros((256, 1, 3), dtype=np.float64)
        intensities = floor_intensity + gradient[:, 0] * (1.0 - floor_intensity)
        cbar_img[:, 0, :] = intensities[:, np.newaxis] * base_color[np.newaxis, :]
        cbar_img = np.clip(cbar_img, 0.0, 1.0)

        ax_cb.imshow(cbar_img, aspect='auto', origin='lower',
                     extent=[0, 1, v_min, v_max])
        ax_cb.set_xlim(0, 1)
        ax_cb.set_ylim(v_min, v_max)
        ax_cb.set_xticks([])

        # Ticks on the right side only, 3 ticks: min, mid, max
        mid = (v_min + v_max) / 2.0
        ax_cb.set_yticks([v_min, mid, v_max])
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_label_position('right')
        ax_cb.tick_params(axis='y', labelsize=7, colors='white', length=3, pad=2)
        ax_cb.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

        # Channel name above each colorbar, in its own color
        ax_cb.set_title(c, fontsize=8, color=base_color.tolist(), pad=3)

        ax_cb.set_facecolor('black')
        for spine in ax_cb.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

    if show:
        plt.show()
    else:
        return fig