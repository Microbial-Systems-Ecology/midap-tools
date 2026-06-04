import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def _filter_snapshot(df: pd.DataFrame, frame_column: str, start_time: float) -> pd.DataFrame:
    """Return rows whose frame_column value equals start_time (float-safe)."""
    return df[np.isclose(df[frame_column].values.astype(float), float(start_time))]


def _division_time_curve(durations: np.ndarray, n_points: int = 300) -> tuple:
    """Return (t, undivided_fraction) arrays for a vector of duration values."""
    if len(durations) == 0:
        return np.array([]), np.array([])
    t = np.linspace(0.0, float(durations.max()), n_points)
    undivided = np.array([(durations >= ti).mean() for ti in t])
    return t, undivided


def plot_division_time_boxplot(
    data: Dict[str, pd.DataFrame],
    time_to_split_column: str = "time_to_split",
    start_times: List = None,
    frame_column: str = "frame",
    title: str = "Duration Until Next Division",
    group_label: str = "group",
) -> None:
    """
    Boxplot + stripplot of time-to-split values for each start time and group.

    Generates one figure with all start_times on the x-axis and groups as hue. Boxes show
    the interquartile range; individual observations are overlaid as a strip plot.

    Args:
        data (dict[str, pd.DataFrame]): mapping of group name → DataFrame.
            Each DataFrame must contain frame_column and time_to_split_column.
        time_to_split_column (str): column holding time-to-split values. Defaults to "time_to_split"
        start_times (list): snapshot times at which to slice each DataFrame
        frame_column (str): column used to identify snapshot rows. Defaults to "frame"
        title (str): figure title. Defaults to "Duration Until Next Division"
        group_label (str): legend title for the hue axis. Defaults to "group"
    """
    if not start_times:
        raise ValueError("start_times must be a non-empty list")

    records = []
    for group_name, df in data.items():
        for start_time in start_times:
            sub = _filter_snapshot(df, frame_column, start_time)
            for val in sub[time_to_split_column].dropna():
                records.append(
                    {frame_column: start_time, group_label: str(group_name), time_to_split_column: float(val)}
                )

    if not records:
        print("plot_division_time_boxplot: no data found at the requested start_times")
        return

    plot_df = pd.DataFrame(records)
    time_order = sorted(plot_df[frame_column].unique())
    plot_df[frame_column] = pd.Categorical(plot_df[frame_column], categories=time_order, ordered=True)

    fig, ax = plt.subplots(figsize=(max(8, len(start_times) * 0.9), 6))

    sns.boxplot(
        data=plot_df,
        x=frame_column,
        y=time_to_split_column,
        hue=group_label,
        palette="Set2",
        width=0.6,
        linewidth=1.4,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x=frame_column,
        y=time_to_split_column,
        hue=group_label,
        palette="Set2",
        dodge=True,
        size=4,
        alpha=0.7,
        linewidth=0.5,
        edgecolor="black",
        jitter=True,
        legend=False,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"Start time ({frame_column})", fontsize=12)
    ax.set_ylabel(f"Duration ({time_to_split_column})", fontsize=12)
    ax.legend(title=group_label, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.tick_params(axis="x", rotation=30)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_division_time_curves(
    data: Dict[str, pd.DataFrame],
    time_to_split_column: str = "time_to_split",
    start_times: List = None,
    frame_column: str = "frame",
    title: str = "Division Time Curves",
) -> None:
    """
    Division time curve panels for each start time, arranged into a single figure.

    Each subplot shows one start_time with curves overlaid for all groups. The y-axis
    is the fraction of cells that have not yet divided, so both inter-group and
    cross-time comparisons are visible at a glance.

    Args:
        data (dict[str, pd.DataFrame]): mapping of group name → DataFrame.
            Each DataFrame must contain frame_column and time_to_split_column.
        time_to_split_column (str): column holding time-to-split values. Defaults to "time_to_split"
        start_times (list): snapshot times for which to draw division time curves
        frame_column (str): column used to identify snapshot rows. Defaults to "frame"
        title (str): figure suptitle. Defaults to "Division Time Curves"
    """
    if not start_times:
        raise ValueError("start_times must be a non-empty list")

    n_times = len(start_times)
    n_cols = min(4, n_times)
    n_rows = math.ceil(n_times / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    palette = sns.color_palette("tab10", n_colors=max(len(data), 1))
    color_map = {name: palette[i] for i, name in enumerate(data.keys())}

    for idx, start_time in enumerate(start_times):
        ax = axes[idx // n_cols][idx % n_cols]
        for group_name, df in data.items():
            sub = _filter_snapshot(df, frame_column, start_time)
            durations = sub[time_to_split_column].dropna().values
            if len(durations) == 0:
                continue
            t, undivided = _division_time_curve(durations)
            ax.plot(t, undivided, label=str(group_name), color=color_map[group_name], lw=2)

        ax.set_title(f"t = {start_time}", fontsize=10)
        ax.set_xlabel(f"Duration ({time_to_split_column})", fontsize=9)
        ax.set_ylabel("Fraction not yet divided", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.legend(title="group", fontsize=8, title_fontsize=8)

    for idx in range(n_times, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_division_time_heatmap(
    data: Dict[str, pd.DataFrame],
    time_to_split_column: str = "time_to_split",
    start_times: List = None,
    frame_column: str = "frame",
    title: str = "Division Time Heatmap",
    n_duration_bins: int = 50,
    max_time: float = None,
) -> None:
    """
    Heatmap of division time fractions across start times and duration values.

    One heatmap per group. Rows = start_times, columns = duration bins. Color encodes
    the fraction of cells with time_to_split ≥ column value at each start time, giving
    a compact view of how division timing shifts across the experiment. Cells where
    start_time + duration > max_time are unobservable and rendered as white.

    Args:
        data (dict[str, pd.DataFrame]): mapping of group name → DataFrame.
            Each DataFrame must contain frame_column and time_to_split_column.
        time_to_split_column (str): column holding time-to-split values. Defaults to "time_to_split"
        start_times (list): snapshot times to use as heatmap rows
        frame_column (str): column used to identify snapshot rows. Defaults to "frame"
        title (str): figure suptitle. Defaults to "Division Time Heatmap"
        n_duration_bins (int): number of duration columns in the heatmap. Defaults to 50
        max_time (float, optional): upper bound of valid frame values. Any cell where
            start_time + duration > max_time is rendered as white (unobservable).
            If None, derived per group from the data's frame_column maximum.
    """
    if not start_times:
        raise ValueError("start_times must be a non-empty list")

    n_groups = len(data)
    fig_width = max(6, 6 * n_groups)
    fig_height = max(4.0, 0.5 * len(start_times) + 2.0)
    fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, fig_height), squeeze=False)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    for col_idx, (group_name, df) in enumerate(data.items()):
        ax = axes[0][col_idx]

        effective_max_time = max_time if max_time is not None else float(df[frame_column].max())

        all_durs: list = []
        for start_time in start_times:
            if start_time > effective_max_time:
                continue
            sub = _filter_snapshot(df, frame_column, start_time)
            all_durs.extend(sub[time_to_split_column].dropna().tolist())

        if not all_durs:
            ax.set_title(f"{group_name}\n(no data)", fontsize=11)
            continue

        max_dur = float(max(all_durs))
        dur_points = np.linspace(0.0, max_dur, n_duration_bins)

        rows: dict = {}
        for start_time in start_times:
            if start_time > effective_max_time:
                rows[start_time] = [np.nan] * n_duration_bins
                continue
            sub = _filter_snapshot(df, frame_column, start_time)
            vals = sub[time_to_split_column].dropna().values
            max_observable = effective_max_time - start_time
            if len(vals) == 0:
                rows[start_time] = [np.nan] * n_duration_bins
            else:
                rows[start_time] = [
                    (vals >= d).mean() if d <= max_observable else np.nan
                    for d in dur_points
                ]

        heatmap_df = pd.DataFrame(rows, index=dur_points).T
        heatmap_df.columns = [f"{v:.2f}" for v in dur_points]

        tick_interval = max(1, n_duration_bins // 8)
        sns.heatmap(
            heatmap_df,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "fraction not yet divided", "shrink": 0.8},
            xticklabels=tick_interval,
        )
        ax.set_facecolor("white")
        ax.set_title(str(group_name), fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Duration ({time_to_split_column})", fontsize=10)
        ax.set_ylabel(f"Start time ({frame_column})", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
