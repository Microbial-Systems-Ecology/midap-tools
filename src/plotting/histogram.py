import math

import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List

def plot_histogram(
    df: Union[pd.DataFrame, dict],
    column: Union[str, List[str]],
    bins: int = 100,
    title: str = None,
    rec: bool = False
):
    """
    Plots histograms of raw values from specified columns in a DataFrame or dict of DataFrames.
    Uses a grid layout with one subplot per (DataFrame, column) combination.

    Args:
        df (pd.DataFrame or dict of pd.DataFrame): Data source(s).
        column (str or list of str): Column name(s) to plot.
        bins (int): Number of bins for each histogram. Default is 100.
        title (str): Optional overarching title (for multi-plots).
        rec (bool): Internal flag for recursive calls. Should not be set by the user.
    """
    # Normalize column input to list
    if isinstance(column, str):
        column = [column]

    if isinstance(df, dict):
        names = list(df.keys())
        cols = len(names)
        rows = len(column)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=False, sharey=False)

        # Make sure axes is always 2D
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for col_idx, col in enumerate(column):
            for df_idx, name in enumerate(names):
                ax = axes[col_idx][df_idx]
                plt.sca(ax)
                plot_histogram(
                    df[name],
                    column=col,
                    bins=bins,
                    title=f'{name} - {col}',
                    rec=True
                )

        for col_idx, col in enumerate(column):
            axes[col_idx][0].set_ylabel("Frequency")
        for df_idx, name in enumerate(names):
            axes[-1][df_idx].set_xlabel("Value")

        fig.suptitle(title or f'Histogram(s) from {", ".join(column)}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return

    # Single DataFrame case
    for col in column:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        data = df[col].dropna().values

        if not rec:
            plt.figure(figsize=(6, 4))

        plt.hist(data, bins=bins, edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title or f'Histogram of {col}')
        plt.grid(True)

        if not rec:
            plt.tight_layout()
            plt.show()
                  
        
def plot_value_count_histogram(
    df: Union[pd.DataFrame, dict],
    column: Union[str, List[str]],
    bins: int = 100,
    title: str = None,
    rec: bool = False
):
    """
    Plots histograms of value counts from specified columns in a DataFrame or dict of DataFrames.
    Uses a grid layout with one subplot per (DataFrame, column) combination.

    Args:
        df (pd.DataFrame or dict of pd.DataFrame): Data source(s).
        column (str or list of str): Column name(s) to analyze.
        bins (int): Number of bins for each histogram. Default is 100.
        title (str): Optional overarching title (for multi-plots).
        rec (bool): Internal flag for recursive calls. Should not be set by the user.
    """
    if isinstance(column, str):
        column = [column]

    if isinstance(df, dict):
        names = list(df.keys())
        cols = len(names)
        rows = len(column)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=False, sharey=False)

        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for col_idx, col in enumerate(column):
            for df_idx, name in enumerate(names):
                ax = axes[col_idx][df_idx]
                plt.sca(ax)
                plot_value_count_histogram(
                    df[name],
                    column=col,
                    bins=bins,
                    title=f'{name} - {col}',
                    rec=True
                )

        for col_idx, col in enumerate(column):
            axes[col_idx][0].set_ylabel("Frequency")
        for df_idx, name in enumerate(names):
            axes[-1][df_idx].set_xlabel("Value counts")

        fig.suptitle(title or f'Value count histogram(s) for {", ".join(column)}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return

    # Single DataFrame case
    for col in column:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        value_counts = df[col].value_counts()

        if not rec:
            plt.figure(figsize=(6, 4))

        plt.hist(value_counts.values, bins=bins, edgecolor='black')
        plt.xlabel('Value counts')
        plt.ylabel('Frequency')
        plt.title(title or f'Value counts of {col}')
        plt.grid(True)

        if not rec:
            plt.tight_layout()
            plt.show()
