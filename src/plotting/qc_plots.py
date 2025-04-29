import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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