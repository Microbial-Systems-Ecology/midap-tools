import matplotlib.pyplot as plt

def plot_growth_rate_with_ribbon(
    df,
    growth_column="growth_rate",
    frame_column="frame",
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