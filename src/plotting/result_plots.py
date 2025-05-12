from plotnine import (ggplot, 
                      aes, 
                      geom_point, 
                      facet_grid, 
                      theme, 
                      labs, 
                      geom_boxplot, 
                      position_jitterdodge, 
                      element_rect, 
                      element_line, 
                      element_text)

def summary_plot(df, 
                 value_column, 
                 group_column, 
                 bins_column = None,
                 subsetting1 = None, 
                 subsetting2 = None):
    """creates a summary plot of the data
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        value_column (str): Column name for the values to plot.
        group_column (str): Column name for the groups to facet by.
        subsetting1 (str, optional): Column name for the first subsetting variable.
        subsetting2 (str, optional): Column name for the second subsetting variable.
    """
    if bins_column is not None:
        p1 = (
        ggplot(df, aes(bins_column,value_column, color=group_column))
        + geom_boxplot(outlier_shape = None)
        + geom_point(position = position_jitterdodge(jitter_width=0.1), alpha=0.5)
        )
    else:
        p1 = (
        ggplot(df, aes(group_column,value_column, color = group_column))
        + geom_boxplot(outlier_shape = None)
        + geom_point(position = position_jitterdodge(jitter_width=0.1), alpha=0.5)
        )

    # Adjust faceting and figure size based on subsettings
    if subsetting1 is not None and subsetting2 is not None:
        p1 += facet_grid(f"{subsetting1} ~ {subsetting2}", scales="free")
        # Adjust figure size based on the number of unique values in subsetting1 and subsetting2
        n_rows = len(df[subsetting1].unique())
        n_cols = len(df[subsetting2].unique())
        p1 += theme(figure_size=(n_cols * 4, n_rows * 4))
    elif subsetting1 is not None:
        p1 += facet_grid(f"{subsetting1} ~ .", scales="free")
        # Adjust figure size based on the number of unique values in subsetting1
        n_rows = len(df[subsetting1].unique())
        p1 += theme(figure_size=(6, n_rows * 4))
    elif subsetting2 is not None:
        p1 += facet_grid(f". ~ {subsetting2}", scales="free")
        # Adjust figure size based on the number of unique values in subsetting2
        n_cols = len(df[subsetting2].unique())
        p1 += theme(figure_size=(n_cols * 4, 6))
    else:
        # Default figure size when no subsetting is applied
        p1 += theme(figure_size=(8, 6))

    # Add theme and labels
    p1 += theme(
        panel_background=element_rect(fill="white", color=None),  # White panel background
        #panel_grid_major=element_line(color="gray", size=0.5, alpha = 0.5),    # Gray major grid lines
        #panel_grid_minor=element_line(color="lightgray", size=0.25, alpha = 0.5),  # Light gray minor grid lines
        plot_background=element_rect(fill="white", color=None),   # White plot background
        panel_border=element_rect(color="black", size=1),         # Add black border around subplots
        strip_background=element_rect(fill="lightgray", color="black", size=1),  # Add border around facet labels
        axis_text=element_text(size=10),                         # Adjust axis text size
        axis_title=element_text(size=12),                        # Adjust axis title size
        legend_background=element_rect(fill="white", color=None) # White legend background
    )
    
    p1 += labs(title="Summary Plot", x=group_column, y=value_column)
    return p1
