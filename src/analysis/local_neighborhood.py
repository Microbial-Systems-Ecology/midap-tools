import numpy as np
import pandas as pd

def compute_neighborhood_segmentation(data: dict, masks, neighborhood_prefix: str = "density_", distance_threshold: int =50):
    """
    Computes neighborhood segmentation densities for a set of target points within a given distance threshold.
    This function calculates the density of overlap between a circular region around each target point and a set of binary masks. 
    The result is added as new columns to the target DataFrame, with one column per mask.

    Parameters:
    -----------
    data : dict of pd.DataFrame
        A dictionary where keys are target identifiers and values are pandas DataFrames. 
        Each DataFrame must contain the following columns:
        - 'frame': The frame index (int).
        - 'x': The x-coordinate of the target point (float or int).
        - 'y': The y-coordinate of the target point (float or int).

    masks : dict of np.ndarray
        A dictionary where keys are source identifiers and values are 3D numpy arrays representing binary masks.
        Each mask has the shape (frames, height, width), where:
        - frames: The number of frames in the mask.
        - height: The height of the mask.
        - width: The width of the mask.

    neighborhood_prefix : str, optional
        A prefix for the column names added to the target DataFrames. Default is "density_".

    distance_threshold : int, optional
        The rixeadius of the circular region (in pls) used to compute the neighborhood density. Default is 50.

    Returns:
    --------
    dict of pd.DataFrame
        The input `data` dictionary with updated DataFrames. Each DataFrame will have additional columns, 
        one for each mask in `masks`. The column names are prefixed with `neighborhood_prefix` followed by the mask key.

    Notes:
    ------
    - The function uses a circular mask to define the neighborhood region around each target point.
    - The density is calculated as the ratio of the overlapping area between the circular region and the binary mask 
      to the total area of the circular region.
    - If the circular region extends beyond the bounds of the mask, it is clipped to fit within the mask dimensions.
    """
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

            colname = f'{neighborhood_prefix}{source_key}'
            target_df[colname] = densities

    return data