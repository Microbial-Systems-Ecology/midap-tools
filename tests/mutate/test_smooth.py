import pytest
import pandas as pd
import numpy as np
from typing import Union, Tuple
from pandas.testing import assert_series_equal
from mutate.smooth import smooth_svagol, smooth_linear, smooth_loess_optimized, smooth_lowess_fast

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'trackID': [1, 1, 1, 1, 1, 1, 1, 1,
                    1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'frame': [1, 2, 3, 4, 5, 6, 7, 8, 9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9],
        'value': [10, 15, 20, 25, 30, 35, 40, 45, 50,
                  20, 25, 20, 10, 70, 80, 90, 110, 120]
    })


def test_smooth_svagol(sample_df):
    expected_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                       27.0, 13.0, 14.0, 26.7142, 55.4285, 83.4285, 92.5714, 106.2857, 121.4285]
    analyzed = smooth_svagol(sample_df, integration_window= 5, id_column='trackID', x_column= "value")
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["value_smoothed"], pd.Series(expected_values), check_names=False, check_exact=False)
    
def test_smooth_linear(sample_df):
    expected_values = [13.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 44.0, 47.0,
                       21.0, 19.0, 29.0, 41.0, 54.0, 72.0, 94.0, 104.0, 112.0]
    analyzed = smooth_linear(sample_df, integration_window= 5, id_column='trackID', x_column= "value")
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["value_smoothed"], pd.Series(expected_values), check_names=False, check_exact=False)
    
def test_smooth_loess_optimized(sample_df):
    expected_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                       20.1721, 24.3497, 20.0, 10.0, 70.0, 80.0, 90.0, 107.3990, 120.6792]
    analyzed = smooth_loess_optimized(sample_df, integration_window= 5, id_column='trackID', x_column= "value")
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["value_smoothed"], pd.Series(expected_values), check_names=False, check_exact=False)
    
def test_smooth_lowess_fast(sample_df):
    expected_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                       21.1415, 21.3227, 20.0, 45.0, 70, 80.0, 93.0285, 106.8917, 120.9582]
    analyzed = smooth_lowess_fast(sample_df, integration_window= 5, id_column='trackID', x_column= "value")
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["value_smoothed"], pd.Series(expected_values), check_names=False, check_exact=False)