import pytest
import pandas as pd
import numpy as np
from typing import Union, Tuple
from pandas.testing import assert_series_equal
from analysis.growth_rate import calculate_growth_rate, calculate_min_max_deltas, calculate_min_max_deltas_nona

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'trackID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'value': [10, 15, 20, 25, 30, 20, 25, 20, 10, 70]
    })


def test_calculate_growth_rate(sample_df):
    expected_values = [5.0, 5.0, 5.0, np.nan, np.nan, 0.0, -7.5, 25.0, np.nan, np.nan]
    analyzed = calculate_growth_rate(sample_df, integration_window= 2, id_column='trackID', value_column= "value")
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["growth_rate"], pd.Series(expected_values), check_names=False)

def test_calculate_growth_rate_centric(sample_df):
    expected_values = [np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, 12.5, np.nan, np.nan]
    analyzed = calculate_growth_rate(sample_df, integration_window= 5, id_column='trackID', value_column= "value", centric = True)
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["growth_rate"], pd.Series(expected_values), check_names=False)

def test_calculate_min_max_deltas(sample_df):
    expected_mins = [np.nan, 5.0, 5.0, 5.0, np.nan, np.nan, -5.0, -10.0, -10.0, np.nan]
    expected_maxes = [np.nan, 5.0, 5.0, 5.0, np.nan, np.nan, 5.0, -5.0, 60.0, np.nan]
    analyzed = calculate_min_max_deltas(sample_df, integration_window= 2, id_column='trackID', value_column= "value", centric = True)
    assert len(analyzed) == len(sample_df)
    assert_series_equal(analyzed["delta_max"], pd.Series(expected_maxes), check_names=False)
    assert_series_equal(analyzed["delta_min"], pd.Series(expected_mins), check_names=False)


if __name__ == "__main__":
    df = pd.DataFrame({
        'trackID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'value': [10, 15, 20, 25, 30, 20, 25, 20, 10, 70]
    })
    
    analyzed = calculate_min_max_deltas_nona(df, integration_window= 5, id_column='trackID', value_column= "value", centric = True)
    print(analyzed)