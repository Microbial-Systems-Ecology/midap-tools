import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from unittest.mock import patch
from plotting.histogram import plot_value_count_histogram  


@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        'cat': ['a', 'b', 'a', 'c', 'b', 'a'],
        'int': [1, 2, 1, 3, 2, 1], 
        'float': [0.1, 0.2, 0.1, 0.3, 0.2, 0.1] 
    })


@pytest.fixture
def df_dict_mixed(df_mixed):
    return {'A': df_mixed.copy(), 'B': df_mixed.copy()}


@patch("matplotlib.pyplot.show")
def test_value_counts_single_categorical_column(mock_show, df_mixed):
    plot_value_count_histogram(df_mixed, column='cat')
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_value_counts_single_integer_column(mock_show, df_mixed):
    plot_value_count_histogram(df_mixed, column='int')
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_value_counts_multiple_columns_mixed_types(mock_show, df_mixed):
    plot_value_count_histogram(df_mixed, column=['cat', 'int'])
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_value_counts_dict_input_with_integer_data(mock_show, df_dict_mixed):
    plot_value_count_histogram(df_dict_mixed, column=['cat', 'int'])
    assert mock_show.called


def test_value_counts_invalid_column_raises(df_mixed):
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        plot_value_count_histogram(df_mixed, column='nonexistent')