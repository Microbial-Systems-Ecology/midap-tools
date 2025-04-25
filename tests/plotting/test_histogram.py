import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from matplotlib import pyplot as plt
from unittest.mock import patch
from plotting.histogram import plot_histogram 


@pytest.fixture
def df():
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 3, 4, 5, 6]
    })


@pytest.fixture
def df_dict(df):
    return {'A': df.copy(), 'B': df.copy()}


@patch("matplotlib.pyplot.show")
def test_plot_histogram_single_column(mock_show, df):
    plot_histogram(df, column='col1')
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_histogram_multiple_columns(mock_show, df):
    plot_histogram(df, column=['col1', 'col2'])
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_histogram_with_dict(mock_show, df_dict):
    plot_histogram(df_dict, column='col1')
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_histogram_with_dict_and_multiple_columns(mock_show, df_dict):
    plot_histogram(df_dict, column=['col1', 'col2'])
    assert mock_show.called


def test_plot_histogram_invalid_column_raises(df):
    with pytest.raises(ValueError, match="Column 'invalid' not found"):
        plot_histogram(df, column='invalid')