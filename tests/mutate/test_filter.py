import pytest
import pandas as pd
from typing import Union, Tuple
from mutate.filter import filter_tracks

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'trackID': [1, 1, 2, 2, 2, 3, 4, 5, 5, 6],
        'score': [10, 15, 20, 25, 30, 5, 50, 60, 65, 70]
    })

def test_no_filter_applied(sample_df):
    filtered, summary = filter_tracks(sample_df, column='trackID')
    assert len(filtered) == len(sample_df)
    assert summary['rows_before'] == summary['rows_after']
    assert summary['unique_values_before'] == summary['unique_values_after']

def test_min_occurrences_filter(sample_df):
    filtered, summary = filter_tracks(sample_df, column='trackID', min_occurences=2)
    expected_ids = [1, 2, 5]
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert all(filtered['trackID'].isin(expected_ids))
    assert summary['min_occurences'] == 2

def test_min_value_filter(sample_df):
    filtered, summary = filter_tracks(sample_df, column='score', min_value=30)
    assert all(filtered['score'] >= 30)
    assert summary['min_value'] == 30

def test_max_value_filter(sample_df):
    filtered, summary = filter_tracks(sample_df, column='score', max_value=25)
    assert all(filtered['score'] <= 25)
    assert summary['max_value'] == 25

def test_min_and_max_value_filter(sample_df):
    filtered, summary = filter_tracks(sample_df, column='score', min_value=20, max_value=60)
    assert all((filtered['score'] >= 20) & (filtered['score'] <= 60))
    assert summary['min_value'] == 20
    assert summary['max_value'] == 60

def test_combined_filters(sample_df):
    filtered, summary = filter_tracks(sample_df, column='trackID', min_occurences=2, min_value=2, max_value=5)
    # Only IDs 1, 2 and 5 occur >=2 times; but their values must be between 2 and 5
    expected_ids = [2,5]
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert summary['rows_after'] == 5

def test_empty_input():
    df = pd.DataFrame(columns=['trackID', 'score'])
    filtered, summary = filter_tracks(df, column='trackID')
    assert filtered.empty
    assert summary == {}