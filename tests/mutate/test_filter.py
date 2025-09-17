import pytest
import pandas as pd
from typing import Union, Tuple
from mutate.filter import filter_by_column, filter_id_by_data_ranges

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'trackID': [1, 1, 2, 2, 2, 3, 4, 5, 5, 6],
        'score': [10, 15, 20, 25, 30, 5, 50, 60, 65, 70]
    })

# FILTER BY COLUMN TESTS

def test_no_filter_applied(sample_df):
    filtered, summary = filter_by_column(sample_df, column='trackID')
    assert len(filtered) == len(sample_df)
    assert summary['rows_before'] == summary['rows_after']
    assert summary['unique_values_before'] == summary['unique_values_after']

def test_min_occurrences_filter(sample_df):
    filtered, summary = filter_by_column(sample_df, column='trackID', min_occurences=2)
    expected_ids = [1, 2, 5]
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert all(filtered['trackID'].isin(expected_ids))
    assert summary['min_occurences'] == 2

def test_min_value_filter(sample_df):
    filtered, summary = filter_by_column(sample_df, column='score', min_value=30)
    assert all(filtered['score'] >= 30)
    assert summary['min_value'] == 30

def test_max_value_filter(sample_df):
    filtered, summary = filter_by_column(sample_df, column='score', max_value=25)
    assert all(filtered['score'] <= 25)
    assert summary['max_value'] == 25

def test_min_and_max_value_filter(sample_df):
    filtered, summary = filter_by_column(sample_df, column='score', min_value=20, max_value=60)
    assert all((filtered['score'] >= 20) & (filtered['score'] <= 60))
    assert summary['min_value'] == 20
    assert summary['max_value'] == 60

def test_combined_filters(sample_df):
    filtered, summary = filter_by_column(sample_df, column='trackID', min_occurences=2, min_value=2, max_value=5)
    # Only IDs 1, 2 and 5 occur >=2 times; but their values must be between 2 and 5
    expected_ids = [2,5]
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert summary['rows_after'] == 5

def test_empty_input():
    df = pd.DataFrame(columns=['trackID', 'score'])
    filtered, summary = filter_by_column(df, column='trackID')
    assert filtered.empty
    assert summary == {}
    

# FILTER ID BY DATA RANGES TESTS    

def test_id_filter_no_filter_applied(sample_df):
    filtered, summary = filter_id_by_data_ranges(sample_df, column='score')
    assert len(filtered) == len(sample_df)
    assert summary['rows_before'] == summary['rows_after']
    assert summary['unique_ids_before'] == summary['unique_ids_after']

def test_id_filter_min_occurrences_filter(sample_df):
    filtered, summary = filter_id_by_data_ranges(sample_df, column='score', min_occurences=2)
    expected_ids = [1, 2, 5]
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert all(filtered['trackID'].isin(expected_ids))
    assert summary['min_occurences'] == 2
    assert summary['rows_after'] == 7
    assert len(filtered) == 7

def test_id_filter_min_value_filter(sample_df):
    filtered, summary = filter_id_by_data_ranges(sample_df, column='score', min_value=30)
    expected_ids = [4,5,6]
    assert all(filtered['score'] >= 30)
    assert summary['min_value'] == 30
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert summary['rows_after'] == 4
    assert len(filtered) == 4

def test_id_filter_max_value_filter(sample_df):
    filtered, summary = filter_id_by_data_ranges(sample_df, column='score', max_value=25)
    expected_ids = [1,3]
    assert all(filtered['score'] <= 25)
    assert summary['max_value'] == 25
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert summary['rows_after'] == 3
    assert len(filtered) == 3


def test_id_filter_min_and_max_value_filter(sample_df):
    filtered, summary = filter_id_by_data_ranges(sample_df, column='score', min_value=20, max_value=60)
    expected_ids = [2,4]
    assert all((filtered['score'] >= 20) & (filtered['score'] <= 60))
    assert summary['min_value'] == 20
    assert summary['max_value'] == 60
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert summary['rows_after'] == 4
    assert len(filtered) == 4


def test_id_filter_combined_filters(sample_df):
    filtered, summary = filter_id_by_data_ranges(sample_df, column='score', min_occurences=2, min_value=11, max_value=69)
    # Only IDs 1, 2 and 5 occur >=2 times; but their values must be between 11 and 69
    expected_ids = [2,5]
    assert set(filtered['trackID'].unique()) == set(expected_ids)
    assert summary['rows_after'] == 5
    assert summary['filtered_ids_by_min_occurences'] == 3
    assert summary['filtered_ids_by_value_range'] == 1
    assert len(filtered) == 5

def test_id_filter_empty_input():
    df = pd.DataFrame(columns=['trackID', 'score'])
    filtered, summary = filter_id_by_data_ranges(df, column='trackID')
    assert filtered.empty
    assert summary == {}    
