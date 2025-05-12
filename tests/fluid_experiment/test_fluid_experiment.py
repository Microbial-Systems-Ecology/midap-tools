import pytest
import shutil
import pandas as pd
from unittest.mock import patch
from pathlib import Path
from fluid_experiment.fluid_experiment import FluidExperiment
import requests, zipfile, io

# filepath: src/fluid_experiment/test_fluid_experiment.py

# URL for example data
EXAMPLE_DATA_URL = "https://polybox.ethz.ch/index.php/s/tD8gJpFS6mgS9A4/download?path=%2F&files=midap_tools_testdata.zip"
EXAMPLE_DATA_PATH = Path("test_data")


@pytest.fixture(scope="module")
def setup_test_data():
    """
    Fixture to download and extract test data if not already present.
    """
    if not EXAMPLE_DATA_PATH.exists():
        print("Downloading test data...")
        response = requests.get(EXAMPLE_DATA_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(EXAMPLE_DATA_PATH)
    yield EXAMPLE_DATA_PATH
    # Cleanup after tests
    shutil.rmtree(EXAMPLE_DATA_PATH)

@pytest.fixture
def fluid_experiment(setup_test_data):
    """
    Fixture to create a FluidExperiment object for testing.
    """
    path = setup_test_data / "midap_tools_testdata"
    color_channels = ["GFP", "TXRED"]
    return FluidExperiment(path=path, color_channels=color_channels)

def test_initialization(fluid_experiment):
    """
    Test the initialization of FluidExperiment.
    """
    assert fluid_experiment.path.exists()
    assert len(fluid_experiment.positions) > 0
    assert len(fluid_experiment.color_channels) == 2

def test_from_copy(fluid_experiment):
    """
    Test the from_copy method.
    """
    copied_experiment = FluidExperiment.from_copy(fluid_experiment)
    assert copied_experiment is not fluid_experiment
    assert copied_experiment.positions == fluid_experiment.positions
    assert copied_experiment.color_channels == fluid_experiment.color_channels

def test_drop_positions(fluid_experiment):
    """
    Test the drop_positions method.
    """
    initial_positions = fluid_experiment.positions.copy()
    fluid_experiment.drop_positions([initial_positions[0]])
    assert len(fluid_experiment.positions) == len(initial_positions) - 1
    assert initial_positions[0] not in fluid_experiment.positions

def test_drop_color_channels(fluid_experiment):
    """
    Test the drop_color_channels method.
    """
    initial_channels = fluid_experiment.color_channels.copy()
    fluid_experiment.drop_color_channels([initial_channels[0]])
    assert len(fluid_experiment.color_channels) == len(initial_channels) - 1
    assert initial_channels[0] not in fluid_experiment.color_channels

def test_save_and_load(fluid_experiment, tmp_path):
    """
    Test the save and load methods.
    """
    save_path = tmp_path / "test_experiment.h5"
    print(save_path)
    fluid_experiment.save(save_path)
    assert save_path.exists()

    loaded_experiment = FluidExperiment.load(save_path)
    assert loaded_experiment.positions == fluid_experiment.positions
    assert loaded_experiment.color_channels == fluid_experiment.color_channels

def test_filter_data(fluid_experiment):
    """
    Test the filter_data method.
    """
    column = "area"
    fluid_experiment.filter_data(column=column, min_value=10)
    for position in fluid_experiment.positions:
        for channel in fluid_experiment.color_channels:
            assert all(fluid_experiment.data[position][channel][column] >= 10)

def test_calculate_growth_rate(fluid_experiment):
    """
    Test the calculate_growth_rate method.
    """
    fluid_experiment.calculate_growth_rate(
        integration_window=5,
        id_column="trackID",
        value_column="area"
    )
    for position in fluid_experiment.positions:
        for channel in fluid_experiment.color_channels:
            assert "growth_rate" in fluid_experiment.data[position][channel].columns
            
def test_calculate_local_neighborhood(fluid_experiment):
    """
    Test the calculate_local_neighborhood method.
    """
    fluid_experiment.calculate_local_neighborhood(distance_threshold=50)
    for p in fluid_experiment.positions:
        for c in fluid_experiment.color_channels:
            assert any(col.startswith("density_") for col in fluid_experiment.data[p][c].columns)

def test_calculate_transform_data(fluid_experiment):
    """
    Test the calculate_transform_data method.
    """
    fluid_experiment.calculate_transform_data(column="area", type="log")
    for p in fluid_experiment.positions:
        for c in fluid_experiment.color_channels:
            assert "area_log" in fluid_experiment.data[p][c].columns

def test_create_metadata_template(fluid_experiment, tmp_path):
    """
    Test the create_metdata_template method.
    """
    metadata_path = tmp_path / "metadata.csv"
    fluid_experiment.create_metadata_template(path=metadata_path, overwrite=True)
    assert metadata_path.exists()
    df = pd.read_csv(metadata_path)
    assert set(df.columns) == {"position", "group", "experiment", "device_channel"}

def test_load_metadata_template(fluid_experiment, tmp_path):
    """
    Test the load_metadata_template method.
    """
    metadata_path = tmp_path / "metadata.csv"
    fluid_experiment.create_metadata_template(path=metadata_path, overwrite=True)
    fluid_experiment.load_metadata_template(path=metadata_path)
    assert fluid_experiment.metadata is not None
    assert set(fluid_experiment.metadata.columns) == {"position", "group", "experiment", "device_channel"}

def test_save_select(fluid_experiment):
    """
    Test the _save_select method.
    """
    assert fluid_experiment._save_select("pos1") == ["pos1"]
    assert fluid_experiment._save_select(["pos1", "pos2"]) == ["pos1", "pos2"]

def test_update_information(fluid_experiment):
    """
    Test the _update_information method.
    """
    fluid_experiment._update_information()
    assert fluid_experiment.n_frames is not None
    assert fluid_experiment.headers is not None
    assert isinstance(fluid_experiment.unequal_lengths, bool)
    assert isinstance(fluid_experiment.unequal_header, bool)
    
    
def test_fuse_valid_experiments(fluid_experiment):
    """
    Test fusing two valid FluidExperiment objects with different positions and the same channels.
    """
    # Create a copy of the original experiment to fuse with
    other_experiment = FluidExperiment.from_copy(fluid_experiment)
    other_experiment.positions = ["Pos56", "Pos57"]

    # Fuse the experiments
    fluid_experiment.fuse(other_experiment)

    # Check that the positions are correctly updated
    assert "Pos56" in fluid_experiment.positions
    assert "Pos57" in fluid_experiment.positions
    assert len(fluid_experiment.positions) == len(set(fluid_experiment.positions))

    # Check that the data and filter history for the new positions are added
    for pos in ["Pos56", "Pos57"]:
        assert pos in fluid_experiment.data
        assert pos in fluid_experiment.filter_history
        for channel in fluid_experiment.color_channels:
            assert channel in fluid_experiment.data[pos]
            assert channel in fluid_experiment.filter_history[pos]


def test_fuse_raises_error_on_wrong_type(fluid_experiment):
    """
    Test that fusing with an object of the wrong type raises a TypeError.
    """
    with pytest.raises(TypeError, match="The input must be an instance of FluidExperiment."):
        fluid_experiment.fuse("not_a_fluid_experiment")


def test_fuse_raises_error_on_different_channels(fluid_experiment):
    """
    Test that fusing two experiments with different channels raises a ValueError.
    """
    other_experiment = FluidExperiment.from_copy(fluid_experiment)
    other_experiment.color_channels = ["DifferentChannel"]

    with pytest.raises(ValueError, match="The color_channels of the two experiments are not identical."):
        fluid_experiment.fuse(other_experiment)


def test_fuse_preserves_original_names(fluid_experiment):
    """
    Test that fusing two experiments with different position names preserves the original names.
    """
    other_experiment = FluidExperiment.from_copy(fluid_experiment)
    other_experiment.positions = ["NewPos1", "NewPos2"]

    fluid_experiment.fuse(other_experiment)

    # Check that the new positions are added with their original names
    assert "NewPos1" in fluid_experiment.positions
    assert "NewPos2" in fluid_experiment.positions


def test_fuse_handles_duplicate_position_names(fluid_experiment):
    """
    Test that fusing two experiments with the same position names correctly adapts the names.
    """
    other_experiment = FluidExperiment.from_copy(fluid_experiment)

    # Fuse the experiments
    fluid_experiment.fuse(other_experiment)

    # Check that duplicate positions are renamed
    for pos in other_experiment.positions:
        assert any(pos in p for p in fluid_experiment.positions if p != pos)


def test_fuse_metadata(fluid_experiment):
    """
    Test that metadata is correctly fused or set to None if either experiment has no metadata.
    """
    # Case 1: Both experiments have metadata
    fluid_experiment.metadata = pd.DataFrame({"position": fluid_experiment.positions, "group": ["A"] * len(fluid_experiment.positions)})
    other_experiment = FluidExperiment.from_copy(fluid_experiment)
    other_experiment.metadata = pd.DataFrame({"position": other_experiment.positions, "group": ["B"] * len(other_experiment.positions)})

    fluid_experiment.fuse(other_experiment)

    assert fluid_experiment.metadata is not None
    assert set(fluid_experiment.metadata["group"]) == {"A", "B"}

    # Case 2: One experiment has no metadata
    fluid_experiment.metadata = None
    other_experiment.metadata = pd.DataFrame({"position": other_experiment.positions, "group": ["B"] * len(other_experiment.positions)})

    fluid_experiment.fuse(other_experiment)

    assert fluid_experiment.metadata is None
    
def test_fuse(fluid_experiment):
    """
    Test the fuse method.
    """
    # Create a copy of the original experiment to fuse with
    other_experiment = FluidExperiment.from_copy(fluid_experiment)

    # Modify the positions in the other experiment to simulate a different dataset
    other_experiment.positions = [f"{pos}_new" for pos in other_experiment.positions]
    for pos in other_experiment.positions:
        other_experiment.data[pos] = other_experiment.data[fluid_experiment.positions[0]].copy()
        other_experiment.filter_history[pos] = other_experiment.filter_history[fluid_experiment.positions[0]].copy()

    # Fuse the two experiments
    fluid_experiment.fuse(other_experiment)

    # Check that the positions have been updated correctly
    assert len(fluid_experiment.positions) == len(set(fluid_experiment.positions))
    assert all(pos in fluid_experiment.positions for pos in other_experiment.positions)

    # Check that the data and filter history for the new positions have been added
    for pos in other_experiment.positions:
        assert pos in fluid_experiment.data
        assert pos in fluid_experiment.filter_history
        for channel in fluid_experiment.color_channels:
            assert channel in fluid_experiment.data[pos]
            assert channel in fluid_experiment.filter_history[pos]

    # Check that metadata has been updated correctly if it exists
    if fluid_experiment.metadata is not None:
        assert all(pos in fluid_experiment.metadata["position"].values for pos in other_experiment.positions)