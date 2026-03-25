import pandas as pd
import pytest
from fluid_experiment.fluid_experiment import FluidExperiment


def _make_experiment(positions, channels, frames_per_pos):
    """Build a minimal FluidExperiment without loading from disk."""
    exp = FluidExperiment.__new__(FluidExperiment)
    exp.path = "."
    exp.name = "test"
    exp.data_type = "family_machine"
    exp.positions = list(positions)
    exp.color_channels = list(channels)
    exp.metadata = None
    exp.filter_history = {p: {c: [] for c in channels} for p in positions}
    exp.file_paths = {p: "." for p in positions}
    exp.data = {}
    for p in positions:
        exp.data[p] = {}
        frames = frames_per_pos[p]
        for c in channels:
            exp.data[p][c] = pd.DataFrame({
                "frame":       frames,
                "trackID":     [1] * len(frames),
                "first_frame": [frames[0]] * len(frames),
                "last_frame":  [frames[-1]] * len(frames),
                "split":       [0] * len(frames),
            })
    exp._update_information()
    return exp


@pytest.fixture
def exp():
    return _make_experiment(
        positions=["pos1", "pos2", "pos3"],
        channels=["GFP"],
        frames_per_pos={
            "pos1": [0, 1, 2, 3, 4],
            "pos2": [0, 1, 2, 3, 4],
            "pos3": [0, 1, 2, 3, 4],
        },
    )


# ---------------------------------------------------------------------------
# Default new-column mode
# ---------------------------------------------------------------------------

def test_new_column_created(exp):
    exp.apply_offset_correction(offsets={"pos1": 3})
    assert "frame_offset" in exp.data["pos1"]["GFP"].columns


def test_new_column_values_shifted(exp):
    exp.apply_offset_correction(offsets={"pos1": 3})
    assert list(exp.data["pos1"]["GFP"]["frame_offset"]) == [3, 4, 5, 6, 7]


def test_original_frame_column_unchanged(exp):
    exp.apply_offset_correction(offsets={"pos1": 3})
    assert list(exp.data["pos1"]["GFP"]["frame"]) == [0, 1, 2, 3, 4]


def test_first_last_frame_unchanged_in_new_column_mode(exp):
    exp.apply_offset_correction(offsets={"pos1": 2})
    df = exp.data["pos1"]["GFP"]
    assert df["first_frame"].iloc[0] == 0
    assert df["last_frame"].iloc[0] == 4


def test_unspecified_position_gets_identity_column(exp):
    """Positions not in offsets must still receive out_column equal to frame."""
    exp.apply_offset_correction(offsets={"pos1": 3})
    assert "frame_offset" in exp.data["pos2"]["GFP"].columns
    assert list(exp.data["pos2"]["GFP"]["frame_offset"]) == list(exp.data["pos2"]["GFP"]["frame"])


def test_multiple_positions(exp):
    exp.apply_offset_correction(offsets={"pos1": 2, "pos2": 5})
    assert list(exp.data["pos1"]["GFP"]["frame_offset"]) == [2, 3, 4, 5, 6]
    assert list(exp.data["pos2"]["GFP"]["frame_offset"]) == [5, 6, 7, 8, 9]


def test_custom_out_column(exp):
    exp.apply_offset_correction(offsets={"pos1": 3}, out_column="my_frame")
    assert "my_frame" in exp.data["pos1"]["GFP"].columns
    assert "frame_offset" not in exp.data["pos1"]["GFP"].columns


# ---------------------------------------------------------------------------
# Overwrite mode
# ---------------------------------------------------------------------------

def test_overwrite_mode_shifts_frame(exp):
    exp.apply_offset_correction(offsets={"pos1": 3}, out_column="frame")
    assert list(exp.data["pos1"]["GFP"]["frame"]) == [3, 4, 5, 6, 7]


def test_overwrite_mode_shifts_first_last_frame(exp):
    exp.apply_offset_correction(offsets={"pos1": 2}, out_column="frame")
    df = exp.data["pos1"]["GFP"]
    assert df["first_frame"].iloc[0] == 2
    assert df["last_frame"].iloc[0] == 6


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_invalid_position_raises(exp):
    with pytest.raises(ValueError, match="not found in experiment"):
        exp.apply_offset_correction(offsets={"nonexistent": 3})


# ---------------------------------------------------------------------------
# By-group mode
# ---------------------------------------------------------------------------

@pytest.fixture
def exp_with_metadata():
    exp = _make_experiment(
        positions=["pos1", "pos2", "pos3"],
        channels=["GFP"],
        frames_per_pos={
            "pos1": [0, 1, 2, 3, 4],
            "pos2": [0, 1, 2, 3, 4],
            "pos3": [0, 1, 2, 3, 4],
        },
    )
    exp.metadata = pd.DataFrame({
        "position":       ["pos1", "pos2", "pos3"],
        "group":          ["Cond1", "Cond1", "Cond2"],
        "experiment":     ["e1", "e1", "e1"],
        "device_channel": ["d1", "d1", "d1"],
    }).set_index("position")
    return exp


def test_by_group_shifts_matching_positions(exp_with_metadata):
    exp_with_metadata.apply_offset_correction(offsets={"Cond1": 5}, by_group="group")
    assert list(exp_with_metadata.data["pos1"]["GFP"]["frame_offset"]) == [5, 6, 7, 8, 9]
    assert list(exp_with_metadata.data["pos2"]["GFP"]["frame_offset"]) == [5, 6, 7, 8, 9]


def test_by_group_unspecified_group_gets_identity_column(exp_with_metadata):
    """Positions whose group is not in offsets must still receive out_column equal to frame."""
    exp_with_metadata.apply_offset_correction(offsets={"Cond1": 5}, by_group="group")
    assert "frame_offset" in exp_with_metadata.data["pos3"]["GFP"].columns
    assert list(exp_with_metadata.data["pos3"]["GFP"]["frame_offset"]) == list(exp_with_metadata.data["pos3"]["GFP"]["frame"])


def test_by_group_multiple_groups(exp_with_metadata):
    exp_with_metadata.apply_offset_correction(
        offsets={"Cond1": 3, "Cond2": 7}, by_group="group"
    )
    assert list(exp_with_metadata.data["pos1"]["GFP"]["frame_offset"]) == [3, 4, 5, 6, 7]
    assert list(exp_with_metadata.data["pos3"]["GFP"]["frame_offset"]) == [7, 8, 9, 10, 11]


def test_by_group_without_metadata_raises(exp):
    with pytest.raises(ValueError, match="Metadata must be loaded"):
        exp.apply_offset_correction(offsets={"Cond1": 5}, by_group="group")
