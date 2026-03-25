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


# ---------------------------------------------------------------------------
# Default new-column mode
# ---------------------------------------------------------------------------

def test_new_column_created():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    assert "frame_aligned" in exp.data["pos1"]["GFP"].columns
    assert "frame_aligned" in exp.data["pos2"]["GFP"].columns


def test_original_frame_column_unchanged():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    assert list(exp.data["pos1"]["GFP"]["frame"]) == list(range(6))
    assert list(exp.data["pos2"]["GFP"]["frame"]) == list(range(3, 8))


def test_all_rows_kept_in_new_column_mode():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    assert len(exp.data["pos1"]["GFP"]) == 6
    assert len(exp.data["pos2"]["GFP"]) == 5


def test_rows_inside_shared_range_reindexed():
    """pos1: 0-5, pos2: 3-7 → shared 3-5 → frame_aligned 0,1,2 for those rows."""
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    inside_pos1 = exp.data["pos1"]["GFP"]
    inside_pos1 = inside_pos1[inside_pos1["frame"].isin([3, 4, 5])]
    assert list(inside_pos1["frame_aligned"]) == [0.0, 1.0, 2.0]


def test_rows_outside_shared_range_get_nan():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    outside = exp.data["pos1"]["GFP"][exp.data["pos1"]["GFP"]["frame"] < 3]
    assert outside["frame_aligned"].isna().all()


def test_aligned_column_consistent_across_positions():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    vals_pos1 = set(exp.data["pos1"]["GFP"]["frame_aligned"].dropna())
    vals_pos2 = set(exp.data["pos2"]["GFP"]["frame_aligned"].dropna())
    assert vals_pos1 == vals_pos2


def test_custom_out_column():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames(out_column="my_aligned")
    assert "my_aligned" in exp.data["pos1"]["GFP"].columns
    assert "frame_aligned" not in exp.data["pos1"]["GFP"].columns


# ---------------------------------------------------------------------------
# Overwrite mode (out_column="frame")
# ---------------------------------------------------------------------------

def test_overwrite_mode_drops_rows_outside_range():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames(out_column="frame")
    assert len(exp.data["pos1"]["GFP"]) == 3
    assert len(exp.data["pos2"]["GFP"]) == 3


def test_overwrite_mode_all_positions_same_frame_values():
    exp = _make_experiment(
        positions=["pos1", "pos2", "pos3"], channels=["GFP"],
        frames_per_pos={
            "pos1": list(range(10)),
            "pos2": list(range(4, 12)),
            "pos3": list(range(2, 9)),
        },
    )
    exp.align_frames(out_column="frame")
    frame_sets = [set(exp.data[p]["GFP"]["frame"]) for p in exp.positions]
    assert len(set(frozenset(s) for s in frame_sets)) == 1


def test_overwrite_mode_frames_start_at_zero():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames(out_column="frame")
    assert exp.data["pos1"]["GFP"]["frame"].min() == 0
    assert exp.data["pos2"]["GFP"]["frame"].min() == 0


# ---------------------------------------------------------------------------
# Multiple channels
# ---------------------------------------------------------------------------

def test_multiple_channels_all_get_aligned_column():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP", "TXRED"],
        frames_per_pos={"pos1": list(range(6)), "pos2": list(range(3, 8))},
    )
    exp.align_frames()
    for c in ["GFP", "TXRED"]:
        assert "frame_aligned" in exp.data["pos1"][c].columns
        assert "frame_aligned" in exp.data["pos2"][c].columns


# ---------------------------------------------------------------------------
# No overlap → ValueError
# ---------------------------------------------------------------------------

def test_no_overlap_raises():
    exp = _make_experiment(
        positions=["pos1", "pos2"], channels=["GFP"],
        frames_per_pos={"pos1": [0, 1, 2], "pos2": [5, 6, 7]},
    )
    with pytest.raises(ValueError, match="No common frame range"):
        exp.align_frames()
