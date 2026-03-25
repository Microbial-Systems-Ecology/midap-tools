# Development Summary — 2026-03-25 - v1.3.0.9000

## Overview

First development session for midap-tools. Established project conventions and implemented three new features in the `FluidExperiment` class.

## Setup

- Created `CLAUDE.md` at repo root documenting architecture, conventions, column schema, and rules for adding new methods (docstring updates, NEWS.md entries).
- Created `NEWS.md` for changelog tracking under version `v1.3.0.9000 (dev)`.
- Corrected `split` column description: it is a boolean flag (0/1 per row), not a frame index.

## Features Implemented

### 1. `create_new_lineage(start_frame, output_column, orphans_as_root)`
Rebuilds cell lineage from a given frame index. Cells present at `start_frame` become lineage roots; all descendants (via BFS through `trackID_d1`/`trackID_d2`) inherit the root's `trackID` as `lineageID_postfix`. Rows before `start_frame` receive `NaN`. Orphan trackIDs appearing after `start_frame` with no traceable ancestor are self-rooted by default (`orphans_as_root=True`).

- Pure function: `src/analysis/lineage.py`
- Tests: `tests/analysis/test_lineage.py`

### 2. `apply_offset_correction(offsets, by_group, frame_column, out_column)`
Shifts frame values forward for selected positions. By default writes to a new `frame_offset` column, preserving the original `frame` column for bitmap/HDF5 indexing. All positions (including unspecified ones) always receive the output column — unspecified positions get `frame_offset = frame` (identity). Supports direct `{position: offset}` dict or `{group_value: offset}` via metadata column. Set `out_column="frame"` to overwrite in place.

- Pure function: `src/mutate/offset.py`
- Tests: `tests/mutate/test_offset.py`, `tests/fluid_experiment/test_offset_correction.py`

### 3. `align_frames(frame_column, out_column)`
Aligns all positions to their shared frame range (intersection across all positions/channels) and writes reindexed values to `out_column` (default: `frame_aligned`). In new-column mode all rows are kept; rows outside the shared range receive `NaN`. Set `out_column="frame"` to overwrite in place (drops rows outside range, shifts and clamps `first_frame`/`last_frame`).

- Pure function: `src/mutate/align.py`
- Tests: `tests/mutate/test_align.py`, `tests/fluid_experiment/test_align_frames.py`

## Bug Fix

`apply_offset_correction` was not creating the output column for positions not listed in `offsets`. Fixed by iterating over all positions and applying offset=0 for unlisted ones, ensuring consistent column presence across the experiment.
