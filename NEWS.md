# NEWS

## v1.3.0.9000 (dev)

- Implemented `align_frames(frame_column, out_column)`: aligns all positions to their shared frame range (intersection across positions) and writes reindexed values to `out_column` (default: `"frame_aligned"`). In new-column mode all rows are kept; rows outside the shared range receive `NaN`. Set `out_column="frame"` to overwrite in place (drops rows, shifts `first_frame`/`last_frame`).
- Implemented `apply_offset_correction(offsets, by_group, frame_column, out_column)`: shifts frame values forward for selected positions, writing results to `out_column` (default: `"frame_offset"`). In new-column mode the original `frame` column is preserved for bitmap/HDF5 indexing. Set `out_column="frame"` to overwrite in place. Accepts a dict of `position → offset` (direct mode) or `group_value → offset` (by_group mode via metadata).
- Implemented `create_new_lineage(start_frame, output_column, orphans_as_root)`: rebuilds cell lineage from a given frame index. Cells present at `start_frame` become lineage roots; all descendants inherit the root's `trackID` as `lineageID_postfix`. Rows before `start_frame` receive `NA`. Orphan trackIDs (appearing after `start_frame` with no traceable ancestor) are self-rooted by default.
