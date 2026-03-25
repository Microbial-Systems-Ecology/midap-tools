# NEWS

## v1.3.0.9000 (dev)

- Implemented `create_new_lineage(start_frame, output_column, orphans_as_root)`: rebuilds cell lineage from a given frame index. Cells present at `start_frame` become lineage roots; all descendants inherit the root's `trackID` as `lineageID_postfix`. Rows before `start_frame` receive `NA`. Orphan trackIDs (appearing after `start_frame` with no traceable ancestor) are self-rooted by default.
