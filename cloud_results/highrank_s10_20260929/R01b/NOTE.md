# R01b note (setup session)

Declared rerun of 20 R01 arms (see ../R01/NOTE.md). All 20 finished with exit 0.
While this shard ran, the runner computed each finished arm's canonical record on the whole
execution directory; for two arms (h2 r2 seed 9352, h2 r4 seed 9356) the summarizer saw files of a
still-running arm change and correctly marked those two records `input_snapshot_changed`.
Those two records were recomputed after the shard finished on a frozen view containing only
finished arms (the runner now always does this) and replaced; nothing else changed.
