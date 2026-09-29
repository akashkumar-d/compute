# R01 note (setup session)

R01 ran 56 ReLU arms in the setup session with the pinned environment. At 02:22:38 UTC a plotting
install in the same session upgraded numpy inside that environment (1.26.4 -> 2.5.3). The 18 arms
started afterwards (h2 r4 seeds 9353-9360, h2 r2 seeds 9351-9360) failed at import, before any
training, and two h2 r4 arms (seeds 9351, 9352) were in their last seconds during the change.
Those 20 arms are rerun, unchanged, in the declared rerun shard R01b; the R01 copies are kept but
superseded. No other arm overlapped the change.

`canonical/a01.json` is the runner's original bundle-level output (its last update ran after the
change). `canonical/<arm>.json.gz` were recomputed afterwards for the 36 unaffected arms with the
restored pinned environment (numpy 1.26.4, scipy 1.11.4) and the same canonical summarizer
(57ca1a05...). Use the per-arm files. Raw arm packages are unchanged.
