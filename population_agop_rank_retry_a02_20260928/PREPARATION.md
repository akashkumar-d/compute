# Rank retry attempt2: prepared, pending review

The isolated `bundle/` contains exactly12 SwiGLU retries: ranks16/8/4 × h2/raw ReLU × seeds641/642. Selection is locked to the original manifest and frozen remote inventory. The inventory contains20 completion markers—16 ReLU and four rank2 SwiGLU—so none of those arms is queued. This includes the rank2 completion recorded after the stale scheduler snapshot. Completion markers are retry-selection evidence, not scientific-success evidence.

All12 config files,22 scientific Python files and31 total scientific/support files remain byte-identical to the reviewed original rank bundle. No horizon, update cap, quadrature, seed, teacher or success criterion changed. Original IDs remain intact, with explicit attempt2 provenance per manifest entry; the original study denominator stays32 and no new independent seeds are added. Old partial and interrupted artifacts were not modified, and selection does not depend on the local download.

Runtime is12 single-thread workers plus20 reserved slots on at least32 allocated CPUs,32GiB available memory and10GiB free disk;1990s per arm and2150s global. The launcher changes only resource/subset validation and its dedicated retry journal/output name; the existing execution/cleanup architecture is retained. Reference manifest/inventory hashes, exact original config bytes and scientific source hashes gate execution. The parent checks aggregate live capacity and stopped old jobs, independently reviews the manifest, publishes, and manually dispatches.

Static dry-run and all28 mocked tests pass; no model was imported or evaluated and no execution directory was created. One inherited test fixture required an explicit old output-directory argument after the new default changed; its correction did not alter runtime or science. Detailed checks and logs are in `QA.json`, `DRY_RUN.json` and `MOCK_TESTS.log`.

- Final manifest SHA256: `0b5058b12bc75d0f86276b1dac65c2416a27b9de31ab917f06175a3238a4aac6`.
- Original manifest SHA256: `e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01`.
- Remote inventory SHA256: `1307b4d1c51cbb757e765db10bb29d366a8a20b03fc3a7a5df40c29e456177eb`.
- New publication leaf: `population_agop_rank_retry_a02_20260928`; new job: `agop-ranks-retry-20260928-a02`; new output: `execution_remote_a02`.

`bundle/REVIEW.json` remains pending and no launch/publication/network call was performed. The unchanged scientific limitations and32-arm interpretation remain mandatory; retain both attempts and identify which complete trajectory supplies each reported result.
