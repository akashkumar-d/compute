# Cubic head-scaling execution preparation

The executable bundle is frozen for independent runtime review at bundle/. All six original configuration files are preserved byte for byte, with their already reviewed q=1,0.3,0.1 coupling and reused seeds641/642. All11 v8 scientific files and the frozen SOURCE_MANIFEST remain unchanged. No model evaluation, training, network access, publication or automatic dispatch was performed. Rank-sweep and live-job files were not modified.

The SERVER-only scheduler allows6 single-thread workers, requires at least32 effective allocated CPUs,12GiB available memory and5GiB free disk, verifies inherited nice>=10, and caps each arm at1990s (1800training+180diagnostics+10cleanup), with2150s global compute. Existing-output refusal, exact-design/source/config/review gates and a persistent restart journal preserve prior/partial attempts. The six-arm limit and exact frozen protocol are covered by meaningful mocked tests.

Before the parent manually dispatches, v9 validation must be reviewed and other live task training must be<=24 so the total including this batch is<=30, leaving2 CPUs free. The bundle does not globally coordinate the other jobs and does not auto-wait or launch on capacity changes. REVIEW.json is pending; the parent owns review and publication/launch. Exact static checks and eventual server command are in bundle/README.md.

Static validation and24 standard-library/mock tests passed (1.609s). Six original configs,11 scientific files and all34 frozen hashes were checked; no execution directory was created. Success criteria are unchanged; original raw/normalized losses and physical/normalized clocks remain distinct. The historical q=1 low-order runs cannot replace the matched common-order controls.

Manifest SHA256: `7042a39550d2499d656dd6ae5a3d8a7795d40df83641ae603518a795f98477e0`.
Design SHA256: `f7769bd338ee43de91d4cdc53fcb76a34e7ad76623a2fe7615b6156c02d3fa5d`.
