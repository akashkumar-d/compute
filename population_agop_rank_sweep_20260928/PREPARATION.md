# Rank-sweep preparation

The remote-only bundle is frozen for review at bundle/. The independently authored DESIGN.json owns the exact32-arm numerical design. No code in the scientific ReLU/SwiGLU engines was changed. No local training, model evaluation, network access, publication or remote launch was performed in this workstream.

The prepared scheduler admits28 single-thread rank workers on at least32 allocated CPUs, reserving2 slots for the v9 validation pair and2 free CPUs. It enforces nice>=10,32GiB available memory,10GiB free disk,1990s hard arm caps (1800training+180diagnostics+10cleanup), and the parent's revised3600s global cap. The fixed queue starts rank16 then8,4,2; seed/teacher/student order alternates architectures within each rank. Last-wave caps/unstarted arms are retained.

Independent runtime/source/config review is required before execution. REVIEW.json remains pending; publication and launch belong to the parent. See bundle/README.md for exact safe static commands, the eventual server-side command, preservation/restart rules and required analysis.

Final manifest SHA256: `e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01`. Design SHA256: `9af41b663760bf5a9bcc4dea18e2f50921c947b33fd001c964b4cbe135066ea7`. Static validation and24 standard-library/mock tests passed;31 scientific files are byte-identical and79 frozen hashes verified. No execution directory was created. `QA.json` records these checks.
