#!/bin/bash
# Pinned engine environment for every worker; matches the Lightning runtime used by the
# reviewed runs (CPython 3.12, numpy 1.26.4, scipy 1.11.4, clarabel 0.11.1). Idempotent.
set -euo pipefail
ENV=${1:-/tmp/agopenv}
if [ ! -x "$ENV/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv -q -p 3.12 "$ENV" 2>/dev/null || uv venv -q "$ENV"
    uv pip install -q -p "$ENV/bin/python" numpy==1.26.4 scipy==1.11.4 clarabel==0.11.1
  else
    python3 -m venv "$ENV"
    "$ENV/bin/pip" install -q numpy==1.26.4 scipy==1.11.4 clarabel==0.11.1
  fi
fi
"$ENV/bin/python" -c "import sys,os,numpy,scipy,clarabel;print('env ok: python',sys.version.split()[0],'numpy',numpy.__version__,'scipy',scipy.__version__,'clarabel',clarabel.__version__,'cpus',len(os.sched_getaffinity(0)))"
