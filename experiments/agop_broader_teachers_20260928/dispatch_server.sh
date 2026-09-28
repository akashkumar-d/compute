#!/usr/bin/env bash
# Only this entrypoint and launcher/launch.py are approved for running this bundle.
set -euo pipefail
if [[ "$(uname -s)" != "Linux" || "${AGOP_EXECUTION_SITE:-}" != "SERVER" ]]; then
  echo "Refusing execution: requires Linux and AGOP_EXECUTION_SITE=SERVER" >&2
  exit 2
fi
cd "$(dirname "$0")"
if [[ -e execution || -e runtime || -e SERVER_PREFLIGHT.json ]]; then
  echo "Refusing to overwrite an existing execution, runtime, or preflight" >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
timeout --kill-after=5 180 python -m venv --system-site-packages runtime
timeout --kill-after=5 180 runtime/bin/python -m pip install --disable-pip-version-check -r requirements.txt
runtime/bin/python - <<'PYENV'
import importlib.metadata, json, platform, sys
from pathlib import Path
record={"python":sys.version,"platform":platform.platform(),"packages":{k:importlib.metadata.version(k) for k in ("numpy","scipy","clarabel")}}
with Path("SERVER_ENVIRONMENT.json").open("x") as f: json.dump(record,f,indent=2)
PYENV
timeout --kill-after=5 95 runtime/bin/python checks/server_preflight.py --out SERVER_PREFLIGHT.json --seconds 90
set +e
runtime/bin/python launcher/launch.py
batch_code=$?
set -e
if [[ -d execution ]]; then
  runtime/bin/python analysis/summarize.py --execution execution --out execution/summary.json
fi
exit "$batch_code"
