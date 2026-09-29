#!/usr/bin/env bash
set -euo pipefail
# Invoke from the new lrun repository root; never resume an existing output.
AGOP_BUNDLE=population_agop_diagnostic_transfer_20260929
AGOP_PYTHON=../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python
export AGOP_EXECUTION_SITE=SERVER
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
"$AGOP_PYTHON" "$AGOP_BUNDLE/prepare_runtime.py"
exec timeout --signal=TERM --kill-after=5s 355s nice -n 10 \
  "$AGOP_PYTHON" "$AGOP_BUNDLE/probe.py" --output probe_a01
