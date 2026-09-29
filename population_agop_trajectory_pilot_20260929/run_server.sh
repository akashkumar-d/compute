#!/usr/bin/env bash
# Explicit one worker only. Parent must verify external concurrency and account.
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:?Specify smoke or arm explicitly}"; shift
case "$mode" in
  smoke) term_seconds=175 ;;
  arm) term_seconds=1985 ;;
  *) printf '%s\n' 'Only smoke or arm mode supported' >&2; exit 2 ;;
esac
export AGOP_EXECUTION_SITE=SERVER
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
pilot_python="../../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python"
test -x "$pilot_python"
exec timeout --signal=TERM --kill-after=5 "${term_seconds}s" nice -n 10 "$pilot_python" -B run.py "$mode" "$@"
