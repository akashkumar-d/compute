#!/bin/bash
# Print shard progress and push any newly finished arms. Usage: poll.sh JOB SHARD
JOB=$1; SHARD=$2
REPO=$(git rev-parse --show-toplevel)
python3 - "$REPO/cloud_results/$JOB/$SHARD/SHARD_STATUS.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print('status: not written yet', exc); sys.exit(0)
print(f"state={d['state']} done={d['arms_done']}/{d['arms_total']} active={d['arms_active']} "
      f"pending={len(d['arms_pending'])} elapsed_min={d['elapsed_seconds']/60:.1f}")
for o in d['outcomes'][-4:]:
    print('  ', o['id'], 'rc', o['returncode'], 'stop', o['stop_reason'], 'indicator', o['completion_indicator_exists'],
          f"{o['elapsed_seconds']:.0f}s")
PY
if ! pgrep -f "run_shard.py --job $JOB --shard $SHARD" >/dev/null; then echo "runner process: not running"; else echo "runner process: running"; fi
tail -n 2 "/tmp/shard_$SHARD.log" 2>/dev/null
bash "$(dirname "$0")/sync_results.sh" "$JOB" "$SHARD"
