#!/bin/bash
# Commit and push this shard's results to the session's own branch. Safe to call repeatedly.
# Usage: sync_results.sh JOB SHARD [tag]
set -uo pipefail
JOB=$1; SHARD=$2; TAG=${3:-progress}
REPO=$(git rev-parse --show-toplevel); cd "$REPO" || exit 2
BR=$(git branch --show-current)
case "$BR" in ""|claude-workers|codex/*|main|master) echo "[sync] refusing to push to branch '$BR'"; exit 2;; esac
[ -d "cloud_results/$JOB/$SHARD" ] || { echo "[sync] no results directory yet"; exit 0; }
git add "cloud_results/$JOB/$SHARD"
if git diff --cached --quiet; then echo "[sync] nothing new to push"; exit 0; fi
N=$(python3 -c "import json;print(json.load(open('cloud_results/$JOB/$SHARD/SHARD_STATUS.json'))['arms_done'])" 2>/dev/null || echo "?")
git commit -q -m "cloud results $JOB $SHARD: $N arms done ($TAG)"
for i in 1 2 3 4 5; do
  if git push -q origin "HEAD:$BR" 2>/tmp/cw_push_err; then echo "[sync] pushed ($N arms done) to $BR"; exit 0; fi
  sleep $((i*15)); git pull -q --rebase origin "$BR" 2>/dev/null || true
done
echo "[sync] push failed:"; cat /tmp/cw_push_err; exit 1
