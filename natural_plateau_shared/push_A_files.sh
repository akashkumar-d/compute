#!/bin/bash
# Push A's files (claude/plateau-review) from this folder to the shared GitHub branch.
# Run on your Mac:   bash push_A_files.sh            (uses/creates the clone at ~/compute)
#               or   bash push_A_files.sh /path/to/your/clone
# Copies ONLY files that A owns, so it never overwrites B's log, claims or results.
# Uses your own GitHub login (git credential helper / gh); it never stores credentials.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
CLONE="${1:-$HOME/compute}"
BRANCH="claude/focused-gauss-cy16t3"

if [ ! -d "$CLONE/.git" ]; then git clone https://github.com/akashkumar-d/compute.git "$CLONE"; fi
cd "$CLONE"
git checkout "$BRANCH"
git pull --rebase

DST="$CLONE/natural_plateau_shared"
mkdir -p "$DST/log" "$DST/code/tasks" "$DST/claims" "$DST/inbox" "$DST/results"

# A-owned files
for f in README.md TASKS.md START_HERE_account_B.md .gitignore push_A_files.sh log/A.md \
         code/natplat_server.py claims/README.md inbox/README.md results/README.md; do
  cp "$SRC/$f" "$DST/$f"
done
cp "$SRC"/code/tasks/*.json "$DST/code/tasks/"
for c in "$SRC"/claims/*.A; do [ -e "$c" ] && cp "$c" "$DST/claims/"; done
for m in "$SRC"/inbox/*_A-to-B_*.md; do [ -e "$m" ] && cp "$m" "$DST/inbox/"; done
# results of tasks claimed by A (no checkpoints, nothing over 50 MB)
for c in "$SRC"/claims/*.A; do
  [ -e "$c" ] || continue
  t="$(basename "$c" .A)"
  if [ -d "$SRC/results/$t" ]; then
    mkdir -p "$DST/results/$t"
    find "$SRC/results/$t" -type f ! -name '*.ckpt.npz' -size -50M -exec cp {} "$DST/results/$t/" \;
  fi
done
# campaign C: A-owned code, docs and manifest (the worker sessions' results/ and status/ live on their own branches)
mkdir -p "$DST/campaign_C/tests"
for f in .gitignore README.md PREREGISTRATION.md WORKER.md START_PROMPTS.md HASHES.json manifest.json make_manifest.py \
         natplat_campaign.py recipe.py worker.py endpoints.py analyze.py tests/test_runner.py; do
  cp "$SRC/campaign_C/$f" "$DST/campaign_C/$f"
done
for f in ANALYSIS.md analysis.json; do
  if [ -e "$SRC/campaign_C/$f" ]; then cp "$SRC/campaign_C/$f" "$DST/campaign_C/$f"; fi
done
# campaign D: A-owned code, docs and manifest (the worker sessions' results/ and status/ live on their own branches)
mkdir -p "$DST/campaign_D/tests"
for f in .gitignore README.md WORKER.md START_PROMPTS.md HASHES.json manifest.json make_manifest.py \
         natplat_campaign.py worker.py endpoints.py analyze.py tests/test_runner.py; do
  cp "$SRC/campaign_D/$f" "$DST/campaign_D/$f"
done
for f in ANALYSIS.md analysis.json; do
  if [ -e "$SRC/campaign_D/$f" ]; then cp "$SRC/campaign_D/$f" "$DST/campaign_D/$f"; fi
done
# campaign E (transformer pilot): A-owned code, docs and manifest (the worker sessions' results/ and status/ live on their own branches)
mkdir -p "$DST/campaign_E/tests"
for f in .gitignore README.md PLAN.md WORKER.md START_PROMPTS.md HASHES.json manifest.json make_manifest.py \
         tfm_runner.py tfm_analyze.py endpoints.py worker.py analyze.py tests/test_tfm.py; do
  cp "$SRC/campaign_E/$f" "$DST/campaign_E/$f"
done
for f in ANALYSIS.md analysis.json FINDINGS.md describe.py; do
  if [ -e "$SRC/campaign_E/$f" ]; then cp "$SRC/campaign_E/$f" "$DST/campaign_E/$f"; fi
done
# campaign F (new seeds at the campaign-D and depth-2 settings): A-owned code, docs and manifest (results live on the session branches)
mkdir -p "$DST/campaign_F/tests"
for f in .gitignore README.md PLAN.md CHECKS.md WORKER.md START_PROMPTS.md HASHES.json manifest.json make_manifest.py \
         natplat_campaign.py worker.py endpoints.py analyze.py tests/test_runner.py; do
  cp "$SRC/campaign_F/$f" "$DST/campaign_F/$f"
done
for f in ANALYSIS.md analysis.json; do
  if [ -e "$SRC/campaign_F/$f" ]; then cp "$SRC/campaign_F/$f" "$DST/campaign_F/$f"; fi
done
# B's log: create only if it does not exist yet (never overwrite B's)
[ -e "$DST/log/B.md" ] || cp "$SRC/log/B.md" "$DST/log/B.md"

git add natural_plateau_shared
if git diff --cached --quiet; then echo "Nothing new to push."; exit 0; fi
git commit -m "A: sync natural_plateau_shared (protocol, tasks, code, campaigns C, D, E and F, A log/claims/results)"
git push origin "$BRANCH"
echo "Pushed to $BRANCH."
