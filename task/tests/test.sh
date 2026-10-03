#!/usr/bin/env bash
# Verifier entry point. Runs the sealed submission checks (test_submission.py
# only; unit tests live in authoring/tests and are not shipped) and writes
# exactly 0 or 1 to $LOG_DIR/reward.txt, plus a CTRF report to $LOG_DIR/ctrf.json.
#
# Fail closed: reward is 0 unless pytest exits 0 AND the CTRF summary shows at
# least one test, every test passed, and none were skipped/pending/other.
#
# Overridable for authoring runs (defaults are the verifier container's):
#   LOG_DIR          /logs/verifier
#   SUBMISSION_PATH  /app/output/submission.json  (read by test_submission.py)
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/logs/verifier}"
REWARD="$LOG_DIR/reward.txt"
CTRF="$LOG_DIR/ctrf.json"

mkdir -p "$LOG_DIR"
printf 0 > "$REWARD"
rm -f "$CTRF"

cd "$TESTS_DIR"
export PYTHONDONTWRITEBYTECODE=1
python3 -m pytest test_submission.py -p no:cacheprovider -rA --ctrf "$CTRF"
status=$?
echo "pytest exit status: $status"

if [ "$status" -eq 0 ] && python3 - "$CTRF" <<'EOF'
import json, sys
summary = json.load(open(sys.argv[1], encoding='utf-8'))['results']['summary']
print('CTRF summary:', {k: summary.get(k) for k in ('tests', 'passed', 'failed', 'skipped', 'pending', 'other')})
clean = all(summary.get(k, 0) == 0 for k in ('failed', 'skipped', 'pending', 'other'))
sys.exit(0 if summary.get('tests', 0) > 0 and summary.get('passed') == summary.get('tests') and clean else 1)
EOF
then
    printf 1 > "$REWARD"
fi

echo "reward: $(cat "$REWARD")"
exit 0
