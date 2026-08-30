#!/usr/bin/env bash
# iterate.sh <candidate.py> [cases] [dtype] [incumbent.py]
#
# One evaluation cycle. If incumbent.py is omitted, the current full-sweep
# champion for the dtype is selected from the ledger and timed directly beside
# the challenger.
#
#   sync candidate to the pod -> evaluate -> pull the report -> record in ledger
#
# Exits non-zero if the pod is unreachable so the caller can restart it rather
# than recording a bogus zero score for a candidate that never actually ran.
set -uo pipefail
cd "$(dirname "$0")/.."
CAND="$1"
CASES="${2:-1,2,3,4,5,6,7,8,9,10,11,12,13}"
DTYPE="${3:-float32}"
INCUMBENT="${4:-}"
NAME="$(basename "$CAND" .py)"
REPORT="results/${NAME}.json"

if [ -z "$INCUMBENT" ]; then
  CHAMPION="$(python3 tools/ledger.py champion "$DTYPE" 2>/dev/null || true)"
  if [ -n "$CHAMPION" ]; then
    INCUMBENT="candidates/$CHAMPION"
  fi
fi
if [ -n "$INCUMBENT" ] && [ "$(basename "$INCUMBENT")" = "$(basename "$CAND")" ]; then
  INCUMBENT=""
fi

echo "=== iterate: $NAME (cases=$CASES dtype=$DTYPE incumbent=${INCUMBENT:-none}) ==="

if ! tools/podrun 'echo ALIVE' 2>/dev/null | grep -q ALIVE; then
  echo "POD_UNREACHABLE"; exit 3
fi

SYNC_FILES=("$CAND" tools/harness.py tools/search_stats.py tools/shapes.py candidates/*.py)
if [ -n "$INCUMBENT" ]; then
  SYNC_FILES+=("$INCUMBENT")
fi
tools/podsync /workspace/techjam "${SYNC_FILES[@]}" || exit 4

INCUMBENT_OPT=""
if [ -n "$INCUMBENT" ]; then
  INCUMBENT_OPT="--incumbent $INCUMBENT"
fi

tools/podrun "cd /workspace/techjam && python tools/harness.py \
  --candidate $CAND $INCUMBENT_OPT --cases $CASES --dtype $DTYPE --repeats 30 \
  --out results/${NAME}.json 2>&1 | tail -40"

tools/podget "/workspace/techjam/results/${NAME}.json" "$REPORT" || exit 5
python3 tools/ledger.py add "$REPORT"
