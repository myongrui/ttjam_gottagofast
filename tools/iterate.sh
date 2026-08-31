#!/usr/bin/env bash
# iterate.sh <candidate.py> [cases] [dtype] [incumbent.py] [report.json]
#
# One evaluation cycle. If incumbent.py is omitted, the current full-sweep
# champion for the contract hardware and dtype is selected from canonical events
# the challenger.
#
#   sync candidate to the pod -> evaluate -> pull immutable report -> append events
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
REPORT="${5:-results/evaluations/${NAME}__$(date -u +%Y%m%dT%H%M%SZ)__$$.json}"

if [ -e "$REPORT" ]; then
  echo "REFUSING_TO_OVERWRITE_IMMUTABLE_REPORT: $REPORT"; exit 6
fi
mkdir -p "$(dirname "$REPORT")"

if [ -z "$INCUMBENT" ]; then
  CHAMPION="$(python3 tools/event_store.py champion --dtype "$DTYPE" 2>/dev/null || true)"
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
tools/podrun "mkdir -p /workspace/techjam/$(dirname "$REPORT")" || exit 4

INCUMBENT_OPT=""
if [ -n "$INCUMBENT" ]; then
  INCUMBENT_OPT="--incumbent $INCUMBENT"
fi

tools/podrun "cd /workspace/techjam && python tools/harness.py \
  --candidate $CAND $INCUMBENT_OPT --cases $CASES --dtype $DTYPE --repeats 30 \
  --out $REPORT 2>&1 | tail -40"

tools/podget "/workspace/techjam/$REPORT" "$REPORT" || exit 5
python3 tools/event_store.py add "$REPORT"
