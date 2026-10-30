#!/usr/bin/env bash
# Run one tier of the SAOS validation suite and build its report.
#   ci/run_ci.sh [fast|nightly|heavy] [extra pytest arguments]
# Results: $SAOS_CI_ROOT (default ~/simulations/results/ci)/<date>_<tier>_<sha>/ with report.html, results.json,
# junit.xml and one folder per test; latest_<tier> points to the last run; history.csv collects the trend metrics.
# Environment: SAOS_CI_PYTHON (interpreter, default: python in PATH), SAOS_CI_ROOT, SAOS_CI_THREADS (default 4).
set -uo pipefail

TIER=${1:-fast}
shift || true
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${SAOS_CI_PYTHON:-python}
ROOT=${SAOS_CI_ROOT:-$HOME/simulations/results/ci}

case "$TIER" in
    fast)    MARK="fast";            BUDGET=900;   export CUDA_VISIBLE_DEVICES="" ;;   # T1 always on CPU
    nightly) MARK="fast or nightly"; BUDGET=5400 ;;
    heavy)   MARK="heavy";           BUDGET=7200 ;;
    *) echo "unknown tier '$TIER' (fast|nightly|heavy)" >&2; exit 2 ;;
esac

SHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)
[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && SHA="${SHA}-dirty"
OUT="$ROOT/$(date +%Y%m%d_%H%M%S)_${TIER}_${SHA}"
mkdir -p "$OUT"

export SAOS_CI_OUT="$OUT" SAOS_CI_TIER="$TIER"
export OMP_NUM_THREADS=${SAOS_CI_THREADS:-4} MKL_NUM_THREADS=${SAOS_CI_THREADS:-4} OPENBLAS_NUM_THREADS=${SAOS_CI_THREADS:-4}
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export PYTHON_GIL=${PYTHON_GIL:-0}                  # free-threaded run as documented (OpenCV does not declare free-threading support)

echo "SAOS validation, tier $TIER ($MARK), commit $SHA -> $OUT"
cd "$REPO"
# `timeout` is a last resort against hangs at interpreter exit (pytest-timeout covers each test).
timeout --kill-after=60 "$BUDGET" "$PY" -m pytest tests -m "$MARK" --junitxml="$OUT/junit.xml" "$@" 2>&1 | tee "$OUT/pytest.log"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" -eq 5 ] && echo "No tests selected for tier $TIER (marker '$MARK')" >&2

"$PY" "$REPO/ci/make_report.py" "$OUT" --history "$ROOT/history.csv" || echo "report generation failed" >&2
ln -sfn "$OUT" "$ROOT/latest_$TIER"
echo "Report: $OUT/report.html (pytest exit status $STATUS)"
exit "$STATUS"
