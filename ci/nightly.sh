#!/usr/bin/env bash
# Nightly validation on a clean worktree of origin/develop (run by ci/systemd/saos-ci-nightly.timer).
# Runs only when origin/develop changed since the last nightly run; notifies on failure.
#   SAOS_CI_REPO      clone used to fetch (default: the repository holding this script)
#   SAOS_CI_WORKTREE  clean worktree (default ~/simulations/ci_worktree)
#   SAOS_CI_PYTHON    interpreter with SAOS dependencies and pytest (default: python)
#   SAOS_CI_FORCE=1   run even if the commit did not change
set -uo pipefail
REPO=${SAOS_CI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WT=${SAOS_CI_WORKTREE:-$HOME/simulations/ci_worktree}
ROOT=${SAOS_CI_ROOT:-$HOME/simulations/results/ci}
STAMP="$ROOT/.nightly_last_sha"
mkdir -p "$ROOT"

git -C "$REPO" fetch -q origin develop || { echo "fetch failed" >&2; exit 1; }
SHA=$(git -C "$REPO" rev-parse origin/develop)
if [ "${SAOS_CI_FORCE:-0}" != "1" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$SHA" ]; then
    echo "origin/develop unchanged ($SHA), nothing to do"; exit 0
fi
if [ ! -d "$WT" ]; then
    git -C "$REPO" worktree add --detach "$WT" "$SHA" || exit 1
else
    git -C "$WT" checkout -q --force --detach "$SHA" && git -C "$WT" clean -qfdx || { echo "checkout of $SHA failed" >&2; exit 1; }
fi
# The worktree must be at the fetched commit; otherwise nothing is tested or stamped
if [ "$(git -C "$WT" rev-parse HEAD)" != "$SHA" ]; then
    echo "worktree is not at $SHA, aborting" >&2; exit 1
fi

# The worktree's SAOS must be the one imported
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
"$WT/ci/run_ci.sh" nightly -q
STATUS=$?
echo "$SHA" > "$STAMP"
if [ "$STATUS" -ne 0 ]; then
    MSG="SAOS nightly validation FAILED for ${SHA:0:7}: $ROOT/latest_nightly/report.html"
    command -v notify-send >/dev/null && notify-send -u critical "SAOS CI" "$MSG"
    echo "$MSG" >&2
fi
exit "$STATUS"
