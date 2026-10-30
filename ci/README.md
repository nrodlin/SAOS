# SAOS validation suite

The tests in `tests/` compare each SAOS module with theory: phase-screen statistics, WFS slopes, PSFs, DM shapes,
control laws and so on. Each test isolates one module and feeds it synthetic input. Results and figures are written
outside the repository and gathered into an HTML report.

## Tiers

| Tier | Marker | Content | Where | Duration |
|---|---|---|---|---|
| T1 | `fast` | Module physics at small scale, deterministic seeds | CPU | < 5 min |
| T2 | `nightly` | More realisations, extended object, CPU vs GPU, short closed loops | Workstation GPU | < 45 min |
| T3 | `heavy` | Reduced MCAO scenarios (D 0.64 m, 8x8 SH, 3 DMs), scale-dependent code paths | Laptop / workstation | < 1 h |

Status: T1 and T2 are implemented; T3 is planned (no test carries the `heavy` marker yet). Every test is documented in the
Validation section of the docs (`docs/testing/`).

Every test compares SAOS with an independent physical or mathematical expectation. Tolerances only cover the statistical
uncertainty of the estimator (about 3.5 sigma, measured over seeds) or the numerical precision of the reference; they
never absorb a measured SAOS discrepancy. A failing test is reported with its measured value and tolerance, and the report groups the failures.

## Running

```bash
ci/run_ci.sh fast                 # T1 (forces CPU)
ci/run_ci.sh nightly              # T1 + T2
ci/run_ci.sh heavy                # T3
ci/run_ci.sh fast -k wfs -x       # extra arguments go to pytest
python -m pytest tests -m fast    # plain pytest; results to ~/simulations/results/ci/dev
```

Results go to `~/simulations/results/ci/<date>_<tier>_<sha>/` (override with `SAOS_CI_ROOT`):

- `report.html`, `results.json`, `junit.xml`, `pytest.log`;
- one folder per test with its figures and `metrics.json`.

`latest_<tier>` points to the last run. `history.csv` collects the trend metrics.

The suite needs `pytest`, `pytest-timeout` and `pyyaml` in the SAOS environment. Choose the interpreter with
`SAOS_CI_PYTHON`.

## Running automatically (local resources only)

**Git hook (T1 before pushing to develop/master):**

```bash
git config core.hooksPath ci/githooks       # SAOS_SKIP_CI=1 git push to skip once
```

**Nightly timer (T2 on a clean worktree of origin/develop, only when it changed):**

```bash
mkdir -p ~/.config/systemd/user
cp ci/systemd/saos-ci-nightly.* ~/.config/systemd/user/     # check the paths inside the .service
systemctl --user daemon-reload && systemctl --user enable --now saos-ci-nightly.timer
loginctl enable-linger $USER                                # run even when not logged in
```

**Self-hosted GitHub runner:** a ✓/✗ per commit on GitHub, computed on your machine
(`.github/workflows/validation.yml`). Nothing runs periodically:

- every push to develop/master runs T1;
- every pull request to master runs T1 + T2 (release gate);
- any tier can be run on demand: Actions → validation → Run workflow, choosing `fast`, `nightly` or `heavy`.

Jobs triggered while the machine is off wait in the queue and start when the runner comes back (GitHub cancels them
after 24 h).

1. Register a runner with labels `saos,gpu` (Settings → Actions → Runners).
2. Release gate: in Settings → Branches (or Rules → Rulesets) protect `master` with "Require a pull request before
   merging" and "Require status checks to pass", selecting the `release-gate` check (T1 + T2 on the runner).
   Merge develop into master through a pull request.

The repository is public: the `release-gate` job only runs for pull requests opened from a branch of this repository
(`head.repo == repository`), so a pull request from a fork never reaches the runner. Never use `pull_request_target`.

## Writing a test

- Add a `pytestmark` with the tier at the top of the module.
- Pass the `logger` fixture to every SAOS object. With `logger=None`, SAOS starts a logging thread that can block
  the interpreter at exit.
- The docstring states the theory and the criterion.
- Record every quantity with `report.check(name, value, expected, rtol/atol, lo/hi)`. Add figures with
  `report.figure(fig, name)`, and end with `report.assert_ok()`.
- Tolerances live in `tests/tolerances/<area>.yaml`. Statistical tolerances are about 3.5σ of the estimator, with σ
  measured over several seeds.
- Generic references go in `tests/theory.py`, which never imports SAOS. Shared numerical helpers go in
  `tests/citools.py`.
