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

Status: T1 is implemented; T2 and T3 are planned (no tests carry those markers yet, so those tiers select nothing).

Every test compares SAOS with an independent physical or mathematical expectation. Tolerances only cover the statistical
uncertainty of the estimator (about 3.5 sigma, measured over seeds) or the numerical precision of the reference; they
never absorb a measured SAOS discrepancy. Known SAOS bugs are labelled with `citools.known_issue(ID, description,
location)`: those tests fail until the bug is fixed (no xfail), and the report groups the failures by issue.

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

**Self-hosted GitHub runner:** a ✓/✗ per commit on GitHub, computed on your machine.

1. Register a runner with labels `saos,gpu` (Settings → Actions → Runners), under a dedicated user without sudo.
2. Copy `ci/github/validation.yml` to `.github/workflows/`.

The repository is public, so the workflow never triggers on pull requests.

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
