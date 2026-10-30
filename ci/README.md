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

Every test compares SAOS with an independent physical or mathematical reference. Tolerances cover the statistical
uncertainty of the estimator (about 3.5 sigma, measured over seeds) or the numerical precision of the reference. The
report lists every check with its measured value and tolerance.

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

## Running automatically

**Git hook (T1 before pushing to develop/master):**

```bash
git config core.hooksPath ci/githooks       # SAOS_SKIP_CI=1 git push to skip once
```

**Nightly timer (T2 on a clean worktree of origin/develop, only when it changed):**

```bash
mkdir -p ~/.config/systemd/user
cp ci/systemd/saos-ci-nightly.* ~/.config/systemd/user/     # paths are set in the .service
systemctl --user daemon-reload && systemctl --user enable --now saos-ci-nightly.timer
loginctl enable-linger $USER                                # runs without an open session
```

**GitHub Actions (self-hosted runner):** `.github/workflows/validation.yml` runs the suite on a runner registered with
the labels `saos,gpu` (Settings → Actions → Runners):

| Event | Tiers |
|---|---|
| Push to develop or master | T1 |
| Pull request to master (`release-gate` job) | T1 + T2 |
| Manual run (Actions → validation → Run workflow) | `fast`, `nightly` or `heavy` |

Jobs triggered while the runner is offline stay queued until it connects (GitHub cancels them after 24 h). The
`release-gate` job only runs for pull requests opened from a branch of this repository. Protecting master with a
required `release-gate` status check (Settings → Rules → Rulesets) makes every merge from develop pass T1 + T2.

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
