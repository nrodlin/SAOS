# Validation suite

SAOS is validated by the test suite in `tests/`. Each test isolates one module, feeds it synthetic input and compares the
result with an independent physical or mathematical reference (closed-form theory written in the tests, never SAOS code).
This section documents every test: what it checks, the module it covers, the theory and assumptions behind the reference,
the tolerance and the references.

## Tiers

| Tier | Marker | Content | Where | Duration |
|---|---|---|---|---|
| T1 | `fast` | Module physics at small scale, deterministic seeds | CPU | < 5 min |
| T2 | `nightly` | More realisations, extended object, CPU vs GPU, packaging | Workstation (GPU optional) | < 45 min |
| T3 | `heavy` | Reduced MCAO scenarios (planned, no tests yet) | Workstation | < 1 h |

`gpu` marks tests that need CUDA (skipped when no GPU is visible).

## Running

```bash
ci/run_ci.sh fast                 # T1 (forces CPU)
ci/run_ci.sh nightly              # T1 + T2
ci/run_ci.sh fast -k wfs -x       # extra arguments go to pytest
python -m pytest tests -m fast    # plain pytest
```

Results, figures and an HTML report go to `~/simulations/results/ci/<date>_<tier>_<sha>/` (override with `SAOS_CI_ROOT`);
see `ci/README.md` for the nightly timer, the git hooks and the GitHub workflow.

## Tolerance policy

Tolerances (`tests/tolerances/*.yaml`) only cover the statistical uncertainty of the estimator (about 3.5 sigma, measured
over several seed sets) or the numerical precision of the reference. They never absorb a measured SAOS discrepancy: a
systematic offset is either corrected or an explicit, documented convention.

## Conventions and common theory

Conventions used by every test (from `tests/citools.py`): `report.check(name, value, expected, rtol, atol, lo, hi)` passes when $|v-e| \le \mathrm{atol} + \mathrm{rtol}\,|e|$ (expected given) or $\mathrm{lo}\le v\le \mathrm{hi}$ (bounds given), element-wise for arrays; `report.assert_ok()` fails the test with all out-of-tolerance metrics after every metric has been recorded. `report.metric` records a value with no pass/fail criterion. Statistical tolerances in the YAML files are, unless stated otherwise, $3.5\sigma$ of the estimator measured over several seeds. Phase screens are in rad at 500 nm; OPD in m, and $\varphi = 2\pi\,\mathrm{OPD}/500\,\mathrm{nm}$.

Analytic references (`tests/theory.py`, independent of SAOS) used by the atmosphere tests:

$$\Phi(f) = 0.0229\, r_0^{-5/3}\left(f^2 + L_0^{-2}\right)^{-11/6}\ \mathrm{[rad^2\,m^2]},\qquad
C(0) = 0.0229\,\pi\,\tfrac{6}{5}\,r_0^{-5/3} L_0^{5/3}\approx 0.0863\,(L_0/r_0)^{5/3}$$

$$C(r) = C(0)\,\frac{2^{1/6}}{\Gamma(5/6)}\,x^{5/6}K_{5/6}(x),\quad x = \frac{2\pi r}{L_0},\qquad D(r) = 2\,[C(0)-C(r)] \xrightarrow{\,r\ll L_0\,} 6.88\,(r/r_0)^{5/3}$$

## Test files

219 test entries (a parametrised test counts as one entry).

| File | Content | Entries | Tier |
|---|---|---|---|
| [`test_atmosphere.py`](test_atmosphere.md) | von Kármán screens, extrusion, layer geometry, footprints and wind (T1) | 22 | fast |
| [`test_atmosphere_t2.py`](test_atmosphere_t2.md) | atmosphere at EST scale: r0 scan, Zernike variances, frozen flow, anisoplanatism, seeing (T2) | 7 | nightly |
| [`test_perturbations.py`](test_perturbations.md) | Vibration, LocalSeeing, spider arms, NCPA, logging at exit (T2) | 16 | nightly |
| [`test_timing.py`](test_timing.md) | loop latency: impulse response of the AO chain (T1) | 1 | fast |
| [`test_detector.py`](test_detector.md) | Detector noise statistics, quantisation, saturation, reproducibility (T1) | 11 | fast |
| [`test_modal_basis.py`](test_modal_basis.md) | modal bases for DM commands and interaction matrices (T1) | 12 | fast |
| [`test_dm.py`](test_dm.md) | DeformableMirror: influence functions, fitting, altitude footprints, geometry, dynamics, misregistration (T1) | 15 | fast |
| [`test_packaging.py`](test_packaging.md) | wheel contents and declared dependencies (C.13, nightly) | 2 | nightly |
| [`test_wfs.py`](test_wfs.md) | Shack-Hartmann (point source) validation | 18 | fast |
| [`test_corrsh.py`](test_corrsh.md) | Correlating Shack-Hartmann (solar, extended scene) | 13 | fast |
| [`test_extended_source.py`](test_extended_source.md) | Extended (solar) source with ScienceCam and Atmosphere | 15 | nightly |
| [`test_sciencecam.py`](test_sciencecam.md) | Science camera PSF and long-exposure accumulation | 10 | fast |
| [`test_im_controller.py`](test_im_controller.md) | Interaction matrix and control laws | 24 | fast |
| [`test_learn_estimator.py`](test_learn_estimator.md) | Learn & Apply estimator (von Karman slope covariance, tomographic reconstructor) | 14 | fast (10) / nightly (4) |
| [`test_savepoint.py`](test_savepoint.md) | Saved metrics and HDF5 output | 13 | fast |
| [`test_sharepoint.py`](test_sharepoint.md) | ZMQ publisher and request client | 17 | fast |
| [`test_parallel_gpu.py`](test_parallel_gpu.md) | Parallel (threads) and GPU execution | 9 | nightly |
