# `test_learn_estimator.py` — Learn & Apply estimator (von Karman slope covariance, tomographic reconstructor)

Source: `tests/test_learn_estimator.py`

Covers `SAOS.tomography.learnEstimator.LearnEstimator`: von Karman structure-function series (`vk_structure_torch`), theoretical slope covariance (`compute_covariance_matrix_torch`), the Phase-1 atmosphere fit (`learn`), the slope-to-gradient gain conversion, and the static and predictive tomographic reconstructor (`build_reconstructor`). `LearnEstimator` only reads a few attributes of its light paths, so the tests use `SimpleNamespace` stand-ins (6x6 WFS of $d=0.5$ m, plate scale 0.5"/px, 640 nm), and no optics are involved. Independent references (no SAOS code): von Karman covariance with scipy Bessel-K in `tests/theory.py`; a slope model as a difference of phase between the midpoints of the subaperture edges; synthetic FFT von Karman phase screens with 3 levels of Lane subharmonics generated in the test. Tiers are set per test (`fast` for analytic and algebraic checks, `nightly` for the statistical screen tests). Tolerances: `tolerances/learn_estimator.yaml` (policy: numerical precision of the independent reference, or estimator statistics of about 3.5 sigma measured over seeds; never absorbing a measured SAOS discrepancy).

Reference model. The von Karman phase covariance and structure function (rad$^2$ at the wavelength of $r_0$; $x=2\pi r/L_0$):

$$
C(r)=C(0)\,\frac{2^{1/6}}{\Gamma(5/6)}\,x^{5/6}K_{5/6}(x),\qquad C(0)=\frac{6\pi}{5}\,0.0229\,\Big(\frac{L_0}{r_0}\Big)^{5/3},\qquad D(r)=2\,[C(0)-C(r)],
$$

with PSD $\Phi(f)=0.0229\,r_0^{-5/3}(f^2+L_0^{-2})^{-11/6}$ (`theory.vk_psd`). The slope is modelled as the phase difference between the edge midpoints $A_i,a_i$ of subaperture $i$, $s_i=[\phi(A_i)-\phi(a_i)]/d_i$, so for layers $l$ with weights $w_l$,

$$
\mathrm{Cov}(s_i,s_j)=\sum_l w_l\,\frac{C(A_i-B_j)-C(A_i-b_j)-C(a_i-B_j)+C(a_i-b_j)}{d_i\,d_j}.
$$

The MMSE (Learn & Apply) reconstructor and the frozen-flow prediction:

$$
R=C_{ts}\,(C_{ss}+C_{nn})^{-1},\qquad \mathrm{Cov}\big(\phi(\mathbf r_a,t+\Delta t),\phi(\mathbf r_b,t)\big)=C(\mathbf r_a-\mathbf r_b-\mathbf V\Delta t).
$$

## `test_vk_series_exact_grid`
- **Module:** `LearnEstimator.vk_structure_torch`, `prepare_vk_constants_torch`.
- **Tier:** fast.
- **What it checks:** the 10-term series for the von Karman structure function against the exact Bessel-K $D(r)=2[C(0)-C(r)]$ on the EST range of separations $r\in[1\,\mathrm{cm},8.5\,\mathrm{m}]$ (60 log-spaced points; 8.5 m = metapupil at 15 km with a 60" field), for $L_0\in\{10,25,50\}$ m, $r_0=0.15$ m. The value at $r=8$ m, $L_0=5$ m, outside the range of use, is recorded for information.
- **Theory / assumptions:** series is exact in the range of use; it is invalid for $L_0\lesssim7$ m with $r\ge6$ m ($x=\pi r/L_0>3$).
- **Tolerance:** `vk_series.use_rtol` = $10^{-4}$ (measured $6.1\times10^{-5}$ at $L_0=10$ m, $r=8.5$ m; $\le10^{-8}$ for $L_0\ge15$ m).
- **References:** von Karman covariance as in Roddier, "Adaptive Optics in Astronomy" (chapter on atmospheric turbulence); not further specified in the code.

## `test_phase1_L0_lower_bound`
- **Module:** `LearnEstimator.learn` (Phase-1 fit), `L0_min`.
- **Tier:** fast.
- **What it checks:** `learn()` bounds $L_0$ to the validity of the series, $L_0\ge\pi r_{max}/3$ ($r_{max}$ the largest separation of the problem, about 3.5-4.3 m for the 6x6, 3 m WFS). Starting from $L_0=1$ m the bound is in $[\pi\cdot3/3,\pi\cdot4.5/3]$ and the fitted $L_0$ is not below it. Uses synthetic slopes from independent screens (100 samples, 5 iterations).
- **Theory / assumptions:** series validity $x=\pi r/L_0\le3$.
- **Tolerance:** bound range and sign hard-coded.
- **References:** none.

## `test_vk_series_convergent_region`
- **Module:** `LearnEstimator.vk_structure_torch`.
- **Tier:** fast.
- **What it checks:** control: where the series converges ($x=\pi r/L_0\le1$) it equals the exact Bessel-K $D(r)$, for $r\in[1\,\mathrm{cm},L_0/\pi]$ and $L_0\in\{5,10,25,50\}$ m; also $D(0)=0$ exactly.
- **Theory / assumptions:** as above.
- **Tolerance:** `vk_series.rtol` = $10^{-6}$ (reference cancellation error $C(0)-C(r)\le10^{-10}$ relative at $r=1$ cm); $D(0)$ exact.
- **References:** none specific.

## `test_css_matches_analytic`
- **Module:** `LearnEstimator.compute_covariance_matrix_torch` (theoretical $C_{ss}$).
- **Tier:** fast.
- **What it checks:** the covariance matrix for 3 WFS (6x6, $d=0.5$ m, directions (0",0), (30",0), (30",90)), 3 layers at 0, 4, 10 km with weights [0.5, 0.3, 0.2], $r_0=0.12$ m, $L_0=25$ m, equals the independent Bessel-K reference built from the WFS geometry; with per-WFS noise variance [0.5, 1, 2] added on the diagonal it equals $C_{ref}+C_{nn}$.
- **Theory / assumptions:** edge-midpoint slope covariance above; layers shifted by $h\,\theta$ at the source direction.
- **Tolerance:** `matrix.rtol` = $10^{-8}$ relative to $\max|C_{ref}|$ (float64, same formula).
- **References:** Vidal, Gendron & Rousset 2010, JOSA A 27:A253 (theoretical $C_{ss}$ of Learn & Apply).

## `test_css_symmetric_positive_semidefinite`
- **Module:** `LearnEstimator.compute_covariance_matrix_torch`.
- **Tier:** fast.
- **What it checks:** the noise-free $C_{ss}$ of the same configuration is symmetric ($|C-C^T|/|C|$) and positive semi-definite (lowest over largest eigenvalue).
- **Theory / assumptions:** a covariance matrix of real slopes.
- **Tolerance:** `matrix.asym_rtol` = $10^{-12}$, `matrix.psd_rtol` = $10^{-9}$ (float64 `eigvalsh`).
- **References:** none.

## `test_css_vs_independent_screens`
- **Module:** `LearnEstimator.compute_covariance_matrix_torch`, compared with Monte Carlo.
- **Tier:** nightly.
- **What it checks:** the theoretical slope covariance against the empirical covariance of slopes from 1200 independent von Karman screens ($N=256$, $h=d/4$, $r_0=0.15$ m, $L_0=10$ m, one 6x6 WFS, one layer, FFT with Lane subharmonics generated in the test). The statistic $Q=\sum(\hat C-C)^2/\sum\mathrm{Var}$ is compared with its expectation 1, and $\mathrm{tr}\,\hat C/\mathrm{tr}\,C$ with 1.
- **Theory / assumptions:** for $n$ independent zero-mean Gaussian realisations the sample covariance has $\mathrm{Var}(\hat C_{ij})=(C_{ii}C_{jj}+C_{ij}^2)/n$, hence $E[Q]=1$. The reference has a bias of at most 1.5 % due to FFT grid truncation above Nyquist. Subaperture-averaged slopes would differ from the midpoint model and are not part of the criteria.
- **Tolerance:** `screens`: `q_lo` = 0.67, `q_hi` = 1.38 (mean 1.02, sigma 0.10 over 9 seeds, band mean $\pm3.5\sigma$); `trace_atol` = 0.035 (mean 0.985, sigma 0.55 %, 1.5 % bias plus $3.5\sigma$).
- **References:** Lane subharmonics screen generation as commonly used (not cited in the code).

## `test_gain_slopes_to_phase_gradient`
- **Module:** `LearnEstimator.learn` (gain passed to `compute_measurement_covariance`).
- **Tier:** fast.
- **What it checks:** the factor that `learn()` passes to `compute_measurement_covariance` (captured by a spy on random slopes, 1 iteration) equals $2\pi\,\mathrm{ps}/(206265\times500\,\mathrm{nm})$, independent of the WFS wavelength (640 nm here) and the subaperture size.
- **Theory / assumptions:** a slope of $s$ px moves the spot by $s\,\mathrm{ps}/206265$ rad, i.e. an OPD gradient (achromatic); the von Karman model is the phase structure function of $r_0$ at 500 nm (Atmosphere convention), so the conversion to a phase gradient in rad/m uses $2\pi/500$ nm.
- **Tolerance:** `gain.rtol` = $10^{-9}$.
- **References:** none specific.

## `test_phase1_recovers_r0`
- **Module:** `LearnEstimator.learn` (Phase 1: gradient-based fit of $r_0$, $L_0$, noise variance).
- **Tier:** nightly.
- **What it checks:** slopes from independent von Karman screens ($r_0=0.15$ m, $L_0=10$ m, one layer, 1000 samples, noise-free) with the fit started at $1.3\,r_0$; the gain is replaced by the correct one so the test is independent of the gain test. $|r_{0,fit}/r_0-1|$ must be within the tolerance.
- **Theory / assumptions:** maximum-likelihood style fit of the theoretical $C_{ss}$ to the measured slope covariance; L0 and noise couple to $r_0$ in the joint fit (+0.9 % mean bias).
- **Tolerance:** `phase1.r0_rtol` = 0.03 (mean +0.9 %, sigma 0.5 % over 9 seeds; mean plus $3.5\sigma$).
- **References:** Vidal, Gendron & Rousset 2010 (Learn & Apply); Martin et al. 2016 (on-sky L&A) is not cited in the test.

## `test_phase1_recovers_r0_end_to_end`
- **Module:** `LearnEstimator.learn`.
- **Tier:** nightly.
- **What it checks:** the same as `test_phase1_recovers_r0` with the unmodified gain of `learn()`: $r_0$ must be recovered from slopes in pixels.
- **Theory / assumptions:** as above.
- **Tolerance:** `phase1.r0_rtol` = 0.03.
- **References:** as above.

## `test_reconstructor_static`
- **Module:** `LearnEstimator.build_reconstructor` (static, $\Delta t=0$).
- **Tier:** fast.
- **What it checks:** parametrised over `permute` = False, True. The non-predictive reconstructor $R=C_{ts}(C_{ss}+C_{nn})^{-1}$ (3 WFS sensed, 2 target directions, 3 layers, noise [0.3, 0.6, 0.9], regularisation $10^{-9}$) equals the independent Bessel-K reference, both in the natural order [X of all WFS, Y of all WFS] and with `permute_for_controller` ([WFS0_X, WFS0_Y, WFS1_X, ...]).
- **Theory / assumptions:** MMSE (Bayesian) reconstructor, $R=C_{ts}(C_{ss}+C_{nn})^{-1}$.
- **Tolerance:** `reconstructor.rtol` = $10^{-7}$ (float64, same regularisation; Cholesky solve conditioning).
- **References:** Vidal, Gendron & Rousset 2010 (Learn & Apply).

## `test_reconstructor_predicts_future`
- **Module:** `LearnEstimator.build_reconstructor` (predictive with winds and $\Delta t$).
- **Tier:** fast.
- **What it checks:** with frozen flow, winds (10, 3), (-6, 8), (0, 0) m/s and $\Delta t=20$ ms, the covariance between the target at $t+\Delta t$ and the sensed slopes at $t$ is $C(\mathbf r_t-\mathbf r_s-\mathbf V\Delta t)$. Two criteria: the exact error variance $\mathrm{tr}\,E[(Rs-t)(Rs-t)^T]$ of $R_{saos}$ equals that of the optimal reference built with $-V\Delta t$; and $R_{saos}$ itself matches that reference. The retrodiction reconstructor ($+V\Delta t$) error is recorded for information.
- **Theory / assumptions:** Taylor frozen flow, $\phi(\mathbf r,t)=\phi_0(\mathbf r-\mathbf V t)$; error trace $=\mathrm{tr}\,[C_{tt}-RC_{ts}^T-C_{ts}R^T+R\,C_{ss}R^T]$ with exact covariances.
- **Tolerance:** `reconstructor.err_ratio_atol` = $10^{-6}$ on $\mathrm{tr}E[R_{saos}]/\mathrm{tr}E[R_{opt}]-1$; `reconstructor.rtol` = $10^{-7}$.
- **References:** Vidal, Gendron & Rousset 2010; predictive extension as described by Martin et al. 2016 is not cited in the code.

## `test_reconstructor_zero_wind_ignores_dt`
- **Module:** `LearnEstimator.build_reconstructor`.
- **Tier:** fast.
- **What it checks:** control: with zero winds the shift is zero, so any $\Delta t$ gives the static reconstructor (also equal to the reference).
- **Theory / assumptions:** shift $=V\Delta t=0$.
- **Tolerance:** `reconstructor.rtol` = $10^{-7}$.
- **References:** none.

## `test_default_dt_is_delay_plus_one`
- **Module:** `LearnEstimator.build_reconstructor` (`dt=None` default).
- **Tier:** fast.
- **What it checks:** a light path with delay $d$ frames has a total pure delay $(d+1)T_s$ (SAOS convention, `tests/test_timing.py`), so with `dt=None` the reconstructor predicts $(d+1)T_s$ ahead and equals the one with explicit $\Delta t=(d+1)T_s$ ($d=2$, $T_s=5$ ms). The result with $dT_s$ is recorded to show the test discriminates.
- **Theory / assumptions:** timing convention only.
- **Tolerance:** `reconstructor.dt_rtol` = $10^{-9}$ (same code path).
- **References:** none.

## `test_frozen_flow_cross_covariance_sign`
- **Module:** none from SAOS (reference check of the sign used by the predictive tests; uses only `theory.py` and the test's screens).
- **Tier:** nightly.
- **What it checks:** with a layer moving at $V=+x$, $V\Delta t=1$ m (8 px of $h=0.125$ m), $\phi(x,t+\Delta t)=\phi_0(x-V\Delta t)$: the empirical covariance between the slopes of the same 6x6 WFS at $t+\Delta t$ (rows) and at $t$ (columns), from 1500 independent screens, agrees with the theory with shift $-V\Delta t$ and not with $+V\Delta t$ (relative Frobenius norm).
- **Theory / assumptions:** frozen flow, $\mathrm{Cov}(\phi(\mathbf r_a,t+\Delta t),\phi(\mathbf r_b,t))=C(\mathbf r_a-\mathbf r_b-\mathbf V\Delta t)$.
- **Tolerance:** `frozen_flow.err_hi` = 0.18 for the right sign (mean 0.160, sigma 0.006 over 9 seeds); `frozen_flow.wrong_sign_lo` = 1.0 for the wrong sign (measured 1.40, more than 6 times the right-sign error).
- **References:** none specific.
