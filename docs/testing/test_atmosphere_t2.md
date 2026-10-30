# `test_atmosphere_t2.py` — atmosphere at EST scale: r0 scan, Zernike variances, frozen flow, anisoplanatism, seeing (T2)

Source: `tests/test_atmosphere_t2.py`

Covers `PhaseScreenVonKarman` (initial and extruded screens), `Atmosphere` (multi-layer OPD, footprints), `Telescope`, `Source` and `ScienceCam` at the EST geometry (D = 4.149 m, 216 px, 1.92 cm/px, $L_0$ = 25 m, $T_s$ = 1 ms). All references are written inside the test module (von Kármán formulas, Noll Zernike basis orthonormalised on the sampled pupil, exact pupil covariance, finite-aperture long-exposure OTF), independent of SAOS. Statistics are larger than in T1 and several tests use a thread pool. The whole module is `nightly`. Tolerances come from `tests/tolerances/atmosphere_t2.yaml` (calibrated 2026-10-27 over 6 seed sets, each $3.5\times$ the per-run sd, scaled by $1/\sqrt n$ when $n$ differs from the calibration run; none is derived from a SAOS-vs-theory offset).

## `test_r0_scan_pupil_scale`
- **Module:** `PhaseScreenVonKarman.initial_screen` and `add_row`.
- **Tier:** nightly. Parametrised over `source` = `initial`, `extruded`.
- **What it checks:** for $r_0$ = 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0 m, the mean structure function of 16 screens of 216 px (initial FFT screens, or one extruded screen sampled every $2n_x$ rows after a burn-in of $n_x$ rows; the YAML gives `n_maps` = 16, the docstring still says 32) is fitted (lags 1 px to $n_x/2$) for $r_0$ jointly with $L_0$ and with $L_0$ fixed. Fitted/true $r_0$ must be 1 and the joint $L_0$ ratio must lie in $[e^{-t}, e^{t}]$.
- **Theory / assumptions:** $D_{vK}(r; r_0, L_0)$ above; a pupil-sized screen ($\approx 0.16\,L_0$) only weakly constrains $L_0$, hence a tolerance on $\log L_0$.
- **Tolerance:** `atmosphere_t2.r0_scan.{initial,extruded}`: `r0_joint`, `r0_fixed` (per $r_0$; initial 0.032 / 0.116, extruded 0.077 / 0.123) and `logL0` (0.95 initial, 1.6 extruded), all $3.5\sigma$ over seeds.
- **References:** Assémat, Wilson & Gendron 2006; Roddier 1981.

## `test_outer_scale_recovery`
- **Module:** `PhaseScreenVonKarman.initial_screen`.
- **Tier:** nightly. Parametrised over `L0_true` = 5, 12, 25 m.
- **What it checks:** screens of 38.4 m (256 px of 15 cm, $r_0$ = 0.3 m) that span the turnover; joint $r_0$/$L_0$ fit of the mean $D(r)$ of 24 initial screens (lags 0.45 m to 19 m) recovers $L_0$ (within $e^{\pm t}$) and $r_0$.
- **Theory / assumptions:** as for the T1 outer-scale tests.
- **Tolerance:** `atmosphere_t2.outer_scale.L0_{5,12,25}`: `r0` 0.011 / 0.028 / 0.032 and `logL0` 0.021 / 0.084 / 0.112 ($3.5\sigma$ over seeds).
- **References:** Assémat, Wilson & Gendron 2006.

## `test_zernike_variances_initial_screen`
- **Module:** `PhaseScreenVonKarman.initial_screen`.
- **Tier:** nightly.
- **What it checks:** the variances of the Z2 to Z11 coefficients of 1000 initial 216 px screens ($r_0$ = 0.10 m, $L_0$ = 25 m; the docstring still says 600) on the circular pupil against the exact von Kármán projection computed on the sampled pupil; the mean of the ratios over Z2 to Z11; and the largest off-diagonal element of the whitened covariance (Cholesky of the theory), which must be near zero.
- **Theory / assumptions:** the coefficients are $a_j = \frac{1}{N}\sum_{\mathrm{pupil}}\varphi Z_j$ with Noll-normalised Zernikes orthonormalised on the sampled pupil; their covariance is $G = B^{T}CB$, the sampled covariance $C$ projected on the basis (computed by FFT convolution of the sampled $C(r)$ with each basis vector). Gaussian variance estimator: relative sd $\sqrt{2/n}$.
- **Tolerance:** `atmosphere_t2.zernike.initial`: `mode` 0.27, `mean` 0.085, `offdiag` 0.18, with $n$ = 1000 ($3.5\,\mathrm{sd}_{200}/\sqrt5$; off-diagonal limit adds $3.5\times0.06/\sqrt5$ to the expected $0.20/\sqrt5$).
- **References:** Noll 1976, JOSA 66:207.

## `test_zernike_variances_atmosphere`
- **Module:** `Atmosphere` (3 layers at 0/5/12 km, 60/30/10 % Cn$^2$, total $r_0$ = 0.12 m, winds 0/90/0 deg), `Atmosphere.updateLayer`, `getOPD`.
- **Tier:** nightly.
- **What it checks:** the same Z2 to Z11 variance ratios and whitened covariance as above for the on-axis OPD, after burn-in, with 1000 samples from 4 independent atmospheres spaced 1000 rows (19 m, decorrelated beyond $L_0$).
- **Theory / assumptions:** a sum of independent layers with $\sum r_{0,i}^{-5/3} = r_0^{-5/3}$ and a common $L_0$ has the von Kármán covariance of $r_0$. The measured mean ratio is about 0.96 (Z4 about 0.78, Z10 about 0.88), so the tolerances are set from the observed spread.
- **Tolerance:** `atmosphere_t2.zernike.atmosphere`: `mode` 0.16, `mean` 0.033, `offdiag` 0.14, $n$ = 1000.
- **References:** Noll 1976, JOSA 66:207.

## `test_taylor_frozen_flow`
- **Module:** `Atmosphere.update`, `Atmosphere.getOPD` (single layer).
- **Tier:** nightly. Parametrised over (k px/step, direction deg) = (1, 0), (1, 90), (2, 0).
- **What it checks:** the temporal structure function of the on-axis OPD of one layer, averaged over the 216 x 216 px and 2000 steps after a burn-in of $2n_x$ steps, at delays $\tau$ = 1, 2, 3, 5, 8, 13, 21, 34, 54, 80, 108 steps (limited to $k\tau\le108$ px), against the spatial $D_{vK}$ at $r = k\tau\,\mathrm{px}$. The effective speed ratio (mean ratio on $r\le n_x/4$ to the power 0.6) is recorded.
- **Theory / assumptions:** Taylor frozen flow,

$$D_\varphi(\tau) = \langle[\varphi(t+\tau)-\varphi(t)]^2\rangle = D_{vK}(v\tau T_s),\qquad D\propto v^{5/3}\ (r\ll L_0),$$

  independent of the direction of motion; integer pixels per step.
- **Tolerance:** `atmosphere_t2.frozen_flow`: per-delay absolute tolerance on the ratio, `small_r` = [0.12, 0.16, 0.19, 0.23, 0.29, 0.34, 0.40, 0.48, 0.61] and `large_r` = [0.80, 1.01]; $3.5\times$ the largest sd of the (1,0) and (1,90) cases, indexed by delay and therefore conservative for $k>1$. The tolerances are wide at large $\tau$ because few independent samples cover the screen.
- **References:** Taylor frozen-flow hypothesis (Conan 1995 discusses temporal statistics; the test uses the structure function, not spectra).

## `test_anisoplanatism`
- **Module:** `Atmosphere.getOPD` for off-axis `Source`, footprint model.
- **Tier:** nightly.
- **What it checks:** the mean over the pupil and over 96 samples of $(\mathrm{OPD}(\theta)-\mathrm{OPD}(0))^2$ for $\theta$ = 5, 10, 15, 20 arcsec (4 layers, $r_0$ = 0.12 m, fractions 0.5/0.2/0.2/0.1, altitudes 0, about 1.6, 4, 8 km tuned to integer-pixel offsets), both pointwise and after removing the pupil mean (piston), against theory.
- **Theory / assumptions:** independent layers, each with $r_{0,i} = r_0 f_i^{-3/5}$ and shift $s_i = h_i\tan\theta$:

$$\langle\Delta\varphi^2\rangle_{pw} = \sum_i D_i(s_i) = \sum_i 2\,[C_i(0)-C_i(s_i)],$$

  and with piston removed the variance is reduced by $\frac{1}{N^2}\sum_{\mathbf{p},\mathbf{q}}\sum_i\left[2C_i(\mathbf{d})-C_i(\mathbf{d}+\mathbf{s}_i)-C_i(\mathbf{d}-\mathbf{s}_i)\right]$ ($\mathbf d=\mathbf p-\mathbf q$), computed with the autocorrelation of the sampled circular pupil. Altitudes are chosen so the footprint offsets are $n+0.02$ px, to decouple rounding of the integer-pixel footprint from the physics.
- **Tolerance:** `atmosphere_t2.aniso`: `rel` = 0.15 on the SAOS/theory ratio with $n$ = 96 ($3.5\,\mathrm{sd}_{32}/\sqrt3$; worst sd$_{32}$ = 0.073).
- **References:** none specific; the pupil-covariance computation is the test's own.

## `test_seeing_limited_fwhm`
- **Module:** `Atmosphere` + `ScienceCam.get_frame` (no AO).
- **Tier:** nightly. Parametrised over $r_0$ = 0.10, 0.15, 0.25 m.
- **What it checks:** the FWHM of the long exposure of 480 ScienceCam frames (6 atmospheres x 80 frames spaced 16 m along the motion; one layer, $L_0$ = 25 m, 500 nm, 1.5 px per $\lambda/D$, 3.6 arcsec field) against the exact expected long exposure of the finite aperture, and first that the Tokovinin formula agrees with that reference. FWHM is measured on the azimuthal mean about the optical axis (quadratic peak fit, linear half-maximum crossing), with the same estimator on reference and SAOS.
- **Theory / assumptions:** long-exposure OTF $T(\rho)=T_{tel}(\rho)\exp[-D_{vK}(\rho)/2]$ (finite aperture) evaluated by FFT on the camera grid, and the seeing FWHM

$$\mathrm{FWHM} \approx 0.976\,\frac{\lambda}{r_0}\sqrt{1-2.183\,(r_0/L_0)^{0.356}},$$

  valid for $r_0\ll L_0$.
- **Tolerance:** `atmosphere_t2.seeing`: `formula_accuracy` 0.03 (Tokovinin vs finite-aperture reference; measured 1.3 to 2.4 %, deterministic) and `rel` 0.084 (SAOS vs reference; $3.5\times$ worst sd 0.024 at $r_0$ = 0.25). FWHM SAOS/Tokovinin is only recorded.
- **References:** Tokovinin 2002 (PASP 114:1156); Roddier 1981.
