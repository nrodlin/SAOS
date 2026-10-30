# `test_detector.py` — Detector noise statistics, quantisation, saturation, reproducibility (T1)

Source: `tests/test_detector.py`

Covers `SAOS.Detector` (`integrate(frame, photons, stream)`, `register_stream`): the frame is normalised to unit sum and scaled to `photons`, converted to electrons with quantum efficiency (Poisson$(\mathrm{QE}\,N)$ for shot noise), plus Poisson dark current and Gaussian readout noise, clipped to the full well and quantised to DN. Statistics are checked on uniform frames where each pixel is an independent sample (4 frames x 128 x 128 px = 65536 pixels), using sample mean and variance against the closed-form moments of the Poisson and Gaussian laws (Janesick 2001 for the detector model). One tolerance policy applies to the whole file: $3.5\sigma$ of the estimator, with $\sigma$ measured over 30 seeds (`tests/tolerances/detector.yaml`); no tolerance absorbs a measured SAOS discrepancy. Whole module `fast`.

## `test_shot_noise_statistics`
- **Module:** `Detector.integrate` (QE and shot noise).
- **Tier:** fast.
- **What it checks:** on a uniform frame of $N$ = 400 photons/px with QE = 0.6 (1 e-/DN, no other noise) the mean and the variance of the counts both equal $\mathrm{QE}\,N = 240$ e-.
- **Theory / assumptions:** Poisson statistics, $\mathrm{E}[n]=\mathrm{Var}[n]=\mathrm{QE}\,N$; independent pixels; large enough count that rounding to DN is negligible.
- **Tolerance:** `detector.shot_noise`: `mean_rtol` $7.3\times10^{-4}$ (measured $\sigma$ of mean/(QE N) = $2.1\times10^{-4}$; theory $1/\sqrt{\mathrm{QE}N n_{px}}$), `var_rtol` 0.020 ($\sigma$ = $5.6\times10^{-3}$; theory $\sqrt{2/n_{px}}$ for a near-Gaussian variance estimate). Applied as absolute tolerance on the ratio to 1.
- **References:** Janesick 2001 (Scientific CCDs, SPIE Press).

## `test_dark_current_statistics`
- **Module:** `Detector.integrate` (dark current).
- **Tier:** fast.
- **What it checks:** no signal, dark current 2000 e-/s/px over 10 ms: mean and variance both 20 e-.
- **Theory / assumptions:** Poisson$(\mathrm{dark}\cdot t_{int})$.
- **Tolerance:** `detector.dark_current`: `mean_rtol` $2.9\times10^{-3}$ ($\sigma = 8.4\times10^{-4}$, theory $1/\sqrt{20\,n_{px}}$), `var_rtol` 0.017 ($\sigma = 4.8\times10^{-3}$).
- **References:** Janesick 2001.

## `test_readout_noise_on_signal`
- **Module:** `Detector.integrate` (readout noise and quantisation).
- **Tier:** fast.
- **What it checks:** on a bright uniform frame (1000 photons/px, so no clipping at 0) with $\sigma_{RON}$ = 5 e- and $q$ = 1 e-/DN, the variance is $\sigma_{RON}^2 + q^2/12 = 25+1/12$.
- **Theory / assumptions:** Gaussian read noise adds in variance; quantisation of a dithered signal adds the uniform-error variance $q^2/12$ (Sheppard's correction). Shot noise is off, so the signal is deterministic.
- **Tolerance:** `detector.readout_noise.var_rtol` = 0.0175 ($\sigma = 5.0\times10^{-3}$; theory $\sqrt{2/n_{px}}$).
- **References:** Janesick 2001.

## `test_readout_noise_dark_frame`
- **Module:** `Detector.integrate` (readout noise, clipping at 0).
- **Tier:** fast.
- **What it checks:** with no signal, $\sigma_{RON}$ = 5 e- and quantisation 0.01 e-/DN (fine, so that quantisation is negligible), the frame follows a zero-mean Gaussian clipped at 0: mean $\sigma/\sqrt{2\pi}$ and variance $\sigma^2(1/2-1/(2\pi))$. Records the SAOS convention: no bias pedestal and unsigned output, so negative excursions are clipped.
- **Theory / assumptions:** for $x=\max(0,\mathcal N(0,\sigma^2))$:

$$\mathrm{E}[x] = \frac{\sigma}{\sqrt{2\pi}},\qquad \mathrm{Var}[x] = \sigma^2\left(\frac12-\frac1{2\pi}\right).$$

- **Tolerance:** `detector.readout_noise`: `dark_mean_atol` 0.020 on mean$/(\sigma/\sqrt{2\pi})-1$ (sd $=\sqrt{(1/2-1/2\pi)/n_{px}}/0.399 = 5.7\times10^{-3}$, times 3.5) and `var_rtol` 0.0175 on the variance ratio.
- **References:** Janesick 2001 (for read noise); clipped-Gaussian moments are standard.

## `test_quantisation_unbiased`
- **Module:** `Detector.integrate` (ADC quantisation).
- **Tier:** fast.
- **What it checks:** 1000 e- per pixel with $q$ = 10 e-/DN and readout noise of 10 e- (which dithers the ADC): the mean output is 100 DN, i.e. rounding is unbiased; truncation would give 99.5 DN.
- **Theory / assumptions:** rounding of a dithered signal has zero mean error; truncation has a mean error of $-1/2$ DN.
- **Tolerance:** `detector.quantisation.bias_atol_dn` = 0.0146 DN ($\sigma(\mathrm{mean}) = 4.2\times10^{-3}$ DN; theory $\sqrt{1+1/12}/\sqrt{n_{px}}$).
- **References:** Janesick 2001.

## `test_dark_calibration_no_wrap`
- **Module:** `Detector` dark calibration (`darkCalibration` = 20 frames) and its output dtype.
- **Tier:** fast.
- **What it checks:** with readout noise 10 e-, 12 bits, 1 e-/DN, full well 60000 and no signal, the calibrated output stays within $[0, 2^{12}-1]$: negative residuals are clipped at 0 (SAOS convention) and not wrapped to values near 65535 by an unsigned cast. The mean DN is recorded.
- **Tolerance:** hard bounds $0\le \mathrm{DN}\le 4095$. The yaml key `dark_calibration.mean_atol_dn` (0.20 DN) exists but this test does not use it.
- **References:** none.

## `test_full_well_saturation`
- **Module:** `Detector.integrate` (full-well clipping).
- **Tier:** fast.
- **What it checks:** 5000 e- per pixel on a 3000 e- full well reads exactly $\mathrm{FWC}/q$ = 3000 DN everywhere ($q$ = 1).
- **Tolerance:** exact (absolute 0).
- **References:** none.

## `test_infinite_full_well_scale`
- **Module:** `Detector.integrate` with infinite full well and automatic quantisation ($q$ = 0, 16 bits).
- **Tier:** fast. Parametrised over `fwc` = `np.inf` and `float('inf')`.
- **What it checks:** the output keeps the absolute signal scale: a frame with twice the photons gives twice the mean counts (ratio 2) and not zero.
- **Tolerance:** relative $10^{-3}$ on the ratio (hard-coded).
- **References:** none.

## `test_seeded_reproducibility`
- **Module:** `Detector` random generator (`randomState`).
- **Tier:** fast.
- **What it checks:** with QE 0.7, shot noise, readout noise 3 e- and dark current 1000 e-/s, two detectors with the same seed give bit-identical frames; a different seed gives frames that differ (fraction of equal pixels at most 0.5).
- **Tolerance:** exact for the same seed; fraction of equal pixels $\le 0.5$ otherwise (hard-coded).
- **References:** none.

## `test_noise_flag_off_is_identity`
- **Module:** `Detector.integrate` with `noiseFlag=False`.
- **Tier:** fast.
- **What it checks:** the input frame is returned unchanged for a random 128x128 input, even with shot noise and readout noise configured.
- **Tolerance:** exact.
- **References:** none.

## `test_noise_streams_independent_and_reproducible`
- **Module:** `Detector.register_stream`, `Detector.integrate(frame, photons, stream)`.
- **Tier:** fast.
- **What it checks:** streams are numbered 0, 1, ... in connection order; the same stream of an identical detector (same seed) reproduces its noise exactly regardless of the order in which streams are drawn; and two streams give uncorrelated noise (correlation of the noise of a flat 64x64 frame).
- **Theory / assumptions:** the correlation coefficient of two independent noise fields of $n$ pixels has standard deviation $1/\sqrt n$.
- **Tolerance:** correlation within $4/\sqrt{n_{px}}$ (hard-coded, 4 sigma); other criteria exact.
- **References:** none.
