# `test_sciencecam.py` — Science camera PSF and long-exposure accumulation

Source: `tests/test_sciencecam.py`

Covers `SAOS.ScienceCam` (`compute_psf`, `get_frame`, plate scale and field resampling) and the long-exposure accumulation done by `SAOS.LightPath` (`sci_frame`, `long_exposure_frame`). `ScienceCam` is driven directly with synthetic phases (no atmosphere or DM); the long-exposure tests use a `LightPath` whose "atmosphere" is a stub (`StubAtmosphere`) returning a prescribed OPD sequence, so the accumulation is tested in isolation. The validation strategy is comparison with an independent noise-free reference: $|{\rm FFT}|^2$ of the zero-padded pupil sampled at `fw` pixels per $\lambda/D$ (`theory.psf_reference`), cropped to the camera field and normalised to unit energy, plus closed-form results (Airy pattern, tilt-to-shift, Strehl of a sampled phase). Conventions: frames are `[row, col]`; a phase ramp increasing along columns moves the image towards +col; pixel `n_pix//2` is the optical axis. Tolerances are in `tolerances/sciencecam.yaml`, calibrated 2026-10-26 (develop a243eb4, CPU) as a few times the deviation measured with correct behaviour. All tests are marked `fast` (module-level). No test uses the `known_issue` marker although the helper is imported; the yaml refers to known resampling effects (bilinear resample, SCI-SCALE).

Far-field model used by the PSF tests (Fraunhofer, Goodman; discrete implementation as a zero-padded FFT):

$$
\mathrm{PSF}(\boldsymbol\theta)=\left|\mathcal F\{P(\mathbf x)\,e^{i\phi(\mathbf x)}\}\right|^2,\qquad \text{pixel scale}=\frac{\lambda}{N_{pad}\,dx}.
$$

## `test_airy_psf_unobstructed`
- **Module:** `ScienceCam.get_frame` / `compute_psf`.
- **Tier:** fast.
- **What it checks:** flat phase, unobstructed 64 px pupil, camera field sampled exactly like the pupil (`n_pix = res`, 4 px per $\lambda/D$, no resampling). The frame has unit energy, peaks on the optical axis (within 0.02 px), FWHM equal to $1.029\lambda/D$ (`theory.airy_fwhm`), first dark ring at $1.220\lambda/D$ from the azimuthal mean of an 8x Fourier-upsampled frame, and a radial profile within a tolerance of the Airy law for $r<2\lambda/D$.
- **Theory / assumptions:** unobstructed circular aperture,

$$
\frac{I(x)}{I(0)}=\left[\frac{2J_1(\pi x)}{\pi x}\right]^2,\quad x=\theta D/\lambda,
$$

FWHM $=1.029\,\lambda/D$, first zero at $x=1.220$. The pupil is pixelated at 64 px (a small bias against the continuous Airy pattern).
- **Tolerance:** `psf`: `fwhm_rtol` = 0.01 (measured -0.15 %), `first_zero_rtol` = 0.02 (0.25 px radial bins), `airy_profile_atol` = 0.01 of the peak; energy `rtol` = $10^{-12}$ (hard-coded).
- **References:** Goodman, "Introduction to Fourier Optics".

## `test_psf_obstructed_vs_reference`
- **Module:** `ScienceCam.get_frame` for an obstructed pupil.
- **Tier:** fast.
- **What it checks:** with a 0.3 central obstruction, flat phase, `n_pix = res`, the frame equals the independent zero-padded FFT of the same pupil mask (`theory.psf_reference`, normalised to unit sum); compares the FWHM ratio and $\max|{\rm frame}-{\rm ref}|/{\rm peak}$.
- **Theory / assumptions:** same Fraunhofer model; SAOS and the reference should be the same computation when no resampling is needed.
- **Tolerance:** `psf.fwhm_rtol` = 0.01; `psf.reference_profile_atol` = $10^{-6}$ of the peak.
- **References:** Goodman.

## `test_image_motion_tilt`
- **Module:** `ScienceCam.get_frame` (image motion, sign convention).
- **Tier:** fast.
- **What it checks:** phase ramps along x (columns) and y (rows) of 0.5, 1, 2, 4, 6 $\lambda/D$, camera sampled like the pupil. The peak (16x Fourier upsampling plus parabola) moves by $\theta/\mathrm{plate\_scale}=\theta\cdot fw$ px on the tilted axis (ratio 1), with no motion on the other axis and positive ramp giving a positive pixel shift.
- **Theory / assumptions:** Fourier shift theorem: $\phi=2\pi\,t\,x/D$ gives a shift of $t\,\lambda/D$ in the focal plane, i.e. $t\,fw$ px; the peak estimator error is about 0.01 px for shifts of 2-24 px.
- **Tolerance:** `motion.shift_rtol` = 0.003, `motion.cross_axis_atol_px` = 0.02 px.
- **References:** Goodman.

## `test_plate_scale_with_resampled_pupil`
- **Module:** `ScienceCam` field resampling (`n_pix != res`).
- **Tier:** fast.
- **What it checks:** for `n_pix = 160` and `res = 64` (the typical case, e.g. EST red 555 px vs 216) a 4 $\lambda/D$ tilt moves the image by 4 fw px and the FWHM equals $1.029\lambda/D$, both within 0.2 %. The `align_corners` scale model $n_{pix}(res-1)/((n_{pix}-1)res)$ is recorded as a metric only.
- **Theory / assumptions:** requested plate scale must be the effective plate scale.
- **Tolerance:** `plate_scale.rtol` = 0.002.
- **References:** none specific.

## `test_strehl_vs_native_grid`
- **Module:** `ScienceCam.compute_psf` (Strehl after pupil-phase resampling).
- **Tier:** fast.
- **What it checks:** on-axis intensity of the PSF relative to the aberration-free one, for high-order phases (PSD $f^{-11/3}$ above 8 cycles/$D$, rms 0.5 and 0.9 rad, 4 realisations each, camera field $n_{pix}=2.5\,res$), equals the exact Strehl of the sampled phase on its native grid, averaged per rms.
- **Theory / assumptions:** exact Strehl of the sampled phase over the pupil support,

$$
S=\left|\langle e^{i\phi}\rangle_{P}\right|^2 ,
$$

which is the on-axis value ratio (not the Maréchal approximation $S\approx e^{-\sigma^2}$, which is not used for the criterion). The ratio of the mean SAOS SR to the mean exact SR (per rms) must be 1; the phase is resampled from the 64 px pupil grid to the camera pupil grid inside `compute_psf`.
- **Tolerance:** `strehl.sr_rtol` = 0.02.
- **References:** none specific; Strehl definition as in Roddier (ed.), "Adaptive Optics in Astronomy".

## `test_long_exposure_frame_count`
- **Module:** `LightPath.propagate` long-exposure logic with `ScienceCam(integrationTime, decimation)`.
- **Tier:** fast.
- **What it checks:** parametrised over `fs` = 1000, 2000, 1500, 1800 Hz. A long exposure of $T_{int}=k\cdot\mathrm{decimation}\cdot T_s$ (written as $k/f_s$, as a user would) must accumulate exactly $k$ short frames, for $k=1..12$ with decimation 1 and $k=1..6$ with decimation 2. The non-integer $f_s$ values (1500, 1800) are the floating-point rounding cases.
- **Theory / assumptions:** frame counting only.
- **Tolerance:** exact (`atol` = 0).
- **References:** none.

## `test_long_exposure_non_integral_ratio`
- **Module:** `LightPath.propagate` long-exposure logic.
- **Tier:** fast.
- **What it checks:** when $T_{int}$ is not a multiple of $T_s$ the exposure is rounded up and never shorter than $T_{int}$: 2.4 $T_s$ gives 3 frames; 1/56 s at 2 kHz (35.7 $T_s$) gives 36 frames.
- **Theory / assumptions:** convention (ceiling).
- **Tolerance:** exact.
- **References:** none.

## `test_long_exposure_mean_and_energy`
- **Module:** `LightPath` long-exposure accumulation with `ScienceCam`.
- **Tier:** fast.
- **What it checks:** with random tilts per frame from the stub atmosphere ($k=8$ frames, 32 px pupil, `lightRatio` 0.9), the noise-free long exposure equals the mean of the independent unit-energy short frames, scaled to the collected photons. Two criteria: total energy against $\mathrm{flux}\cdot T_{int}\cdot A\cdot\text{throughput}$ and $\max|{\rm LE}/{\rm photons}-\bar I_{ref}|/\mathrm{peak}$.
- **Theory / assumptions:**

$$
\mathrm{LE}=\Phi\,T_{int}\,A_{pix}\,\eta\ \frac{1}{k}\sum_{n=1}^{k}I_n,\qquad \sum I_n=1,
$$

$A_{pix}$ the pixelated pupil area $\sum P\,(D/res)^2$, $\eta$ = `lightRatio`. Expected frames come from an independent FFT with the tilt ramp.
- **Tolerance:** `long_exposure.energy_rtol` = $10^{-10}$, `long_exposure.mean_rtol` = $10^{-10}$.
- **References:** none specific.

## `test_long_exposure_gaussian_tilt`
- **Module:** `LightPath` long-exposure accumulation with `ScienceCam`.
- **Tier:** fast.
- **What it checks:** Gaussian image motion ($\sigma=0.8\lambda/D$ per axis) via a $30\times30$ grid of Gaussian quantiles, 900 frames. (1) LE equals the mean of independent frames of the tilts actually used; (2) LE equals the independent diffraction-limited frame convolved with a Gaussian (FWHM ratio and $\max|{\rm LE}-{\rm ref}|/\mathrm{peak}$). The quadrature law $\mathrm{FWHM}_{LE}^2=\mathrm{FWHM}_{DL}^2+(2.355\sigma)^2$ is reported only (approximate for an Airy core).
- **Theory / assumptions:** random image motion with distribution $p$ blurs the long exposure into $\mathrm{PSF}_{DL}*p$; in Fourier space the Gaussian transfer function is $\exp[-2(\pi\sigma f)^2]$ in the code's pixel units ($\sigma_{px}=\sigma\,fw$). The quantile grid approximates the continuous Gaussian to under 1 % (about 0.3 % in the grid itself).
- **Tolerance:** `long_exposure`: `discrete_profile_atol` = $10^{-3}$ (exact discrete comparison; measured 2e-4 at the FoV edge), `gaussian_fwhm_rtol` = 0.015, `gaussian_profile_atol` = 0.015.
- **References:** none specific.
