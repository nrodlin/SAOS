# `test_wfs.py` — Shack-Hartmann (point source) validation

Source: `tests/test_wfs.py`

Covers `SAOS.ShackHartmann` (`wfs_measure`, `get_subaps`, `get_psf`, subaperture validity logic, parameter validation) with a point source. The sensor has $6\times6$ subapertures of $d=0.1$ m over a 48 px pupil (8 phase px per subaperture), band R (640 nm), 2 px per $\lambda/d$ unless stated, and is driven with synthetic phases; slopes are in detector pixels. The strategy is to compare against an independent reference written in the test file (`propagate_cog`): a direct Fourier sum of the continuous phase over the physical square subaperture (side $d$, $N_{ref}$ equal cells), evaluated at the detector pixel centres $u_m=(m-(n-1)/2)/\mathrm{samp}$ in units of $\lambda/d$, followed by the centre of gravity of the $n\times n$ window. The reference does not use SAOS code or SAOS-measured quantities; the centroid window truncation is part of the reference, while the spot-motion scale $k_{px}=\lambda/(2\pi p)\cdot 206265/\mathrm{plate\_scale}$ ($p$ = phase pixel) is derived from the nominal geometry. The tolerance policy (header of `tolerances/wfs.yaml`) is that tolerances only cover the numerical precision of the reference or estimator statistics (about 3.5 sigma) and never absorb a measured SAOS discrepancy. All tests are marked `fast` (module-level `pytestmark`).

Common reference model. For a subaperture field $E(\xi,\eta)=\exp[i\phi(\xi,\eta)]$ on $\xi,\eta\in[-1/2,1/2]$, the far field sampled at window pixel $(m_x,m_y)$ is

$$
I(u_x,u_y)=\left|\frac{1}{M^2}\sum_{\xi,\eta}E(\xi,\eta)\,e^{-2\pi i(u_x\xi+u_y\eta)}\right|^2 ,
$$

and the ideal slope is the window centroid relative to the window centre,

$$
s_x=\frac{\sum_{m} (m-\tfrac{n-1}{2})\,I_m}{\sum_m I_m}.
$$

A phase gradient $g$ [rad/px] gives a spot shift $k_{px}\,g$ pixels (small-angle approximation, flat-field, noise-free).

## `test_tilt_axes_isotropy_cross_term`
- **Module:** `ShackHartmann.wfs_measure`.
- **Tier:** fast.
- **What it checks:** a 0.3 px tilt along x on a square pupil gives positive x slopes, negligible y slopes (cross term), and the same response across all subapertures; a tilt along y gives the same response on the y slopes as x does on x (isotropy); a 30 deg tilt gives slopes whose angle is $\arctan(\bar s_y/\bar s_x)=30^\circ$.
- **Theory / assumptions:** slope = mean phase gradient converted to pixels, $s=k_{px}\,g$; sign convention: positive phase gradient along +x gives a positive x slope. Linear regime, noise-free.
- **Tolerance:** `tilt_axes` in `wfs.yaml`: cross term $\le 10^{-3}$ (`cross_max`), isotropy $10^{-4}$, uniformity (std/mean) $10^{-3}$, angle 0.1 deg; all set by float32 FFT precision.
- **References:** none specific.

## `test_tilt_scale_vs_exact_geometry`
- **Module:** `ShackHartmann.wfs_measure` (angular scale of the sensor, window sampling).
- **Tier:** fast.
- **What it checks:** parametrised over `fov_px` = 15, 30. A 0.05 px tilt gives a mean slope that must equal the ideal window centroid from `tilt_cog` (ratio 1). It also records $N$ (samples across the subaperture) and $N/(N-1)$ as metrics.
- **Theory / assumptions:** ideal centroid of the Fourier-sum reference for the nominal motion $k_{px}g$; sampled reference with $N$ cells differs from the continuum by a few tenths of a percent.
- **Tolerance:** `scale_atol` = 0.01 on the ratio (the reference sampled with $N$ cells differs from the continuum by 0.7 % at $N=15$ and 0.2 % at $N=31$ at the largest shift, per the yaml comment).
- **References:** none specific.

## `test_reference_converged`
- **Module:** none from SAOS; tests the reference (`tilt_cog` / `propagate_cog`).
- **Tier:** fast.
- **What it checks:** for shifts 0.05, 0.5, 2, 5 px, the ideal centroid computed with $N=31$ cells differs from the continuum ($M=1024$ cells) by less than the absolute-slope tolerance; the errors at $N$, $2N$, $4N$ are recorded and plotted (expected $\propto 1/N^2$).
- **Theory / assumptions:** midpoint-rule quadrature of the Fourier sum converges as $1/N^2$.
- **Tolerance:** `absolute.rtol` = $5\times10^{-3}$ on the maximum relative error with $N$ cells (measured $\le 0.2$ %).
- **References:** none specific.

## `test_linearity_sweep`
- **Module:** `ShackHartmann.wfs_measure` (full response curve of the centroid).
- **Tier:** fast.
- **What it checks:** for 18 log-spaced spot motions from 0.01 to 6 px, both signs, 31 px window, the mean x slope satisfies $|s-s_{ref}|\le \mathrm{atol}+\mathrm{rtol}|s_{ref}|$ against the ideal window centroid for the nominal motion (no normalisation by the SAOS small-signal gain). The normalised gain curves are recorded as diagnostics only (not asserted).
- **Theory / assumptions:** ideal windowed centroid, including window truncation at large shifts.
- **Tolerance:** `absolute.rtol` = $5\times10^{-3}$, `absolute.atol` = $10^{-3}$ px (float32 FFT / centroid precision); the combined ratio must be $\le 1$.
- **References:** none specific.

## `test_spot_centred_in_window`
- **Module:** `ShackHartmann.get_subaps`, `wfs_measure` (window positioning).
- **Tier:** fast.
- **What it checks:** parametrised over `fov_px` = 15, 16 (odd and even windows). With a flat phase the centre of gravity of the first subaperture image sits at $(n-1)/2$ in x.
- **Theory / assumptions:** an ideal sensor puts the zero-tilt spot at the window centre; this is the origin of the slope definition.
- **Tolerance:** `centring_atol` = 0.05 px.
- **References:** none specific.

## `test_odd_symmetry_all_pixels`
- **Module:** `ShackHartmann.wfs_measure`, centroid on all pixels.
- **Tier:** fast.
- **What it checks:** with an odd 15 px window and all pixels used, the response is odd, $|s(+\theta)+s(-\theta)|/|s(+\theta)-s(-\theta)|$ stays below the bound for shifts 0.05, 0.5, 1.5, 3 px, and the small-signal gain at 0.002 px equals that at 0.05 px (continuity at zero).
- **Theory / assumptions:** symmetric window and symmetric spot imply odd centroid response.
- **Tolerance:** `symmetry.asym_max` = 0.005 (0.5 %); `symmetry.continuity_atol` = 0.002 on the gain ratio.
- **References:** none specific.

## `test_continuity_default_brightest`
- **Module:** `ShackHartmann.wfs_measure` with `use_brightest=50`.
- **Tier:** fast.
- **What it checks:** the small-signal gain at $\pm0.002$ px equals the gain at $+0.05$ px, and the gain at $-0.05$ px equals that at $+0.05$ px (odd, linear). The weighted selection of the brightest pixels keeps the response continuous at zero shift for a point source.
- **Theory / assumptions:** none beyond linearity and oddness.
- **Tolerance:** `symmetry.brightest_atol` = 0.02 (2 %), larger than the all-pixel case because of the tie-break discontinuity.
- **References:** none specific.

## `test_defocus_astigmatism_relations`
- **Module:** `ShackHartmann.wfs_measure` (linear response to quadratic phases).
- **Tier:** fast.
- **What it checks:** scale-free algebraic identities with $c=5\times10^{-4}$ rad/px$^2$: for defocus $\phi=c(x^2+y^2)$ the x slopes are independent of the row and $s_x[i,j]=s_y[j,i]$; for $\phi=cxy$, $s_x=s_y^{\mathrm{defocus}}/2$ and $s_y=s_x^{\mathrm{defocus}}/2$; for $\phi=c(x^2-y^2)$, $s_x=s_x^{\mathrm{defocus}}$ and $s_y=-s_y^{\mathrm{defocus}}$. Differences are normalised by the largest slope.
- **Theory / assumptions:** slope proportional to the mean phase gradient over the subaperture, $\partial_x\phi=2cx$ (defocus), $cy$ (astigmatism). Identities hold for any common scale, so they say nothing about absolute response (tested in `test_quadratic_phase_absolute_slopes`). Small amplitude keeps spot motions below 0.1 px (linear range).
- **Tolerance:** `quadratic.rel_max` = $2\times10^{-3}$ relative to the largest slope.
- **References:** none specific.

## `test_quadratic_phase_absolute_slopes`
- **Module:** `ShackHartmann.wfs_measure`.
- **Tier:** fast.
- **What it checks:** parametrised over `mode` in {defocus, astig_xy, astig_x2y2} and `c` in {0.002, 0.01, 0.03} rad/px$^2$ (9 cases). The slopes of every subaperture must match `quadratic_expected`, which propagates each subaperture independently over the 31 px window, so the reference includes window truncation and phase curvature inside the subaperture (spot motions up to about 3.7 px).
- **Theory / assumptions:** per-subaperture Fourier-sum centroid as above, phase $a_{xx}x^2+a_{yy}y^2+a_{xy}xy$ about the pupil centre; subaperture centres are on the pupil grid (`SUB_C`).
- **Tolerance:** $|s-s_{ref}|\le\mathrm{atol}+\mathrm{rtol}|s_{ref}|$ with `absolute.rtol` = $5\times10^{-3}$, `absolute.atol` = $10^{-3}$ px.
- **References:** none specific.

## `test_defocus_subaperture_geometry`
- **Module:** `ShackHartmann` subaperture grid geometry.
- **Tier:** fast.
- **What it checks:** for defocus with $c=0.002$ the mean gradient over subaperture $j$ is $2cx_j$, so the profile of the slopes is affine in the pupil-grid centres $x_j=n_pj+(n_p-1)/2-(\mathrm{res}-1)/2$. A line $s_j=a x_j+b$ is fitted (scale $a$ absorbed): the centre offset $b/a$ and the maximum residual $/a$ (pitch uniformity), in px of subaperture position, must be small, for both axes.
- **Theory / assumptions:** $\bar{\partial_x\phi}=2c\,x_j$, linear regime.
- **Tolerance:** `geometry.offset_atol` = 0.05 px, `geometry.pitch_atol` = 0.05 px.
- **References:** none specific.

## `test_spot_fwhm`
- **Module:** `ShackHartmann.wfs_measure` / `get_subaps` (spot shape and sampling).
- **Tier:** fast.
- **What it checks:** the flat-phase spot of a square subaperture, 4 px per $\lambda/d$, 31 px window: the FWHM along x and y (band-limited upsampled cut through the peak, then half-maximum crossings) divided by $0.8859\,\lambda/d$ equals 1.
- **Theory / assumptions:** a square aperture of side $d$ gives a $\mathrm{sinc}^2(d\,\theta/\lambda)$ spot, with FWHM $0.8859\,\lambda/d$ (not the $1.03\lambda/D$ of a circular aperture).
- **Tolerance:** `fwhm_atol` = 0.02 on the ratio.
- **References:** Goodman, "Introduction to Fourier Optics" (Fraunhofer diffraction by a rectangular aperture).

## `test_cog_photon_noise`
- **Module:** `ShackHartmann.wfs_measure` with detector shot noise.
- **Tier:** fast.
- **What it checks:** with QE 1, 1 e-/DN, no readout or dark noise and all window pixels, the variance of the slopes over 120 frames (all valid subapertures, x and y) equals $M_2/N_e$ for three flux levels (magnitude 2, 1, 0). $M_2$ is the second central moment of the ideal spot computed in the test; $N_e$ is the electron count measured in the noisy frames themselves, so photometric calibration is not involved.
- **Theory / assumptions:** photon-noise-limited centre of gravity, $\sigma_s^2=M_2/N_e$ with $M_2=\sum(x-x_c)^2I/\sum I$. Valid for negligible per-pixel rounding at these fluxes.
- **Tolerance:** `photon_noise.ratio_atol` = 0.053 on the ratio (8640 samples give $\sigma(\mathrm{var})/\mathrm{var}=\sqrt{2/8640}=1.5$ %; band is 3.5 sigma).
- **References:** none specific (standard photon-noise CoG variance, e.g. Rousset 1999 in Roddier, "Adaptive Optics in Astronomy").

## `test_reference_slopes_averaged`
- **Module:** `ShackHartmann` reference-slope computation (`reference_slopes_maps`, `n_reference_frames`).
- **Tier:** fast.
- **What it checks:** with `noiseFlag=True` the reference slopes are averaged over `n_reference_frames` (100) noisy frames, so $\mathrm{rms}(\text{ref}_{noisy}-\text{ref}_{clean})=\sigma_1/\sqrt n$, where $\sigma_1$ is the single-frame noise rms measured over 30 extra frames; checked as $\sqrt n\cdot\mathrm{ratio}=1$.
- **Theory / assumptions:** independent frame noise, so averaging reduces rms by $\sqrt n$.
- **Tolerance:** `reference.n_sigma` = 3.5: band $1\pm 3.5/\sqrt{2M}$, $M$ the number of valid slopes (estimator statistics).
- **References:** none specific.

## `test_partial_subaperture_energy`
- **Module:** `ShackHartmann` validity of partial subapertures (`valid_subapertures`, `get_psf`).
- **Tier:** fast.
- **What it checks:** parametrised over pupil resolution 48, 96, 192 (ids `res48`, `res96`, `res192`; 8, 16, 32 px per subaperture). For a circular pupil with 20 % central obstruction and flat phase, the valid-subaperture map equals (fraction of pixels in the pupil $\ge 0.5$), and the energy of each valid subaperture image, relative to a fully illuminated one, equals the energy transmitted by the binary mask through that subaperture computed by an exact Fourier sum (`partial_energy_discrepancy`).
- **Theory / assumptions:** the aperture contract: the pupil is a piecewise-constant binary mask, each pixel a uniform cell transmitted with unit amplitude; window energy = $\sum|E_{far}|^2$ over the window. The reference uses at least 256 Fourier cells per subaperture to avoid aliasing (about 1 % error otherwise).
- **Tolerance:** `energy_atol` = $2\times10^{-3}$ (float32 FFT and exact Fourier-sum reference); the valid map must match exactly.
- **References:** none specific.

## `test_partial_energy_converges`
- **Module:** same as above.
- **Tier:** fast.
- **What it checks:** the rms (over valid subapertures) of energy ratio $-1$ is below the tolerance at each of the three pupil resolutions, i.e. the energy is a property of the aperture, not its sampling.
- **Theory / assumptions:** as above.
- **Tolerance:** `energy_atol` = $2\times10^{-3}$.
- **References:** none specific.

## `test_undersampled_spot_rejected`
- **Module:** `ShackHartmann.__init__` parameter validation.
- **Tier:** fast.
- **What it checks:** a plate scale coarser than $\lambda/d$ (`samp=0.8`, spot below 1 px) raises `ValueError` at construction.
- **Theory / assumptions:** convention/validation only.
- **Tolerance:** exception raised.
- **References:** none.

## `test_use_brightest_validated`
- **Module:** `ShackHartmann.__init__` / centroid selection (`use_brightest`).
- **Tier:** fast.
- **What it checks:** parametrised with `n` = 0 (`zero`) and $15^2+1$ (`more_than_window`). Out-of-range `use_brightest` must either raise `ValueError` at construction, or behave as all pixels: 0.3 px tilt slopes equal those of `use_brightest = npix^2`.
- **Theory / assumptions:** validation only.
- **Tolerance:** max slope difference $\le 10^{-6}$ px (hard-coded).
- **References:** none.

## `test_unknown_kwarg_reported`
- **Module:** `ShackHartmann.__init__` keyword handling.
- **Tier:** fast.
- **What it checks:** a misspelt option (`validSubapTreshold`) must raise `TypeError` or log a warning that names it.
- **Theory / assumptions:** validation only.
- **Tolerance:** exception or warning present.
- **References:** none.
