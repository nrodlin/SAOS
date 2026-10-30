# `test_corrsh.py` — Correlating Shack-Hartmann (solar, extended scene)

Source: `tests/test_corrsh.py`

Covers `SAOS.CorrelatingShackHartmann` with `SAOS.ExtendedSource` in three layers: spot optics (`get_psf`), image formation (`compute_images`) and the end-to-end correlation measurement (`wfs_measure`, `initialize_wfs`). Configuration: $6\times6$ subapertures of $d=0.1$ m over a 48 px pupil, V band (500 nm), plate scale 0.4"/px ($\lambda/d=2.58$ px), 10" field, `ExtendedSource` with $3\times3$ sub-directions (sun patches of 58 px per subaperture), `use_brightest=5` unless stated. Independent references written in the test file (no SAOS code): (i) the far field of a subaperture as a direct Fourier sum of the continuous amplitude (`ref_spot`), sampled at angle $(k-n_{sun}//2)\,\mathrm{plate\_scale}$, the contract of `get_psf` (PSF centred at pixel $n_{sun}//2$, unit energy in the window); (ii) a direct circular convolution of an object built in the test; (iii) the true image shift $\theta/\mathrm{plate\_scale}$ of a global tilt, plus an independent least-squares image registration (`register_shift`). A global tilt of $\theta$ arcsec moves the image by $\theta/\mathrm{plate\_scale}$ px, so a phase gradient of $K_G=2\pi p\,\mathrm{ps}/\lambda$ rad/px moves it by one pixel. All tests are marked `fast`. Tolerances come from `tolerances/corrsh.yaml` (calibrated 2026-10-27, CPU float32; policy: numerical precision of the reference or estimator statistics, never absorbing a measured SAOS discrepancy).

Image formation model used by the convolution tests: the subaperture image is the object convolved with the subaperture PSF (circular convolution on the patch),

$$
I_k(\mathbf r)=\sum_{\mathbf a}\mathrm{PSF}_k(\mathbf a)\,O(\mathbf r-\mathbf a).
$$

## `test_spot_tilt_displacement`
- **Module:** `CorrelatingShackHartmann.get_psf`.
- **Tier:** fast.
- **What it checks:** for global tilts of 0.05, 0.3, 1 and 2 px, both signs and both axes, each subaperture spot moves by exactly the tilt along its axis and not along the other. Compared against `ref_spot`: centroid on the direct and cross axis, spread of the centroid across subapertures, the whole spot image, and the sinc$^2$ fit of the cut through the peak against the true shift (independent of window truncation).
- **Theory / assumptions:** square aperture far field $\mathrm{sinc}^2$; a linear phase $2\pi(\theta/(\lambda/d))\,\xi$ displaces the spot by $\theta$. Centroid relative to pixel $n_{sun}//2$; window truncation belongs to the reference. Fitted shape $A\,\mathrm{sinc}^2((k-k_0)/\mathrm{samp})$.
- **Tolerance:** `tilt` keys: `cog_atol` = 0.002 px (measured 4.3e-4), `image_atol` = 0.002 of the reference peak (measured 3.6e-4), `peak_atol` = $10^{-3}$ px (measured 3e-6).
- **References:** Goodman, "Introduction to Fourier Optics".

## `test_flat_spot_centred`
- **Module:** `CorrelatingShackHartmann.get_psf`.
- **Tier:** fast.
- **What it checks:** with flat phase every spot peaks at pixel $n_{sun}//2$, is mirror symmetric ($I[k]=I[n-k]$), has the same centroid as the reference window (which is slightly non-zero because the window extent $-n//2\ldots n-n//2-1$ is asymmetric), and has unit energy in the window.
- **Theory / assumptions:** the far field of a square aperture is real and even.
- **Tolerance:** `centring`: `mirror_atol` = $10^{-5}$ of the peak (measured 2e-10), `cog_atol` = $5\times10^{-4}$ px (measured 9e-5), `energy_atol` = $10^{-5}$.
- **References:** none specific.

## `test_spot_second_moment`
- **Module:** `CorrelatingShackHartmann.get_psf`.
- **Tier:** fast.
- **What it checks:** parametrised over `phase` = flat, defocus. The second central moment $M_2$ (x and y) of the spot on the $n_{sun}$ window, a quantity that sets correlation width and centroid noise, equals that of the continuous-aperture Fourier sum on the same window. Defocus case: $\phi=c(x^2+y^2)$, $c=0.01$ rad/px$^2$, evaluated per subaperture at its own centre; flat case checks one subaperture (identical).
- **Theory / assumptions:** $M_2=\sum (k-\bar k)^2 I_k/\sum I_k$. The pupil sampling of `get_psf` has a design bias of about $1.22(u/n)^2$ with $u$ the patch half-width in $\lambda/d$ and $n$ the pupil samples per subaperture (about 0.94 % here).
- **Tolerance:** `m2.rel_atol` = 0.01 on $M_2/M_{2,ref}-1$ (measured +0.77 % flat, +0.66 % defocus).
- **References:** none specific.

## `test_partial_subaperture_energy_and_shape`
- **Module:** `CorrelatingShackHartmann` validity map and `get_psf` for partially lit subapertures.
- **Tier:** fast.
- **What it checks:** circular pupil with 20 % obstruction (48 px binary mask), flat phase: (1) valid iff at least 50 % of pixels lie in the pupil; (2) window energy equals the transmitted fraction (mean of the mask block), and invalid subapertures have zero energy; (3) the spot shape equals the far field of the binary mask block (amplitude 1 on its pixels), normalised to unit sum, for all partially lit valid subapertures.
- **Theory / assumptions:** same aperture contract as in `test_wfs.py`: the binary mask is a piecewise-constant amplitude; energy proportional to illuminated area.
- **Tolerance:** `partial.energy_atol` = $10^{-5}$ (measured 1.3e-7), `partial.shape_atol` = $10^{-4}$ of peak (measured 2e-6; mask cells align exactly with pupil samples, 16 per pixel).
- **References:** none specific.

## `test_defocus_mirror_antisymmetry`
- **Module:** `CorrelatingShackHartmann.get_psf`.
- **Tier:** fast.
- **What it checks:** defocus $\phi=c(x^2+y^2)$, $c=0.013$ rad/px$^2$ (spot motion up to about 2 px): the x displacement of subaperture $(i,j)$ is minus that of its mirror $(i,N-1-j)$ and independent of the row (and likewise for y across rows); displacements also equal the per-subaperture reference. The reference uses the piecewise-linear continuation of the pixel map of the quadratic (the input is a pixel map), and the antisymmetry bound includes the reference's own residual caused by the asymmetric window extent.
- **Theory / assumptions:** mean gradient over subaperture $j$ is $2c\,x_j$, odd about the pupil centre; window extent asymmetry gives a small reference residual (0.0097 px per the yaml).
- **Tolerance:** `defocus.asym_atol` = $10^{-3}$ px added to the reference's own antisymmetry residual; `defocus.cog_atol` = $10^{-3}$ px against the pixel-sampled reference (measured 3.8e-4).
- **References:** none specific.

## `test_convolution_delta_not_shifted`
- **Module:** `CorrelatingShackHartmann.compute_images`.
- **Tier:** fast.
- **What it checks:** parametrised over `margin` = 8.0 (`npix_sun_even`) and 8.4 (`npix_sun_odd`). The image of a one-pixel object is the PSF placed at that pixel: a delta at the PSF origin $n_{sun}//2$ returns the PSF unshifted for even and odd $n_{sun}$, and a delta at offsets (0,0), (-9,6), (13,-4) returns the PSF circularly shifted by that offset. Compared as whole images and as centroids, with the PSF carrying a tilt of (0.7, -0.4) px.
- **Theory / assumptions:** convolution with a delta is a shift; circular (FFT) convolution.
- **Tolerance:** `conv.image_atol` = $10^{-5}$ of the peak (measured 1.5e-7 to 2.7e-8), `conv.cog_atol` = $10^{-4}$ px (measured 4e-6).
- **References:** none specific.

## `test_convolution_matches_direct_and_conserves_flux`
- **Module:** `CorrelatingShackHartmann.compute_images`.
- **Tier:** fast.
- **What it checks:** a random smooth positive object per sub-direction (58 px) convolved with spots of the obstructed pupil under flat and tilted phases: (1) selected images equal the direct circular convolution of the object with the PSF (shape at unit sum); (2) the image flux equals the object flux times the transmitted energy of the subaperture, with the same constant for all subapertures and sub-directions.
- **Theory / assumptions:** a unit-sum PSF conserves flux; direct convolution by summing rolled copies, $\mathrm{out}=\sum_{ab}\mathrm{PSF}[a,b]\,\mathrm{roll}(O,(a-n//2,b-n//2))$.
- **Tolerance:** `conv.image_atol` = $10^{-5}$; `conv.flux_atol` = $10^{-5}$ on std/mean of the flux ratio (measured 1e-7).
- **References:** none specific.

## `test_global_tilt_linearity_and_cross_coupling`
- **Module:** `CorrelatingShackHartmann.wfs_measure` (correlation estimator with pseudo-reference, `use_brightest` = 9).
- **Tier:** fast.
- **What it checks:** a fixed high-order residual phase (0.3 rad rms) makes the subaperture images differ so the pseudo-reference (subaperture `pseudoref_index`) is a genuinely different image; that subaperture is excluded. A global tilt scans 41 shifts over $[-R,R]$, $R=4"/0.4=10$ px (80 % of the half window), along x and then y. Linearity: RMS of the mean slopes about a fitted line $gs+b$, relative to $R$. Cross coupling: maximum over shifts of the RMS over subapertures of the other-axis slopes, relative to $R$. The gain, offset and worst-subaperture errors are recorded as information. An independent least-squares registration of the images (shifts up to 3 px) is also recorded to separate image from estimator errors.
- **Theory / assumptions:** ideal response is the unit-gain line $s=\text{shift}$; the gain is absorbed by the interaction matrix so only departure from a line is penalised. Correlation-peak estimator (Michau, Rousset & Fontanella 1993; Löfdahl 2010) on extended scenes.
- **Tolerance:** `linearity.lin_frac` = 0.02 and `linearity.cross_frac` = 0.03 of full scale, the specified linearity and cross-coupling bounds (cross coupling is only exceeded at the window-edge point of 4.25").
- **References:** Michau, Rousset & Fontanella 1993; Löfdahl 2010, A&A 524:A90.

## `test_reference_averaged`
- **Module:** `CorrelatingShackHartmann.initialize_wfs` (pseudo-reference and reference slopes with `noiseFlag=True`).
- **Tier:** fast.
- **What it checks:** the pseudo-reference is the average of `n_reference_frames` images and the reference slopes the average of $n$ further correlated frames. With $\sigma_1$ the single-frame slope noise (30 frames), $\sqrt n\,\mathrm{rms}(\text{ref}_{noisy}-\text{ref}_{clean})/\sigma_1$ lies in $[1,\sqrt2]$ widened by the statistics band, excluding the pseudo-reference subaperture. Uses photon noise only.
- **Theory / assumptions:** the reference retains the noise of the $n$ averaged frames, variance $\sigma_1^2/n$, plus the effect of the residual noise of the averaged pseudo-reference, between 0 and $\sigma_1^2/n$ (linear propagation, symmetric in the two images).
- **Tolerance:** `reference.n_sigma` = 3.5: band widened by $3.5/\sqrt{2M}$, $M$ the number of slopes ($n$ = 100).
- **References:** none specific.

## `test_tilt_axes_isotropy_and_sign`
- **Module:** `CorrelatingShackHartmann.wfs_measure`, compared with `ShackHartmann`.
- **Tier:** fast.
- **What it checks:** (1) sign convention: a tilt towards +x (+y) gives positive x (y) slopes, with the sign agreeing with `ShackHartmann` for the same phase, plate scale and telescope; (2) isotropy: with the sun patches and sub-directions transposed (rows and columns exchanged), the y response equals the original x response, and the cross responses match (0.5 px shift).
- **Theory / assumptions:** convention and symmetry only.
- **Tolerance:** `isotropy.atol` = $2\times10^{-4}$ px (same code path for rows and columns; measured 4e-5); signs exact.
- **References:** none specific.

## `test_unit_in_rad_constructs`
- **Module:** `CorrelatingShackHartmann.__init__` (`unit_in_rad`, `slopes_units`).
- **Tier:** fast.
- **What it checks:** `unit_in_rad=True` constructs and gives a finite positive px/rad conversion `slopes_units`.
- **Theory / assumptions:** none (construction/convention).
- **Tolerance:** boolean check.
- **References:** none.

## `test_unit_in_rad_reference_slopes`
- **Module:** `CorrelatingShackHartmann.initialize_wfs`, `wfs_measure` with `unit_in_rad=True`.
- **Tier:** fast.
- **What it checks:** the reference slopes are in rad like the measurements: a null phase gives zero slopes (below $10^{-9}$ rad), and a 0.5 px tilt gives slope$_{rad}\times$`slopes_units` equal to the slope in px of the `unit_in_rad=False` sensor.
- **Theory / assumptions:** unit conversion only.
- **Tolerance:** null slope $\le10^{-9}$, ratio 1 with `rtol` = $10^{-6}$ (hard-coded).
- **References:** none.

## `test_photons_scale_with_fov_squared`
- **Module:** `CorrelatingShackHartmann.wfs_measure` (photon budget).
- **Tier:** fast.
- **What it checks:** with an exact-count detector (no shot noise, QE 1, no readout or dark noise, 1 e-/DN) the frame electrons equal `flux` $\times D^2\times t\times(\mathrm{FoV}/\mathrm{src.fov})^2$ for FoV 6" and 10", and the ratio of the two frames is 0.36.
- **Theory / assumptions:** photons collected scale with sky area observed and with the pupil area and integration time (1 ms); `lightRatio` = 1.
- **Tolerance:** `photons.rtol` = $10^{-3}$ (per-pixel rounding of the photon frame, about $4\times10^{11}$ photons).
- **References:** none specific.
