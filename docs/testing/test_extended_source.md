# `test_extended_source.py` — Extended (solar) source with ScienceCam and Atmosphere

Source: `tests/test_extended_source.py`

Covers `SAOS.ExtendedSource` (sub-direction patches, blending filters, flux), as used through `LightPath` with `ScienceCam` and `Atmosphere`. The scenario reproduces an asterism workflow: an h5 image of delta-function "stars" at 0.0128"/px (image[x_px, y_px]: first axis = image row) observed by an EST-like telescope (D 4.149 m, obstruction 1.3 m, 450 px resolution, $3\times3$ sub-directions, FoV 9.975", padding 5", margin 4"), with a ScienceCam at 0.0128"/px. Besides the 17 stars of the asterism, a marker star at $(x,y)=(+3.0",-1.2")$ breaks the D4 symmetry so transposition or mirroring is detected. The same 18 directions are also observed as point sources (NGS) for reference. Every expectation is computed independently of SAOS: own geometry, own FFT of an analytic annular pupil with $4\times4$ sub-pixel occupancy and bilinear phase (`RefPSF`, $N=1024$, pixel scale exactly $\lambda/(\mathrm{ps}\,N\,dx)$), own sampling of the atmosphere screen (cubic sub-pixel windows), own convolution. The telescope resolution (450 px, about 2 per camera pupil sample) keeps the `ScienceCam` resampling error below 0.2 % of the PSF peak. Metrics are windows of 79 px (1.0" half-width) normalised to unit sum, compared as $\max|{\rm window}-{\rm ref}|/\max({\rm ref})$. Frames are `[row, col]`, with row = sky y in the SAOS atmosphere/DM frame (an NGS observing the star written at image[x_px, y_px] is placed at azimuth $90^\circ-\varphi$). All tests are marked `nightly` (module-level); fixtures are module-scoped and use the GPU when available. Tolerances: `tolerances/extended_source.yaml` (calibrated 2026-10-26 on develop, GPU, 8 seeds where stated, typically mean + 3.5 sd).

Mosaic model used in the altitude-layer tests: the extended object is split into $n\times n$ patches of size $s=2(\mathrm{fov}+\mathrm{pad})/(n+1)$ with centres every $s/2$ (50 % overlap) and blended with tent weights that sum to one everywhere,

$$
I(\mathbf r)=\sum_{i,j} w_i(x)\,w_j(y)\,\big[O * \mathrm{PSF}_{ij}\big](\mathbf r),
$$

where $\mathrm{PSF}_{ij}$ is the PSF of the line of sight to the centre of patch $(i,j)$, which crosses a layer at altitude $h$ displaced by $h\tan\theta$ (rounded to the layer pixel, the integer-pixel model).

## `test_b1_no_atmosphere_stars`
- **Module:** `ExtendedSource` + `ScienceCam` through `LightPath.propagate`.
- **Tier:** nightly.
- **What it checks:** with no atmosphere, each of the 18 stars appears at its input pixel (position error within 0.1 px, found by Fourier upsampling and a parabola), has the FWHM of the obstructed-pupil PSF, and its 79 px window matches the independent PSF window. NGS PSFs of the same directions are recorded for comparison only.
- **Theory / assumptions:** Fraunhofer PSF of the annulus, $\mathrm{PSF}=|\mathcal F\{P\}|^2$, computed by zero-padded FFT with $N\,dx=\lambda/\mathrm{ps}$.
- **Tolerance:** `b1`: `pos_atol_px` = 0.1 px (measured 0.025), `fwhm_rtol` = 0.005 (measured -0.10 %, from the 450 to 898 px resampling), `psf_atol` = 0.006 of the peak (measured 0.08-0.23 %, 2.6x margin).
- **References:** Goodman, "Introduction to Fourier Optics".

## `test_b1_production_sampling_energy`
- **Module:** `ScienceCam.get_frame` (extended source and point source), `ScienceCam.compute_psf`.
- **Tier:** nightly.
- **What it checks:** no atmosphere, production pupil sampling (200 px over 4.149 m, so $\lambda/\Delta x = 4.97''$ at 500 nm lies inside the 9.975″ field; B1 uses 450 px, where it does not). The fraction of the extended-source frame outside the 18 star windows (1.0″ half-width) must not exceed $1 - EE_{ref}$, and every point-source frame must keep $EE_{ref}$ of its energy in the window, with $EE_{ref}$ the encircled energy of the independent annular-pupil PSF (own FFT) in the same window.
- **Theory / assumptions:** energy conservation of the image: the unit-sum windows of B1 cannot detect energy lost from the core of every star (for instance to PSF replicas). Stars near the field edge lose part of their halo out of the field, so the measured outside fraction can only be smaller than $1 - EE_{ref}$.
- **Tolerance:** `extended_source.b1_production`: `bg_atol` 0.005 (measured 0.0109 vs bound 0.0114), `ee_atol` 0.005 (measured 0.9869 vs 0.9886).
- **References:** Goodman, *Introduction to Fourier Optics*.

## `test_b2_image_not_transposed_or_mirrored`
- **Module:** `ExtendedSource` / `ScienceCam` image orientation.
- **Tier:** nightly.
- **What it checks:** the number of peaks in the noise-free frame is 18, all stars are found at the input pixels (identity) and for every other element of the dihedral group (transposition, row/column mirrors, rotations by 90, 180, 270 degrees and anti-transposition) fewer than all 18 stars match (the marker breaks the symmetry).
- **Theory / assumptions:** pure geometry/convention test, first image axis maps to frame rows.
- **Tolerance:** match radius 0.7 px; counts exact.
- **References:** none.

## `test_b2_subdirection_footprint_orientation`
- **Module:** `Atmosphere.getOPD` for the extended source (sub-direction footprint on a layer).
- **Tier:** nightly.
- **What it checks:** one frozen layer at 15 km. The footprint offset of each of the 9 sub-directions relative to the central one is measured by cross-correlating whitened OPD maps (parabolic sub-pixel peak) and compared with $h\tan(\theta)$ (sky x on rows) and with the transposed-axes version. For all off-diagonal sub-directions ($a\ne b$) none may be closer to the transposed offset than to the physical one.
- **Theory / assumptions:** the line of sight at sky offset $(a,b)$ crosses the layer displaced by $(h\tan a,h\tan b)$ rows/columns; whitening removes low-order content, $a_w=a-G_8 * a$.
- **Tolerance:** orientation criterion, count of misoriented sub-directions = 0 (no numerical tolerance).
- **References:** none.

## `test_b2_subdirection_footprint_subpixel`
- **Module:** `Atmosphere.getOPD` for the extended source.
- **Tier:** nightly.
- **What it checks:** same measurement, restricted to the diagonal sub-directions (0, 4, 8) where transposition makes no difference; the measured offset must equal $h\tan\theta$ rounded to the nearest layer pixel.
- **Theory / assumptions:** integer-pixel footprint model (the layer is displaced by whole layer pixels).
- **Tolerance:** `b2.offset_atol_px` = 0.09 px: precision of the offset estimator on exact sub-pixel windows of the same screens, 8 seeds, mean 0.0557, sd 0.0095, plus 3.5 sd.
- **References:** none.

## `test_b3_ground_layer_isoplanatic`
- **Module:** `ExtendedSource` + `Atmosphere` (single layer at 0 m, $r_0=0.6$ m, $L_0=25$ m).
- **Tier:** nightly.
- **What it checks:** with a ground layer every direction sees the same phase, so each star must have the PSF of that screen (own FFT of the annulus with the screen's phase window) and equal the NGS PSF of its direction. Both comparisons are per star on 79 px windows; independent Strehl is recorded.
- **Theory / assumptions:** isoplanatic ground layer: $\mathrm{PSF}=|\mathcal F\{P\,e^{i\phi}\}|^2$ with the same $\phi$ for all patches, so the image of the object is the object convolved with one PSF and the blending is seamless.
- **Tolerance:** `b3.ref_atol` = 0.012 (extended vs independent: 0.0059-0.0091 over 8 seeds, plus 3.5 sd), `b3.ngs_atol` = 0.008 (extended vs NGS: 0.0030-0.0065, plus 3.5 sd; resampling 898 vs 779 px).
- **References:** none specific.

## `test_b4_altitude_layer_orientation`
- **Module:** `ExtendedSource` + `Atmosphere` (layer at 15 km).
- **Tier:** nightly.
- **What it checks:** the expected star window $E$ (tent-weighted per-patch PSFs of the screen with footprint offsets rounded to the layer pixel) and $E_t$ (transposed layer axes). Among the stars for which they differ by more than max(`b4.ref_atol`, `b4.orientation_min_sep`), none may be closer to $E_t$ than to $E$.
- **Theory / assumptions:** mosaic model above. The 5 % separation threshold is a selection of discriminating stars, not a tolerance.
- **Tolerance:** `b4.orientation_min_sep` = 0.05 and `b4.ref_atol` = 0.0011 (numerical precision of the reference vs a 2048 px FFT / $8\times8$ sub-pixels / cubic-sampled phase, 8 seeds: mean 6.8e-4, sd 1.1e-4, plus 3.5 sd); count of misoriented stars = 0.
- **References:** none.

## `test_b4_altitude_layer_psf_vs_screen`
- **Module:** `ExtendedSource` + `Atmosphere` (layer at 15 km).
- **Tier:** nightly.
- **What it checks:** for every star, $\max|\mathrm{ext}-E|/\max E$ is below the tolerance.
- **Theory / assumptions:** as in the orientation test; the criterion covers the extended-image model error (patch resampling and PSF blending, 0.6-0.9 % on the ground layer in b3).
- **Tolerance:** `b4.psf_atol` = 0.02 (measured 0.13-0.45 % on one seed).
- **References:** none.

## `test_b4_anisoplanatic_difference_grows_with_patch_distance`
- **Module:** `ExtendedSource` + `Atmosphere`, comparison with NGS PSFs.
- **Tier:** nightly.
- **What it checks:** the extended-object star PSF sees the phase of its patch centre(s) while the NGS sees the phase of its own direction, so their difference is zero at a patch centre and grows with the distance to the patch centres. Criteria: maximum difference at patch centres below a floor; mean off-centre difference divided by mean difference at centres above a minimum; Spearman correlation between the difference and $\sigma^2$ above a minimum. Peak ratio ext/NGS against $e^{-\sigma^2}$ is recorded only.
- **Theory / assumptions:** anisoplanatic phase variance of a star $\sigma^2=\sum_k w_k\,S(h\,d_k)$, with $d_k$ the angular distance to patch centre $k$ (converted by $h\tan d_k$) and $S(r)$ the mean squared phase difference at lag $r$ measured on the actual screen (average of row and column lags, `citools.structure_function`).
- **Tolerance:** `b4_growth`: `floor` = 0.014 (0.0045-0.0091 plus 3.5 sd), `ratio_min` = 3.0 (measured 6.6-17), `spearman_min` = 0.70 (measured 0.757-0.840, mean 0.808, sd 0.028, minus 3.5 sd), 8 seeds.
- **References:** none specific.

## `test_b5_uniform_object_gives_uniform_image`
- **Module:** `ExtendedSource` + `ScienceCam` mosaic blending.
- **Tier:** nightly.
- **What it checks:** parametrised over camera `plate` = 0.0128 and 0.016 arcsec/px. A constant object must give a constant image regardless of the phase in each patch, with a flat phase and with the 15 km layer: no NaN, std/mean and $\max|f/\bar f-1|$ below a float64 floor, frame energy 1.
- **Theory / assumptions:** convolution of a constant with a unit-energy PSF is the same constant; the blending filters form a partition of unity, so there are no seams.
- **Tolerance:** `b5_uniform.floor` = $10^{-10}$ (measured 2e-16 and 2e-15); energy `rtol` = $10^{-12}$ (hard-coded).
- **References:** none.

## `test_b5_structured_object_mosaic_has_no_seams`
- **Module:** `ExtendedSource` + `ScienceCam` mosaic blending.
- **Tier:** nightly.
- **What it checks:** a smooth random texture ($\pm30$ % contrast, correlation length 12 px, seed `texture_seed`) observed through the ground layer must equal the object convolved with the PSF of the screen (own `fftconvolve` with a 601 px PSF) over the whole 779 px frame; maximum and rms deviation relative to the mean.
- **Theory / assumptions:** all patches have the same PSF in the ground-layer case, so the frame is a single convolution; seams or offsets would appear as steps of the order of the object contrast. The reference truncates the PSF at $\pm300$ px and the 898 px patches wrap circularly.
- **Tolerance:** `b5_mosaic.max_atol` = 0.005 (measured 0.0021), `rms_atol` = 0.0015 (measured 0.0006).
- **References:** none.

## `test_b5_filters_partition_of_unity`
- **Module:** `ExtendedSource.filter_2D`.
- **Tier:** nightly.
- **What it checks:** parametrised over `n_sub` = 2, 3, 4, 5. With patch $(i,j)$ placed at $(i,j)\times\mathrm{round}(W/2)$ in a mosaic of size $(n-1)\,\mathrm{round}(W/2)+W$ (50 % overlap as `ScienceCam` places them), the sum of the SAOS filters is 1 at every pixel.
- **Theory / assumptions:** $\sum_{ij}f_{ij}(\mathbf r)=1$.
- **Tolerance:** min and max of the sum equal to 1 with `rtol` = $10^{-12}$ (hard-coded).
- **References:** none.

## `test_b5_flux`
- **Module:** `ExtendedSource.flux`, `LightPath.long_exposure_frame`, `ScienceCam.lightRatio`.
- **Tier:** nightly.
- **What it checks:** (a) source flux equals the independent photon flux; sub-direction fluxes sum to it; (b) the long-exposure frame holds flux $\times T\times A_{eff}$ photons with `lightRatio=1`; (c) the noise-free frame equals the independent convolution of the object with the PSF over the 779 px frame (maximum difference relative to the peak) and the flux within 1" of each star matches. The `lightRatio=50` case is recorded for context.
- **Theory / assumptions:** photon flux in a 10 nm band, using the ASTM G173-03 direct irradiance (via `pvlib`) and the solar disc of radius-based solid angle ($960''$):

$$
N=\Delta\lambda\,E(500\,\mathrm{nm})\,\frac{\Omega_{FoV}}{\Omega_{\odot}}\,\frac{\lambda}{hc},\qquad \Omega_{FoV}=\mathrm{FoV}^2,\ \ \Omega_\odot=\pi(960'')^2 .
$$

- **Tolerance:** `b5_flux`: `flux_rtol` = $10^{-9}$ (same formula and table), `frame_atol` = 0.006 of peak (measured 0.0025; PSF wings beyond $\pm300$ px, resampling), `star_flux_rtol` = 0.002 (measured $<10^{-4}$).
- **References:** ASTM G173-03 reference solar spectra (as named in the code).

## `test_b5_default_science_photon_budget`
- **Module:** `ScienceCam` default arguments (`lightRatio`), `LightPath`.
- **Tier:** nightly.
- **What it checks:** with default `ScienceCam` arguments the noise-free long-exposure frame holds flux $\times T\times A_{eff}$ photons (default `lightRatio` = 1.0).
- **Theory / assumptions:** none beyond photon budget.
- **Tolerance:** `rtol` = $10^{-6}$ (hard-coded).
- **References:** none.

## `test_band_without_source_entry`
- **Module:** `ExtendedSource.__init__` (band table).
- **Tier:** nightly.
- **What it checks:** for bands 'IR', 'V', 'R' (including 'IR', absent from the `Source` table) the extended source constructs and its sub-direction sources carry the wavelength of the extended source.
- **Theory / assumptions:** convention (own band table; the wavelengths of the solar images are kept).
- **Tolerance:** exact equality of the wavelength set.
- **References:** none.
