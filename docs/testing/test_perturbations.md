# `test_perturbations.py` — Vibration, LocalSeeing, spider arms, NCPA, logging at exit (T2)

Source: `tests/test_perturbations.py`

Covers `SAOS.Vibration`, `SAOS.LocalSeeing`, `Telescope.apply_spiders`, `SAOS.NCPA`, `SAOS.LightPath` (how the perturbations enter the WFS and science paths), `SAOS.ScienceCam`, `SAOS.ShackHartmann` (only as a witness that NCPA does not reach it), `SAOS.LoggingHelper`, and the `Source` constructor for laser guide stars. Expectations are independent of SAOS: a sky tilt $\theta$ moves a diffraction-limited image by $\theta/\mathrm{plate\_scale}$ pixels; injected OPD equals the file; the area removed by spiders equals an exact strip-annulus integral; spider diffraction spikes are perpendicular to the arms, checked against an FFT of an independently built mask; logging is checked in subprocesses. Whole module `nightly`. Tolerances: `tests/tolerances/perturbations.yaml` (provisional, calibrated 2026-10-26).

Helper estimator used for image shifts: sub-pixel peak position by 16x (or 8x) Fourier upsampling plus parabolic refinement, valid for band-limited PSFs.

## `test_vibration_one_step_tilt_moves_image_by_file_angle`
- **Module:** `Vibration.getCurrentVibrations` through `LightPath.propagate(temporal_tick=True)` and `ScienceCam`.
- **Tier:** nightly.
- **What it checks:** with no atmosphere and no AO, iteration $k$ injects $(x_k, y_k)$ arcsec from an HDF5 file; the ScienceCam peak (64 px, 4 px per $\lambda/D$, 1 m telescope, R band) moves, relative to the zero-vibration frame, by $x_k/\mathrm{plate\_scale}$ px towards +col and $y_k/\mathrm{plate\_scale}$ towards +row. The fitted slope measured/file is recorded and must be 1.00.
- **Theory / assumptions:** a small tilt $\theta$ of the wavefront displaces the PSF by $\theta/p$ pixels ($p$ = plate scale), shift-invariance of the diffraction-limited image.
- **Tolerance:** `perturbations.vibration`: `shift_rtol` 0.005 and `shift_atol_px` 0.01 px (absolute floor, as zero shift is exactly zero), set by the accuracy of the peak estimator on 1 to 2 px shifts.
- **References:** none specific.

## `test_vibration_wraps_with_different_length_series`
- **Module:** `Vibration.getCurrentVibrations` (indexing).
- **Tier:** nightly. Parametrised over $(L_x, L_y)$ = (5, 8) `x_shorter`, (8, 5) `y_shorter`, (6, 6) `equal`.
- **What it checks:** with series of different length iteration $k$ must give $(x[k \bmod L], y[k \bmod L])$ with $L=\min(L_x,L_y)$ and never raise. The tilts fitted by least squares to the injected OPD over the pupil coordinates are compared with the file sample (scale calibrated at iteration 0), over $2\max(L_x,L_y)+1$ iterations. Convention/indexing check.
- **Tolerance:** `perturbations.vibration.index_rtol` = $10^{-9}$ on the maximum error relative to the largest file value (float64 arithmetic).
- **References:** none.

## `test_vibration_psd_equals_file_psd`
- **Module:** `Vibration` + `LightPath` + `ScienceCam`.
- **Tier:** nightly.
- **What it checks:** a 1024-step series at 1 kHz (three tones at 37, 112, 251 Hz plus AR(1) noise, rms 0.3 px in $x$ and 0.2 px in $y$ delayed by 100 samples) is injected; the camera peak series (8x upsampling) has the file periodogram: band-integrated power in 8 equal bands up to 500 Hz, and rms, ratio to the file equal 1.
- **Theory / assumptions:** since the image-motion to peak map is linear and exact to a factor $1/p$, the camera series is the file series up to scale, and so is its PSD (periodogram $|\mathrm{FFT}|^2$ with no windowing, same samples).
- **Tolerance:** `perturbations.vibration.psd_rtol` = 0.01 (estimator error of a linear map).
- **References:** none specific.

## `test_localseeing_opd_equals_file_with_wraparound`
- **Module:** `LocalSeeing`, `LightPath.localSeeing_opd`, `LightPath.sci_opd`.
- **Tier:** nightly.
- **What it checks:** for a random OPD cube (32x32x5 frames, rms 0.1 um), the OPD injected at iteration $k$ is frame $k \bmod T$ (including beyond $T$, over $2T+3$ iterations) both as `localSeeing_opd` and in the science OPD (file times pupil; no atmosphere, DM or NCPA).
- **Tolerance:** `perturbations.localseeing.exact` = 0 (bit equality, since the OPD is copied).
- **References:** none.

## `test_localseeing_resolution_mismatch_is_bilinear_resampling`
- **Module:** `LocalSeeing` (resampling of a file at a different resolution).
- **Tier:** nightly.
- **What it checks:** a 64x64 file used with a 32 px pupil is resampled to the 2x2 block mean times the pupil: halving a grid by bilinear interpolation at pixel centres is the 2x2 mean.
- **Theory / assumptions:** pixel centres of the coarse grid lie at the centre of 2x2 blocks of the fine grid, so linear interpolation averages the four neighbours with equal weights.
- **Tolerance:** `perturbations.localseeing.resize_rtol` = $10^{-6}$ (relative to the maximum; cv2 float32 arithmetic).
- **References:** none.

## `test_localseeing_input_not_modified_by_interaction_matrix_propagate`
- **Module:** `LocalSeeing.localSeeing_input`, `LightPath.propagate(interaction_matrix=True)`.
- **Tier:** nightly.
- **What it checks:** the user's OPD sequence is data: after a normal step and an interaction-matrix step (which suppresses the path disturbances) the stored input is unchanged and still equals the file, and the next normal step injects file frame 0. The interaction-matrix propagate does not zero or overwrite the user's array in place.
- **Tolerance:** exact (0).
- **References:** none.

## `test_spider_removed_area`
- **Module:** `Telescope.apply_spiders`.
- **Tier:** nightly. Parametrised over arm-angle sets `cardinal` [0, 90, 180, 270], `diagonal` [45, 135, 225, 315], `non_cardinal` [20, 140, 260], `rotated` [17, 107, 197, 287], `near_cardinal` [357, 87, 177, 267] (deg); $N$ = 256 px, $D$ = 8 m, obstruction 0.3, arm width $w$ = 0.8 m (25.6 px).
- **What it checks:** the number of pixels removed from the pupil by the arms against the exact area of the half-strips inside the annulus; also records the count of an independent mask at zero offset.
- **Theory / assumptions:** the area of one half-strip $|t|<w/2$, $s>0$ inside $R_{co}\le r\le R$ is $A(R)-A(R_{co})$ with

$$A(r) = \tfrac{w}{2}\sqrt{r^2-\tfrac{w^2}{4}} + r^2\arcsin\!\frac{w}{2r},$$

  so $n$ arms remove $n[A(R)-A(R_{co})]/p^2$ pixels ($p=D/N$), approximately $n\,w\,(R-R_{co})/p^2$. Arms aligned with the lattice are quantised deterministically for any implementation centred on the array.
- **Tolerance:** absolute, the larger of $3.5\sigma$ (`perturbations.spider.sigma_mult`) of the pixel count of the independent mask over 64 random sub-pixel offsets (`n_lattice`) and the deviation of that mask's count at zero offset from the exact area.
- **References:** none.

## `test_spider_arm_angles`
- **Module:** `Telescope.apply_spiders`.
- **Tier:** nightly. Parametrised over the same five arm-angle sets.
- **What it checks:** that the number of arms found equals the requested number, and that the direction of each removed region (principal axis of its connected component, pointing away from the centre, counter-clockwise from +x with y = +row) matches the requested angles; the estimator is first validated on the independent mask.
- **Tolerance:** `perturbations.spider.angle_deg` = 1.0 deg (the estimator measured less than 0.2 deg on the independent mask; margin for the grid).
- **References:** none.

## `test_spider_single_arm_is_one_half_strip`
- **Module:** `Telescope.apply_spiders`.
- **Tier:** nightly. Parametrised over `angle` = 30, 0, 357 deg.
- **What it checks:** one arm removes a single connected half-strip beyond the central obstruction (count of connected components = 1). An arm whose direction lies just below a multiple of 90 deg (357 deg) is also a single piece.
- **Tolerance:** exact (one component).
- **References:** none.

## `test_spider_psf_spikes_perpendicular_to_arms`
- **Module:** `Telescope.apply_spiders`, `ScienceCam.compute_psf`.
- **Tier:** nightly.
- **What it checks:** for two perpendicular arms at 20 and 110 deg ($N$ = 128, $D$ = 4 m, obstruction 0.1, $w$ = 0.1 m, 4 px per $\lambda/D$) the azimuthal profile of the SAOS PSF over the annulus 2.5 to 7 $\lambda/D$ (normalised to unit mean, 3 deg smoothing) is compared with that of the independent FFT of an independently built mask. The rms profile difference must be within the pixelisation floor, estimated by the same comparison between lattices displaced by 32 random sub-pixel offsets. The mirrored-arm case and the spike azimuths are recorded only.
- **Theory / assumptions:** by Fourier optics the PSF is $|\mathcal F\{P\}|^2$; a thin strip of width $w$ produces a sinc-like spike perpendicular to its length, on both sides of the centre, so the two arms give four spikes at 20, 110, 200 and 290 deg.
- **Tolerance:** rms difference $\le \overline{d}_{floor} + 3.5\,\sigma_{floor}$ (`perturbations.spider.sigma_mult`).
- **References:** Hardy 1998 (Fourier optics of pupils with struts, generic).

## `test_ncpa_load_same_shape_round_trip`
- **Module:** `NCPA.load`, `NCPA.getPhase`.
- **Tier:** nightly.
- **What it checks:** a 32x32 OPD file with the pupil shape is returned by `getPhase` after `load` (which returns `True`).
- **Tolerance:** `perturbations.ncpa.float32_rtol` = $10^{-6}$ of the maximum (the NCPA buffer is float32).
- **References:** none.

## `test_ncpa_load_resized_file_is_bilinear_resampling`
- **Module:** `NCPA.load`.
- **Tier:** nightly.
- **What it checks:** a 64x64 file used with a 32 px pupil is resampled to the 2x2 block mean (same reasoning as for LocalSeeing).
- **Tolerance:** `perturbations.ncpa.resize_rtol` = $10^{-6}$ of the maximum.
- **References:** none.

## `test_ncpa_enters_science_path_not_wfs_path`
- **Module:** `NCPA.setOPD`, `LightPath.propagate` (`sci_opd`, `wfs_opd`, `slopes_1D`), `ShackHartmann`, `ScienceCam`.
- **Tier:** nightly.
- **What it checks:** a tilt NCPA of 0.05 arcsec, per the LightPath documentation $\mathrm{wfs\_opd}=\mathrm{atm}+\mathrm{vib}+\mathrm{LS}+\mathrm{DM}$ and $\mathrm{sci\_opd}=\mathrm{wfs\_opd}+\mathrm{NCPA}$: (i) moves the science image by $\theta/p$ px along +col (and 0 along rows, within 0.01 px), (ii) leaves a 4x4 Shack-Hartmann slope vector bit-equal to the case without NCPA, (iii) gives $\mathrm{sci\_opd}=(\mathrm{wfs\_opd}+\mathrm{NCPA})\cdot\mathrm{pupil}$ exactly.
- **Tolerance:** `perturbations.ncpa.shift_rtol` = 0.005 for the shift; `perturbations.ncpa.exact` = 0 for the slopes and OPD identity; 0.01 px (hard-coded) for the orthogonal shift.
- **References:** none.

## `test_exit_after_exception_with_logger_none`
- **Module:** `Telescope`, `Source`, `LightPath`, `LoggingHelper` with `logger=None` (shared logging queue and its thread at interpreter exit).
- **Tier:** nightly. Parametrised over `name` = `telescope`, `lightpath_source`, `logginghelper`.
- **What it checks:** a subprocess that creates the objects and then raises `RuntimeError` exits by itself with status 1 and prints the traceback within the timeout (a hang is killed and counts as failure).
- **Tolerance:** `perturbations.logging.exit_timeout_s` = 10 s.
- **References:** none.

## `test_logger_none_prints_each_message_once`
- **Module:** `LoggingHelper` with several objects using `logger=None`.
- **Tier:** nightly.
- **What it checks:** in a subprocess with a `Telescope`, `Source` and `LightPath` created with `logger=None`, a warning is printed exactly once, the process exits by itself with status 0, and no "Exception ignored in" message appears from `__del__` at exit.
- **Tolerance:** exact counts; timeout 10 s (`perturbations.logging.exit_timeout_s`).
- **References:** none.

## `test_lgs_source_not_implemented`
- **Module:** `Source.__init__`.
- **Tier:** nightly. Parametrised over `altitude` = 90 km, `Na_profile`, `FWHM_spot_up` = 1.0, `laser_coordinates` = [1.0, 0.0].
- **What it checks:** any laser-guide-star parameter raises `NotImplementedError` (message matching "Laser guide stars"), never a silent natural guide star. Laser guide stars are not implemented.
- **Tolerance:** exception raised (no numerical criterion).
- **References:** none.
