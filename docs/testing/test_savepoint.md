# `test_savepoint.py` — Saved metrics and HDF5 output

Source: `tests/test_savepoint.py`

Covers `SAOS.Savepoint` (`compute_stats`, `save`, `add_metadata`, `module_metadata`) and the HDF5 files it writes. `Savepoint.compute_stats` is fed synthetic data with a known answer: OPD maps, slope vectors, PSFs computed in the test by an independent zero-padded FFT (`theory.psf_reference`) and small solar-like images. The HDF5 round trip and the saving cadence use a small `LightPath` (32 px telescope, `ScienceCam` at 2 px per $\lambda/D$ with `n_pix = res`, so no pupil resampling) whose atmosphere is a stub returning prescribed band-limited OPDs; files go to pytest's `tmp_path`. Many checks are exact comparisons (rtol $10^{-12}$ or equality) because they verify bookkeeping and statistics, not physics. All tests are marked `fast` (module-level). Tolerances: `tolerances/savepoint.yaml` (calibrated 2026-10-26, develop a243eb4, CPU, fixed seeds).

Strehl definition used as reference in the Strehl tests: for a high-order phase without tilt the on-axis ratio equals the coherent energy of the pupil phase,

$$
S=\left|\langle e^{i\phi}\rangle_{P}\right|^2,
$$

(exact for the sampled phase); the Maréchal approximation $S\approx e^{-\sigma_\phi^2}$ is not used by the tests.

## `test_opd_statistics`
- **Module:** `Savepoint.compute_stats` with data type `'opd'`.
- **Tier:** fast.
- **What it checks:** min, max, mean, std (piston-removed rms, the WFE) and rms of an OPD map are computed over the pupil pixels only. The test map is an asymmetric ramp plus offset plus a Gaussian bump in one quadrant with large values (1e-3) outside the pupil, so any statistic taken outside the mask is wrong.
- **Theory / assumptions:** none (direct computation over the pupil mask).
- **Tolerance:** ratio to direct numpy value equal to 1 with `rtol` = $10^{-12}$ (hard-coded).
- **References:** none.

## `test_slopes_1d_statistics_split_xy`
- **Module:** `Savepoint.compute_stats` with `'slopes_1D'`.
- **Tier:** fast.
- **What it checks:** for $\mathrm{slopes\_1D}=[x\ (n),\ y\ (n)]$ the min, max, mean, std and rms are reported per axis as `[[x, y]]`, equal to numpy statistics of each half (40 slopes per axis with different means 3 and -1).
- **Theory / assumptions:** layout convention of the slope vector.
- **Tolerance:** `rtol` = $10^{-12}$.
- **References:** none.

## `test_slopes_2d_statistics_split_xy`
- **Module:** `Savepoint.compute_stats` with `'slopes_2D'`.
- **Tier:** fast.
- **What it checks:** for a 2D map $[x\text{ block }(n\text{ rows});\ y\text{ block }(n\text{ rows})]$ as saved by `LightPath`, min and max per axis are correct (Savepoint expands 2D data to 3D).
- **Theory / assumptions:** layout convention.
- **Tolerance:** `rtol` = $10^{-12}$.
- **References:** none.

## `test_strehl_known_phase`
- **Module:** `Savepoint.compute_stats` with `'sci_frame'` (`strehl`).
- **Tier:** fast.
- **What it checks:** for 9 PSFs (rms 0.3, 0.6, 0.9 rad x 3 realisations, SR 0.45-0.92) of a high-order phase (PSD $f^{-11/3}$ above 4 cycles/$D$, no tilt), computed by independent FFT at Nyquist (2 px per $\lambda/D$) in a wide field (32 $\lambda/D$), the saved Strehl equals $|\langle e^{i\phi}\rangle|^2$ over the pupil.
- **Theory / assumptions:** exact Strehl of the sampled phase; no tilt so that the peak pixel is at the centre; the field edge plays no role.
- **Tolerance:** `strehl.known_phase_rtol` = 0.03 (peak-pixel SR at Nyquist sampling).
- **References:** none specific.

## `test_strehl_subpixel_tilt`
- **Module:** `Savepoint.compute_stats` (`strehl`, peak search).
- **Tier:** fast.
- **What it checks:** a pure tilt does not change the Strehl: a diffraction-limited PSF shifted by 0.25 and 0.5 px in both axes at Nyquist sampling scores SR = 1.
- **Theory / assumptions:** tilt does not change the PSF shape; the sub-pixel peak must be found by interpolation.
- **Tolerance:** `strehl.tilt_rtol` = 0.02.
- **References:** none.

## `test_strehl_in_field_convention`
- **Module:** `Savepoint.compute_stats` (`strehl`).
- **Tier:** fast.
- **What it checks:** documents the convention (owner's decision 2026-10-27) that the saved Strehl is the measured SR in the camera field. For a 1 rad rms high-order phase and fields of 32 $\lambda/D$ and 6 $\lambda/D$ the saved value must equal $SR_{peak}\,f_0/f$. Independent terms: $SR_{peak}$ is the maximum of the aberrated PSF sampled 16x finer (unit ideal peak); $f/f_0$ from the independent frames' sums. The exact $|\langle e^{i\phi}\rangle|^2$ is recorded.
- **Theory / assumptions:**

$$
SR=\frac{I_{peak}/E_{field}}{I^{ideal}_{peak}/E^{ideal}_{field}}=SR_{peak}\,\frac{f_0}{f},
$$

with $f$ and $f_0$ the fractions of the total energy inside the field for the aberrated and ideal PSF. Frames carry unit energy in the field, hence the explicit ratio of sums.
- **Tolerance:** `strehl.convention_rtol` = 0.002 (1/32 px interpolation grid of the sub-pixel peak vs the 16x oversampled reference peak).
- **References:** none.

## `test_contrast_and_mfgs`
- **Module:** `Savepoint.compute_stats` for solar frames (`contrast`, `MFGS`).
- **Tier:** fast.
- **What it checks:** contrast $=\sigma/\mu$: $1+a\sin(2\pi m x/N)$ over whole periods gives $a/\sqrt2$; a constant image gives 0. MFGS (mean fraction of gradient similarity) of a deterministic $12\times12$ texture (smooth waves, a hot pixel, a pseudo-random pattern at amplitudes 0, 0.05, 0.3) equals the independent loop implementation `mfgs_reference`, and both contrast and MFGS are invariant to image scale (scales 1, $10^{-6}$, $10^6$, as frames are saved in photons or DN). Regression: the median-filter border is extended symmetrically (zero padding biased border medians low, SAV-MFGS-BORDER).
- **Theory / assumptions:** MFGS definition as implemented in the reference: image divided by its mean; $M$ = $3\times3$ median filter (symmetric border); Sobel-like gradient with kernel $[[-3,0,3],[-10,0,10],[-3,0,3]]$ (Scharr) and its transpose, magnitude $g=\sqrt{g_x^2+g_y^2}$;

$$
\mathrm{MFGS}=\Big\langle\frac{2\,g_I\,g_M+C}{g_I^2+g_M^2+C}\Big\rangle_{\rm pixels},\quad C=10^{-10}.
$$

- **Tolerance:** contrast $a/\sqrt2$ and constant image: `rtol` = $10^{-12}$ / `atol` = $10^{-12}$; MFGS and contrast vs reference: `rtol` = $10^{-9}$ (hard-coded).
- **References:** none specific (the metric is the one used by the owner's solar-image code; no citation in the test).

## `test_mfgs_constant_image`
- **Module:** `Savepoint.compute_stats` (`MFGS`).
- **Tier:** fast.
- **What it checks:** an image without structure has no gradients, so every pixel is "similar" and MFGS = 1 (for constants 3.7 and $10^5$).
- **Theory / assumptions:** $C$ regularises $0/0$ in the similarity formula.
- **Tolerance:** `atol` = $10^{-12}$.
- **References:** none.

## `test_hdf5_roundtrip`
- **Module:** `Savepoint.save` through `LightPath.propagate`; HDF5 groups `atm_opd`, `sci_opd`, `sci_frame_shortExp`.
- **Tier:** fast.
- **What it checks:** over 6 iterations, the saved data equal what was physically injected: atmosphere OPD per direction (the stub OPD, iteration index), science OPD (equal to the OPD on the pupil mask and zero outside), std over the pupil, 6 short-exposure frames equal to the independent FFT frame (unit energy in the field), and the saved Strehl per frame equal to the exact $|\langle e^{2\pi i\,\mathrm{OPD}/\lambda}\rangle|^2$. The OPDs are band-limited to 4-7 cycles/$D$ with 0.6 rad rms so the halo stays inside the $\pm8\lambda/D$ field.
- **Theory / assumptions:** exact Strehl of the sampled phase; frames from the independent Fraunhofer FFT.
- **Tolerance:** OPD maps $\le10^{-15}$ m; std `rtol` = $10^{-9}$; `frame_atol` = $10^{-6}$ of the peak (same computation, float precision); `strehl.known_phase_rtol` = 0.03.
- **References:** none.

## `test_saved_frames_match_decimation`
- **Module:** `Savepoint.save`, `ScienceCam(decimation)`, `LightPath`.
- **Tier:** fast.
- **What it checks:** with the camera decimated by 3 and frames saved every 3 iterations, the 4 frames taken in 12 iterations (at iterations 2, 5, 8, 11, i.e. $(i+1)\%3==0$, the Savepoint convention) are saved with the iteration index of the OPD they come from and the content of the independent FFT frame of that OPD.
- **Theory / assumptions:** cadence convention; frames from independent FFT.
- **Tolerance:** `frame_atol` = $10^{-6}$ of the peak; count and indices exact.
- **References:** none.

## `test_metadata_provenance_and_config`
- **Module:** `Savepoint` metadata (`/metadata` group, `add_metadata`, light-path group attributes).
- **Tier:** fast.
- **What it checks:** `/metadata` holds the SAOS version (equal to `SAOS.__version__`), provenance (Python version, host, command line, creation and last-saved times, Savepoint settings JSON), `last_iteration` updated at each save, user metadata from the constructor and `add_metadata()` (lists, strings, dicts as JSON), the telescope configuration (D, resolution, spiders 'none'), and light-path attributes (source wavelength, delay, science camera plate scale).
- **Theory / assumptions:** I/O and bookkeeping only.
- **Tolerance:** exact equality.
- **References:** none.

## `test_metadata_large_values_as_datasets`
- **Module:** `Savepoint` metadata storage.
- **Tier:** fast.
- **What it checks:** user metadata above the attribute size limit (a 10000-element float array, 80 kB, above the 64 kB HDF5 object-header limit) is stored as a dataset of `/metadata/user` without failure, and small values stay attributes.
- **Theory / assumptions:** HDF5 attribute-size limit (I/O only).
- **Tolerance:** exact.
- **References:** HDF5 documentation (attribute size limit), implicit.

## `test_metadata_of_modules`
- **Module:** `SAOS.Savepoint.module_metadata` for `Telescope`, `Atmosphere`, `DeformableMirror`.
- **Tier:** fast.
- **What it checks:** the returned dictionaries carry the parameters the modules were created with: telescope spider angles and thickness (and 'none' after `set_pupil`), atmosphere $r_0$, $L_0$, altitudes, wind speeds, zenith in degrees, DM `nActs`, `nValidAct`, altitude and misregistration entries; the dictionary is JSON-serialisable.
- **Theory / assumptions:** none (metadata round trip).
- **Tolerance:** exact (zenith `rtol` = $10^{-12}$).
- **References:** none.
