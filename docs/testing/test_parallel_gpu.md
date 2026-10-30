# `test_parallel_gpu.py` — Parallel (threads) and GPU execution

Source: `tests/test_parallel_gpu.py`

Covers the execution-mode independence of SAOS: joblib threaded `LightPath.propagate` (sharing `Atmosphere`, `Telescope`, `DeformableMirror`, `ShackHartmann`, `ScienceCam`), CPU versus CUDA execution of `ShackHartmann`, `ScienceCam`, `DeformableMirror`, `InteractionMatrixHandler` and `Controller`, thread counts and the VRAM peak. By documented policy these are the only tests whose reference is SAOS itself ("equivalence tests"): the result of a configuration must not depend on how it is executed. There is no theory; numerical agreement is set by the arithmetic precision of each computation (float64 for DM, ScienceCam, Controller; float32 inside the ShackHartmann field cube and centroid). `Controller` and `InteractionMatrixHandler` hard-code "cuda if available", so the CPU version is built with `torch.cuda.is_available` patched to False during construction. Module-level marker is `nightly`; tests that need CUDA additionally carry `gpu` (skipped when no GPU is visible). Tolerances: `tolerances/parallel_gpu.yaml` (provisional, calibrated 2026-10-26, GTX 1660).

The `Multi` fixture builds a telescope (48 px, $D=0.6$ m), a 3-layer atmosphere (winds 8/15/25 m/s, directions 0/70/200 deg, altitudes 0/4/11 km), a shared ground DM ($7\times7$) and $n$ light paths (one source direction, one WFS and one ScienceCam each), rebuilt with identical seeds for each execution mode.

## `test_threads_bit_identical_to_sequential_noise_off`
- **Module:** `LightPath.propagate` under `joblib.Parallel(prefer='threads')`.
- **Tier:** nightly (no `gpu` marker; runs on CPU).
- **What it checks:** six light paths sharing atmosphere, telescope and one DM, each with its own WFS and science camera, are propagated for 3 steps with `n_jobs` = 1, 2, 8 (noise off). OPD, slopes, WFS frames and science frames are bit-identical to the sequential run.
- **Theory / assumptions:** the same computation, only the scheduling differs; shared objects must be thread-safe.
- **Tolerance:** `threads.exact` = 0.0 (bit equality). The maximum slope magnitude is recorded to show a signal is present.
- **References:** none.

## `test_threads_reproducible_with_per_object_seeds_and_noise`
- **Module:** `LightPath.propagate` with `Detector` noise (`ShackHartmann(noiseFlag=True, randomState)`).
- **Tier:** nightly.
- **What it checks:** with detector noise on and a distinct seed per WFS (seed = 100 + path index), results for `n_jobs` = 1 and 8 are bit-identical, and two runs with `n_jobs` = 8 coincide.
- **Theory / assumptions:** the noise stream of a detector does not depend on scheduling when each detector is used by one light path (a shared RNG would break this; see the next test).
- **Tolerance:** `threads.exact` = 0.0.
- **References:** none.

## `test_threads_reproducible_with_shared_detector`
- **Module:** `LightPath.propagate` with one WFS (one `Detector`, one seed) shared by four light paths.
- **Tier:** nightly.
- **What it checks:** four light paths sharing one WFS are propagated with `n_jobs` = 1 and 8; a reproducible simulator should give the same noisy slopes for the same path regardless of scheduling, and two runs with `n_jobs` = 8 should coincide. The maximum slope difference is recorded.
- **Theory / assumptions:** reproducibility with a shared random state; the test fails if the order of noise draws from the shared Detector random state depends on thread scheduling.
- **Tolerance:** `threads.exact` = 0.0 (no `known_issue` marker in the code).
- **References:** none.

## `test_thread_count_inside_threaded_lightpaths_within_cpu_count`
- **Module:** `Atmosphere.getOPD` for an extended (sun-like) source with nine point directions, called from 8 joblib threads.
- **Tier:** nightly.
- **What it checks:** resource criterion. Eight concurrent `getOPD` calls, each parallelising its 9 directions, are repeated 3 times after a warm-up; the worker threads created (peak minus baseline of `threading.active_count()` and, if `psutil` is available, of OS threads) must not exceed `os.cpu_count()`.
- **Theory / assumptions:** none (no oversubscription of threads).
- **Tolerance:** `threads_count.max_threads` = 0, meaning `os.cpu_count()` is used as limit.
- **References:** none.

## `test_gpu_matches_cpu_shackhartmann`
- **Module:** `ShackHartmann.wfs_measure` (`device='cpu'` vs `'cuda'`).
- **Tier:** nightly + gpu.
- **What it checks:** the same Kolmogorov-like phase (1 rad rms, masked by the pupil) through a CPU and a CUDA `ShackHartmann` (6x6, 48 px, noise off) gives the same slopes (absolute, px) and the same frame (relative to its peak).
- **Theory / assumptions:** the cube of subaperture fields is complex64 and the centroid float32, so agreement is at float32 precision.
- **Tolerance:** `cpu_gpu.shackhartmann_slope_px` = $10^{-5}$ px (measured 1.9e-6), `cpu_gpu.shackhartmann_frame_rel` = $10^{-6}$ of the peak (measured 2e-7).
- **References:** none.

## `test_gpu_matches_cpu_sciencecam`
- **Module:** `ScienceCam.get_frame` (`device='cpu'` vs `'cuda'`).
- **Tier:** nightly + gpu.
- **What it checks:** a 2 rad rms smooth phase on a 64 px pupil gives the same science frame on CPU and GPU, relative to the peak.
- **Theory / assumptions:** complex128 FFT on both devices; differences only from the order of floating-point operations.
- **Tolerance:** `cpu_gpu.sciencecam_rel` = $10^{-10}$ of the peak.
- **References:** none.

## `test_gpu_matches_cpu_deformablemirror`
- **Module:** `DeformableMirror.updateDMShape`, `get_dm_opd` (`device='cpu'` vs `'cuda'`).
- **Tier:** nightly + gpu.
- **What it checks:** the OPD of a random command ($10^{-7}$ m scale, $9\times9$ DM) for a ground layer and a 4 km layer (footprint shift for a source at 15", 30 deg) agrees between CPU and GPU, relative to the maximum.
- **Theory / assumptions:** float64 influence-matrix product; only summation ordering differs.
- **Tolerance:** `cpu_gpu.dm_rel` = $10^{-12}$ relative to the maximum.
- **References:** none.

## `test_gpu_matches_cpu_interaction_matrix_and_controller`
- **Module:** `InteractionMatrixHandler.measure`, `Controller.computeControlAction` (CPU vs CUDA).
- **Tier:** nightly + gpu.
- **What it checks:** (i) zernike IMs measured on a CPU-built system (with `torch.cuda.is_available` patched to False) and on a GPU system agree: the slope-level difference $\max|\mathrm{IM}_{cpu}-\mathrm{IM}_{gpu}|\times\mathrm{stroke}$ stays within float32 precision. (ii) Controllers (inversion, leaky gain 0.4 decay 0.99, 12 modes) built on the same IM, one per device and verified to be on `cpu` and `cuda`, give identical modal and DM commands over 10 random slope vectors.
- **Theory / assumptions:** WFS slopes are float32 (hence the IM tolerance); the controller is float64 (matrix products, Cholesky/SVD).
- **Tolerance:** `cpu_gpu.im_slope_px` = $10^{-5}$ px (measured 4.3e-6), `cpu_gpu.controller_rel` = $10^{-10}$ relative to the maximum.
- **References:** none.

## `test_vram_peak_within_budget`
- **Module:** whole GPU path (`ShackHartmann`, `ScienceCam`, `DeformableMirror`, `Atmosphere`, `LightPath`) on CUDA.
- **Tier:** nightly + gpu.
- **What it checks:** for a fixed small configuration (64 px pupil, $8\times8$ SH with 2x oversampled 16 px windows, $9\times9$ DM, 64 px science camera, 5 steps) the peak `torch.cuda.max_memory_allocated` above the baseline is (i) at least the analytic size of the dominant tensor, the complex64 field cube of the SH, and (ii) below the recorded budget times a headroom (regression guard against a leak or per-step copy). The final allocated memory is recorded.
- **Theory / assumptions:** analytic size of the SH field cube,

$$
V_{cube}=n_{subap}^2\,n_{FFT}^2\times 8\ \text{B},\qquad n_{FFT}=\mathrm{round}\!\Big(\frac{\lambda\cdot206265}{d\cdot\mathrm{plate\_scale}}\,n_{lenslet}\Big),
$$

(8 bytes per complex64 element); the peak must be at least this.
- **Tolerance:** `vram.budget_mib` = 8.7 MiB (re-recorded 2026-10-27 after the WFS change that samples the pupil at `npix_pupil > npix_lenslet` to remove spot aliasing, #54; peak was 8.62 MiB, previously 6.69 MiB, GTX 1660) times `vram.headroom` = 1.25.
- **References:** none.
