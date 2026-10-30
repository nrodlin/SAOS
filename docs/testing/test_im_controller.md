# `test_im_controller.py` — Interaction matrix and control laws

Source: `tests/test_im_controller.py`

Covers `SAOS.InteractionMatrixHandler` (IM measurement, bookkeeping, save/load, modal-basis convention) and `SAOS.Controller` (reconstructors: pseudo-inverse, Tikhonov, joint/per-DM, tomography; control laws: leaky, forward/backward PI; POLC). Setup: 48 px square pupil, $D=0.6$ m, $6\times6$ Shack-Hartmann (all window pixels used so the centroid is odd and continuous), $7\times7$ DM (Gaussian influence functions, coupling 0.6), Zernike basis, CPU, stroke $2\times10^{-8}$ m, `NM` = 12 modes for the controller tests. Independent references (no SAOS code or outputs): `dm_surface` re-implements the documented DM (Gaussian influence functions with partition of unity); `sh_slopes` propagates a surface over each physical subaperture by a zero-padded Fourier transform sampled at 2 px per $\lambda/d$ and takes the centre of gravity over the 15 px window; the reconstructor and control-law references are numpy SVD, Tikhonov normal equations and difference equations (`scipy.signal.dlsim`) written from the documented formulas. Tests that compare SAOS with another SAOS path are named `test_regression_*` and are explicitly not physics checks. The controller is driven with a linear modal plant $s(k)=\mathrm{IM}\,[a(k)+c(k-D)]$ where $a$ is the disturbance, $c$ the modal command and $D$ the total pure delay (mock light path returning the slopes), so the closed loop is the textbook loop. All tests are marked `fast` (module-level). Tolerances: `tolerances/im_controller.yaml`.

Control-law and loop model:

$$
c=C(z)\,e,\quad e=-R\,s,\quad u=a+z^{-D}c\ \Rightarrow\ \frac{u}{a}=S(z)=\frac{1}{1+C(z)\,z^{-D}},
$$

$$
C_{\rm leaky}=\frac{g}{1-d\,z^{-1}},\quad C_{\rm fwdPI}=\frac{g+(k_i-g)z^{-1}}{1-z^{-1}},\quad C_{\rm bwdPI}=\frac{g+k_i-g\,z^{-1}}{1-z^{-1}} .
$$

Reconstructors used as references (IM $H$, SVD $H=U\Sigma V^T$): minimum-norm pseudo-inverse with singular values below $r_{cond}\,s_{max}$ discarded, $R=V_r\Sigma_r^{-1}U_r^T$; Tikhonov $R=(H^TH+\alpha I)^{-1}H^T=V\,\mathrm{diag}\!\big(s/(s^2+\alpha)\big)U^T$ with $\alpha=\beta\,s_{max}^2$; joint Tikhonov $R=(H^TH+\mathrm{diag}(\alpha_i))^{-1}H^T$ with $\alpha_i=\beta_i s_{max,i}^2$ of the columns of DM $i$.

## `test_im_columns_vs_independent_forward_model`
- **Module:** `InteractionMatrixHandler.measure` (Zernike modal IM), with `DeformableMirror` and `ShackHartmann`.
- **Tier:** fast.
- **What it checks:** the absolute IM (px of slope per metre of mode amplitude) of the first 12 Zernike modes equals the independent forward model (documented DM surface propagated through a physical SH to the spot centroid). All $2\times36$ slopes of every mode are compared in absolute terms, error normalised to the largest expected entry of the same mode. The least-squares gain SAOS/expected is recorded as information.
- **Theory / assumptions:** DM surface $w(\mathbf x)=\sum_k g_k(\mathbf x)c_k/\sum_k g_k(\mathbf x)$ with Gaussian $g_k=\exp[\ln(0.6)\,r^2/p^2]$ ($p$ = actuator pitch, valid actuators with $r\le D/2+0.5p$); the SH slope per subaperture is the centroid of $|\mathcal F\{e^{2\pi i\,\mathrm{OPD}/\lambda}\}|^2$ sampled at $k\lambda/(\mathrm{samp}\,d)$ over the window, minus the flat-wavefront value; a gradient $g$ moves the spot by $g\,\mathrm{samp}\,d/\lambda$ px. Linear regime (spot motion 0.02-0.05 px).
- **Tolerance:** `forward_model.abs_rel` = $3\times10^{-3}$ (precision of the reference: 32 samples per subaperture side differ from 64 by $9\times10^{-4}$, error $\propto 1/m^2$, $4\times10^{-3}$ at 16).
- **References:** none specific.

## `test_im_symmetry_of_symmetric_modes`
- **Module:** `InteractionMatrixHandler.measure` / `ShackHartmann` slope maps.
- **Tier:** fast.
- **What it checks:** parametrised over `mode` = 2 (`defocus`) and 9 (`spherical`). A DM shape with the sensor's symmetry (even under $x\to-x$, $y\to-y$, $x\leftrightarrow y$) must give slope maps with $s_x(r,c)=-s_x(r,5-c)=s_x(5-r,c)=s_y(c,r)$, errors normalised to the largest slope. The independent DM surface symmetry is checked first (to $10^{-10}$).
- **Theory / assumptions:** symmetry of the problem, independent of the sensor scale. The docstring records a diagnosis (scratch experiment, SAOS untouched): an asymmetry of order $5\times10^{-2}$ (x-mirror) found in the sensor's subaperture grid (WFS-SUBAP-GRID, +0.5 px shift of the subaperture split, see `test_wfs.py::test_defocus_subaperture_geometry`), not in the DM or IM; with a symmetric independent split the errors drop to about $2\times10^{-4}$. WFS-SUBAP-GRID was fixed on 2026-10-27 (cell-centred subaperture split): the test passes and is kept as a regression test.
- **Tolerance:** `symmetry.rel` = $10^{-3}$ (float32 FFT round-off); DM surface symmetry $\le10^{-10}$ (hard-coded).
- **References:** none.

## `test_regression_im_column_equals_direct_wfs_response`
- **Module:** `InteractionMatrixHandler.measure` vs `ShackHartmann.wfs_measure` (equal-path regression).
- **Tier:** fast.
- **What it checks:** for the first 12 modes the IM column equals the WFS slopes for the OPD of the same DM shape applied directly, divided by the stroke.
- **Theory / assumptions:** none (regression; SAOS vs SAOS).
- **Tolerance:** `tilt.direct` = $10^{-3}$ relative to the largest entry.
- **References:** none.

## `test_im_linear_in_stroke_and_push_pull`
- **Module:** `InteractionMatrixHandler.measure`.
- **Tier:** fast.
- **What it checks:** IM = slopes/stroke is independent of stroke where the sensor is linear (2e-8 and 4e-8 m, spot motions 0.02-0.05 px) and push ($+$stroke) and pull ($-$stroke) matrices coincide, 12 modes.
- **Theory / assumptions:** linear, odd response of the all-pixel centroid.
- **Tolerance:** `linearity.stroke` = 0.01, `linearity.push_pull` = 0.01 relative to the largest entry.
- **References:** none.

## `test_identical_dms_in_two_lightpaths_both_measured`
- **Module:** `InteractionMatrixHandler.initialize_im_class`, `.measure`, `interaction_matrix_warehouse`.
- **Tier:** fast.
- **What it checks:** two light paths, each with its own DM and WFS (identical properties): two distinct scanned DMs; `warehouse[i][j]` holds an IM only where DM $i$ is seen by light path $j$ (i == j) and is None otherwise; both IMs non-zero and equal to each other.
- **Theory / assumptions:** bookkeeping only; known-issue style regression about DMs with identical properties.
- **Tolerance:** `nonzero_min` = 1000 px/m (a poked DM gives about $10^6$ px/m); equality to `reconstructor.matrix` = $10^{-9}$.
- **References:** none.

## `test_regression_save_load_im_round_trip`
- **Module:** `InteractionMatrixHandler.save_IM`, `.load_IM`.
- **Tier:** fast.
- **What it checks:** saving and loading the IM into a fresh handler of the same optical configuration restores the IM bit by bit, and the modal-basis name ('zonal') is restored.
- **Theory / assumptions:** I/O round trip only.
- **Tolerance:** exact (`hi` = 0).
- **References:** none.

## `test_save_im_refuses_existing_file_and_overwrites_atomically`
- **Module:** `InteractionMatrixHandler.save_IM`, `save_modalBasis`, `check_output_file`.
- **Tier:** fast.
- **What it checks:** with a pre-existing HDF5 file holding unrelated content, `check_output_file` and both save methods raise `FileExistsError` and leave the file byte-identical; with `overwrite=True` the file is replaced (old content gone, `basis_convention` written), no temporary file is left in the folder, and the overwritten IM loads back bit by bit.
- **Theory / assumptions:** none (I/O contract): an existing file is never appended to, and a replacement only happens once the new file is complete (temporary file renamed with `os.replace`).
- **Tolerance:** exact (boolean checks; IM difference 0).
- **References:** none.

## `test_regression_save_load_im_with_science_lightpath_first`
- **Module:** `InteractionMatrixHandler.save_IM`, `.load_IM`.
- **Tier:** fast.
- **What it checks:** a science-only light path (no WFS) listed before the WFS light path does not break the save/load round trip.
- **Theory / assumptions:** I/O and indexing only.
- **Tolerance:** exact.
- **References:** none.

## `test_load_im_identical_dms_and_light_paths`
- **Module:** `InteractionMatrixHandler.save_IM`, `.load_IM`.
- **Tier:** fast.
- **What it checks:** with two light paths with identical DMs, after save and load each IM returns to its own (DM, light path) entry, identified by group indices; the two IMs are made different before saving (one scaled by 2) to tell them apart.
- **Theory / assumptions:** indexing only.
- **Tolerance:** exact.
- **References:** none.

## `test_im_and_modal_basis_convention`
- **Module:** `InteractionMatrixHandler.save_IM`, `.save_modalBasis`, `.load_IM`, `.load_modalBasis`, `BASIS_CONVENTION`.
- **Tier:** fast.
- **What it checks:** IM and modal-basis files carry the `basis_convention` attribute; a legacy IM (attribute removed, convention 1: old sign of the sine Zernikes) cannot be loaded into a handler whose basis uses the current convention (`ValueError`); a legacy IM with the legacy basis loads.
- **Theory / assumptions:** convention bookkeeping (Zernike sine sign) only.
- **Tolerance:** exact.
- **References:** none.

## `test_regression_per_dm_modal_basis_list`
- **Module:** `InteractionMatrixHandler.measure`.
- **Tier:** fast.
- **What it checks:** `measure(['zonal'], ...)` for a one-DM system is accepted and equals `measure('zonal', ...)`.
- **Theory / assumptions:** none (equal-path regression).
- **Tolerance:** $10^{-12}$ relative (hard-coded).
- **References:** none.

## `test_controller_step_response_vs_ztransform`
- **Module:** `Controller.computeControlAction`, reconstructor of `reconstructionMethod='inversion'`; control laws `leaky` and `forwardPI`/`backwardPI`.
- **Tier:** fast.
- **What it checks:** parametrised over (law, gain, decay, $k_i$, $D$) = (leaky, 0.4, 0.99, 0, 2), (forwardPI, 0.1, 1.0, 0.2, 2), (backwardPI, 0.1, 1.0, 0.2, 1) (ids `leaky-D2`, `forwardPI-D2`, `backwardPI-D1`). (a) Open-loop feed (slopes $\mathrm{IM}\,u$, step in mode 0 plus random sequence in the others): the modal command equals `dlsim` of $C(z)$ applied to $e=-u$. (b) Closed loop with pure delay $D$: the residual to a unit step equals `dlsim` of $S(z)=1/(1+C(z)z^{-D})$, other modes stay at zero. Also $R\,\mathrm{IM}=I$ on the controlled modes.
- **Theory / assumptions:** the loop model and transfer functions given in the section introduction; difference equation in float64; the plant is exactly linear.
- **Tolerance:** `controller`: `reconstructor` = $10^{-9}$ on $\max|R\,\mathrm{IM}-I|$ (pinv with `rcond` $10^{-3}$), `difference_eq` = $10^{-12}$, `closed_step` = $10^{-12}$.
- **References:** none specific (standard digital control textbook).

## `test_rejection_transfer_function`
- **Module:** `Controller.computeControlAction` (frequency response of the loop).
- **Tier:** fast.
- **What it checks:** parametrised over (law, gain, decay, $k_i$, $D$) = (leaky, 0.4, 0.99, 0, 2), (leaky, 0.3, 0.995, 0, 3), (forwardPI, 0.1, 1.0, 0.2, 2), (backwardPI, 0.1, 1.0, 0.2, 1). A multisine of 7 injected bins (2, 5, 10, 20, 40, 64, 90 of a 256-frame window) in mode 0 is rejected by the loop; lock-in over an integer number of periods after the transient gives the measured $S(e^{j\omega})$, which must equal the analytic one in magnitude and phase.
- **Theory / assumptions:**

$$
S(e^{j\omega})=\frac{1}{1+C(e^{j\omega})\,e^{-j\omega D}},\quad \omega=2\pi b/N,
$$

with stable poles (the maximum pole modulus is recorded).
- **Tolerance:** `controller.rejection_mag` = $10^{-9}$ (relative magnitude error), `rejection_phase` = $10^{-9}$ rad.
- **References:** none specific.

## `test_single_dm_reconstructor_vs_independent_svd`
- **Module:** `Controller` reconstructor construction (`reconstructionMethod` `inversion` and `tikhonov`, `joint_reconstruction`), `computeControlAction`.
- **Tier:** fast.
- **What it checks:** parametrised over method (inversion, tikhonov) x joint (joint, per-DM) x rank (full-rank, rank-deficient) = 8 cases. For a prescribed IM $H$ ($30\times8$; rank-deficient: last column $=0.7\,c_0-1.3\,c_1$) the reconstructor and the first command of a gain-1 leaky controller (command $=R(-s)$, DM command = basis $\times c$) equal numpy references. For rank-deficient inversion also the row-space projector $RH=V_rV_r^T$, the Penrose condition $HRH=H$, and the absence of any null-space component in the command (minimum norm); rank kept 7.
- **Theory / assumptions:** $R_{inv}=V_r\Sigma_r^{-1}U_r^T$ (singular values $>r_{cond}s_{max}$, $r_{cond}=10^{-3}$), $R_{Tik}=V\,\mathrm{diag}(s/(s^2+\beta s_{max}^2))U^T$ ($\beta=10^{-3}$) $=(H^TH+\alpha I)^{-1}H^T$.
- **Tolerance:** `reconstructor.matrix` = $10^{-9}$ (float64 SVD or normal equations of the same matrix; rank deficiency amplifies round-off to below $10^{-12}$).
- **References:** none specific (Tikhonov regularisation).

## `test_two_dm_coupled_reconstructors_vs_independent_solutions`
- **Module:** `Controller` joint and per-DM reconstruction with several DMs and WFS.
- **Tier:** fast.
- **What it checks:** parametrised over method (inversion, tikhonov) x rank (full-rank, rank-deficient) = 4 cases. Two DMs (6 and 4 modes), two WFS (14 slopes each), both WFS see both DMs. Joint and per-DM reconstructors and commands equal the independent solutions on the stacked matrix; the slope-fit residual equals the independent one; joint inversion equals numpy `lstsq` minimum-norm.
- **Theory / assumptions:** joint inversion $=H^+$ of the stacked $H$, split by DM rows; joint Tikhonov $R=(H^TH+\mathrm{diag}(\beta_i s_{max,i}^2))^{-1}H^T$ ($\beta$ = [1e-3, 3e-3]); per-DM uses only the columns of one DM.
- **Tolerance:** `reconstructor.matrix` = $10^{-9}$.
- **References:** none specific.

## `test_block_diagonal_reconstructors_vs_independent_solutions`
- **Module:** `Controller` reconstruction with block-diagonal IM.
- **Tier:** fast.
- **What it checks:** parametrised over method (inversion, tikhonov). WFS $i$ sees only DM $i$ (blocks $14\times6$ and $14\times4$): the joint reconstructor of DM $i$ is the independent reconstructor of its block on the slopes of WFS $i$ and exactly zero on the other WFS; the per-DM one is that block alone; commands equal the independent ones.
- **Theory / assumptions:** pseudo-inverse and Tikhonov of a block-diagonal matrix are block-diagonal.
- **Tolerance:** `reconstructor.matrix` = $10^{-9}$.
- **References:** none specific.

## `test_coupled_joint_reconstruction_beats_per_dm`
- **Module:** `Controller` (joint vs per-DM reconstruction).
- **Tier:** fast.
- **What it checks:** with both WFS seeing both DMs and consistent noise-free slopes $s=Hc_{true}$ (full-rank $H$), the joint command recovers $c_{true}$ exactly (relative error $\le10^{-9}$), while the per-DM commands have an error above 1 % because they ignore the cross-term (information, makes the comparison meaningful).
- **Theory / assumptions:** $RH=I$ for full column rank $H$ (rcond $10^{-6}$); the per-DM reconstructor $H_i^+$ is not the inverse of the coupled system.
- **Tolerance:** joint error $\le10^{-9}$; per-DM error $\ge0.01$ (hard-coded).
- **References:** none.

## `test_single_mode_shapes`
- **Module:** `Controller` sizing (`nModes`, `im_per_dm`, `reconstructor`).
- **Tier:** fast.
- **What it checks:** with `nModes = 1`, for joint and per-DM reconstruction, `im_per_dm[0]` has shape (nSignal, 1) and `reconstructor[0]` has shape (1, nSignal).
- **Theory / assumptions:** shapes only.
- **Tolerance:** exact.
- **References:** none.

## `test_joint_sizing_with_fewer_im_modes_than_basis`
- **Module:** `Controller` sizing.
- **Tier:** fast.
- **What it checks:** an IM measured with 10 modes of a 37-mode basis gives a controller with 10 modes (reconstructor shape (10, nSignal)).
- **Theory / assumptions:** shapes only.
- **Tolerance:** exact.
- **References:** none.

## `test_polc_reconstructs_the_open_loop_turbulence`
- **Module:** `Controller.computeControlAction` with `operationType='polc'` (Tikhonov, leaky configuration, delay 1).
- **Tier:** fast.
- **What it checks:** POLC reconstructs from the pseudo-open-loop slopes $s_{pol}=s-\mathrm{IM}\,c(k-D)=\mathrm{IM}\,a$ with the filter $c(k)=(1-g)\,c(k-1)+g\,(-R\,s_{pol}(k))$ (unit DC gain; the decay of the configuration is not used), with $g=0.3$. The command must equal that recursion applied to $-R\,\mathrm{IM}\,a$, and must differ from the closed-loop law by more than 5 % of the command scale.
- **Theory / assumptions:** pseudo-open-loop control: the controller reconstructs the open-loop turbulence by adding back the commanded correction, $s_{pol}=s_{res}+D\,c$ (with $D$ here the IM), so the result is independent of the loop; leaky integrator on the POL error.
- **Tolerance:** `polc.match` = $10^{-8}$; `polc.differs` = 0.05.
- **References:** Ellerbroek & Vogel 2003 (POLC), as cited in the docstring.

## `test_polc_slopes_attribute_is_minus_the_turbulence`
- **Module:** `Controller.slopes_polc`.
- **Tier:** fast.
- **What it checks:** after each call `slopes_polc` equals $-s_{turb}=-\mathrm{IM}\,a$ (error-sign convention); the last step is compared (30 frames).
- **Theory / assumptions:** sign convention of the pseudo-open-loop slopes.
- **Tolerance:** `polc.sign` = $10^{-8}$ relative.
- **References:** none.

## `test_polc_per_dm_adds_every_dm_seen_by_the_wfs`
- **Module:** `Controller.computeControlAction` (POLC with per-DM reconstructors, `joint_reconstruction=False`), `Controller._init_polc_cross_ims`.
- **Tier:** fast.
- **What it checks:** one WFS (14 slopes) seen by two DMs with prescribed IMs $B_0$ (14×6) and $B_1$ (14×4), plant $s(k) = t(k) + B_0 c_0(k-1) + B_1 c_1(k-1)$ with a random turbulence sequence $t$, Controller delay 1, 12 steps. The pseudo-open-loop slope vector of each DM must be $-t(k)$.
- **Theory / assumptions:** with the error sign convention $e = -s$, the POLC slopes of the slope vector of DM $i$ (the WFS of its mask row) must remove every DM seen by those WFS:

$$
-s_{pol,i}(k) = -s(k) + \sum_j B_{j\to i}\,c_j(k-d_j) = -t(k),
$$

  so removing only DM $i$ would leave $B_{j}\,c_{j}$ of the other DM (the behaviour before 2026-10-30).
- **Tolerance:** `im_controller.polc.sign` (max relative error over steps and DMs; numerical precision).
- **References:** Ellerbroek & Vogel 2003 (POLC).

## `test_polc_per_dm_missing_cross_im_raises`
- **Module:** `Controller._init_polc_cross_ims`, `Controller.computeControlAction`.
- **Tier:** fast.
- **What it checks:** DM 1 is in the optical path of the WFS used by DM 0 (light path 0) but its IM on that WFS was not measured; DM 1 has its own WFS. The controller records the missing pair `(DM 0 row, DM 1, light path 0)` and `computeControlAction` raises `ValueError` instead of forming incomplete pseudo-open-loop slopes.
- **Theory / assumptions:** none (configuration check).
- **Tolerance:** exact (boolean checks).
- **References:** none.

## `test_tomography_law_vs_stated_equations`
- **Module:** `Controller` with `reconstructionMethod='tomography'` (`R_tomo_path`, `set_tomography_targets`, `joint`).
- **Tier:** fast.
- **What it checks:** parametrised over `joint` (joint, per-DM). One WFS (24 slopes), two DMs (6 and 5 modes), $R_{tomo}$ ($18\times24$), target IMs $D_i$ ($18\times n_i$), leaky gains [0.4, 0.3] and decays [0.98, 0.95], 25 frames of random slopes. The target reconstructors equal the independent ones, and modal and DM commands equal the recursion in numpy at every frame.
- **Theory / assumptions:** learn-and-apply style law: target reconstructor $R=(D^TD+\mathrm{diag}(\beta_i s_{max}(D_i)^2))^{-1}D^T$ with $D=[D_1\ldots D_n]$ (joint) or $R_i=V\,\mathrm{diag}(s/(s^2+\beta_i s_{max}^2))U^T$ of $D_i$ (per-DM); then

$$
e_i(k)=R_i\Big(R_{tomo}(-s(k))-\sum_j D_j\,c_j(k-1)\Big),\qquad c_i(k)=g_i\,e_i(k)+d_i\,c_i(k-1).
$$

- **Tolerance:** `reconstructor.matrix` = $10^{-9}$.
- **References:** Vidal, Gendron & Rousset 2010, JOSA A 27:A253 (Learn & Apply), as named in the docstring ("L&A").
