# `test_dm.py` — DeformableMirror: influence functions, fitting, altitude footprints, geometry, dynamics, misregistration (T1)

Source: `tests/test_dm.py`

Covers `SAOS.DeformableMirror` (`updateDMShape`, `get_dm_opd`, `buildLayer`, `applyDynamics`, `set_actuator_misregistration`, `print_properties`), with `SAOS.LightPath` and `SAOS.MisRegistration` where noted. The reference is an independent re-implementation of the documented actuator model (function `model_opd`), which never touches `dm.coordinates`, `dm.high_res_coords`, `dm.influence_functions` or SAOS-generated OPD: an $n\times n$ actuator grid spanning the metapupil $D_{fov}$ with pitch $p=D_{fov}/(n-1)$, valid actuators $r\le D_{fov}/2+\mathrm{thresh}\cdot p$, Gaussian influence functions and a partition of unity,

$$g_k(\mathbf x)=\exp\!\big(\ln c\;|\mathbf x-\mathbf x_k|^2/p^2\big),\qquad W(\mathbf x)=\frac{\sum_k a_k\,g_k(\mathbf x)}{\sum_k g_k(\mathbf x)},$$

with $c$ the mechanical coupling (0.6 by default; $g=c$ at one pitch), sampled on the pupil pixel grid $(i-(N-1)/2)\,D/N$ ($N$ = 64, $D$ = 4 m, 11x11 actuators). Conventions verified: the DM is transmissive (OPD = IF $\times$ command, no factor 2) and adds to the atmospheric OPD; source azimuth 0 deg moves the altitude footprint along +x (columns), 90 deg along +y (rows). Tests that compare SAOS with its own alternative code path are named `test_regression_*` and are not physics checks. All tests are deterministic with numerical tolerances (`tests/tolerances/dm.yaml`, measured 2026-10-26). Whole module `fast`.

## `test_piston_gives_flat_opd`
- **Module:** `DeformableMirror.updateDMShape` (influence functions).
- **Tier:** fast.
- **What it checks:** an equal command $a=1\,\mu$m on every valid actuator gives OPD $=a$ across the pupil: peak-to-valley over $a$ is zero and the mean over $a$ is 1.
- **Theory / assumptions:** partition of unity, $\sum_k a\,g_k/\sum_k g_k = a$.
- **Tolerance:** `dm.linearity.rel` = $10^{-10}$ (float64 linear algebra; measured $\le10^{-13}$).
- **References:** none.

## `test_poke_matches_independent_actuator_model`
- **Module:** `DeformableMirror.updateDMShape`.
- **Tier:** fast.
- **What it checks:** the OPD of a random command (rms 0.1 um) over the whole pupil grid equals the independent Gaussian/partition-of-unity model, as $\max|\mathrm{OPD}-\mathrm{model}|/\max|\mathrm{model}|$ (absolute amplitude in metres).
- **Tolerance:** `dm.model.rel` = $10^{-10}$ (SAOS adds $10^{-12}$ to the normalisation).
- **References:** none.

## `test_single_actuator_surface_central_and_edge`
- **Module:** `DeformableMirror.updateDMShape` on a 61 px pupil (odd, so that the central actuator falls on a sample).
- **Tier:** fast.
- **What it checks:** for pokes of a central actuator, an edge actuator on the +x border and an inner-ring actuator, SAOS equals the model on every sample; for the central actuator the value at the actuator is the infinite-lattice peak, and the model values at one and two pitches (axis) and the diagonal neighbour are $c^{1},c^{4},c^{2}$ times that value; for the edge actuator the model value at the actuator is $1/\sum_k c^{(d_k/p)^2}$.
- **Theory / assumptions:** for an interior actuator of an infinite lattice poked with a unit command, the surface value at the actuator is

$$\frac{1}{\left(\sum_{i}c^{\,i^2}\right)^2}\approx0.1626\ \ (c=0.6),$$

  and the value at the $n$-th neighbour on an axis is $c^{n^2}$ times it (mechanical coupling = neighbour/actuator ratio). The pupil grid does not sample the other actuators, so only the central one is read from SAOS; the others come from the model, which SAOS equals on every sample.
- **Tolerance:** `dm.model.rel` $10^{-10}$ for the model comparisons; relative $10^{-3}$ for the lattice-peak and neighbour ratios; relative $10^{-9}$ for the edge value (all hard-coded except the first).
- **References:** none.

## `test_regression_poke_superposition_and_scaling`
- **Module:** `DeformableMirror.updateDMShape`, `influence_functions`.
- **Tier:** fast.
- **What it checks:** internal consistency (not a physics check): OPD$(c_1+c_2)$ = OPD$(c_1)$ + OPD$(c_2)$; OPD$(3c)$ = 3 OPD$(c)$; a full $n_{act}^2$ command equals the same command restricted to the valid actuators; a single poke equals the stored influence-function column times the command.
- **Tolerance:** `dm.linearity.rel` = $10^{-10}$.
- **References:** none.

## `test_dm_opd_enters_lightpath_with_positive_sign`
- **Module:** `LightPath.propagate` (`wfs_opd`) with a DM and a stand-in atmosphere.
- **Tier:** fast.
- **What it checks:** the WFS-path OPD is $(\mathrm{OPD}_{atm}+W)\,P$, with $W$ the independent actuator model and $P$ the pupil, i.e. the DM adds with a positive sign and no factor 2 for reflection (transmissive convention).
- **Tolerance:** `dm.model.rel` = $10^{-10}$ on the maximum relative difference.
- **References:** none.

## `test_tilt_and_defocus_fitting`
- **Module:** `DeformableMirror.updateDMShape`.
- **Tier:** fast.
- **What it checks:** with commands sampled on the actuators for a tilt ($c=a\,x_{act}$) and defocus ($c=a\,r_{act}^2$): (1) SAOS equals the independent actuator model over the whole pupil (maximum difference over the rms of the model); (2) the relative rms residual with respect to the target $a x$ or $a r^2$ (piston removed; residual and signal over the same region), computed in the inner region $r<0.7R$ and the edge annulus $0.7R\le r\le R$, equals the model's own value to numerical precision. The fit quality itself is a property of the finite Gaussian actuator model (reported as a metric; edge-annulus error 6.5 % for tilt); SAOS must reproduce it.
- **Tolerance:** `dm.model.rel` $10^{-10}$ for the first criterion; `dm.fitting.ratio_atol` = $10^{-8}$ on SAOS minus model residual ratio (mechanical coupling 0.6, 11x11 actuators over 4 m).
- **References:** Hardy 1998 (DM fitting, generic).

## `test_altitude_dm_footprint_physical_displacement`
- **Module:** `DeformableMirror.get_dm_opd` for an altitude DM (15x15 actuators at 10 km, 60 arcsec field, metapupil $D_{fov}=D+2h\tan(\mathrm{fov}/2)$).
- **Tier:** fast. Parametrised over (azimuth deg, displacement in grid steps) = (0, 0.7), (0, 1.6), (90, 2.3), (180, 1.3), (270, 0.8).
- **What it checks:** a source at field angle $\theta$ and azimuth $az$ sees the DM through the pupil displaced by $\delta=h\tan\theta\,(\cos az,\sin az)$. For the ramp command $c=g(x_{act}+0.7\,y_{act})$ the change of the OPD relative to the on-axis one equals $W(\mathbf x_p+\boldsymbol\delta_r)-W(\mathbf x_p)$ from the independent model, where $\boldsymbol\delta_r$ is $\delta$ rounded to the nearest grid step $D/N$ (integer-pixel model, the same as the atmosphere footprint). Fractional displacements 0.7 to 2.3 steps are chosen away from 0.5 so rounding is unambiguous. Criterion: relative $L_2$ error of the change over the pupil.
- **Theory / assumptions:** small field, line of sight at altitude $h$ displaced by $h\tan\theta$; azimuth 0 along +columns, 90 along +rows.
- **Tolerance:** `dm.footprint.rel` = 0.01 (relative $L_2$ error against the rounded model; an ideal fractional-shift implementation would give $7\times10^{-4}$).
- **References:** none.

## `test_altitude_dm_metapupil_diameter`
- **Module:** `DeformableMirror.buildLayer` (called on a stand-in object, no influence functions built).
- **Tier:** fast.
- **What it checks:** $D_{fov}=D+2h\tan(\mathrm{fov}/2)$ for the EST telescope (4.149 m, 60 arcsec, $h$ = 20 km).
- **Tolerance:** `dm.geometry.metapupil_rel` = $10^{-6}$ (relative, on $D_{fov}/\mathrm{expected}-1$).
- **References:** none.

## `test_altitude_dm_grid_step_matches_ground`
- **Module:** `DeformableMirror.buildLayer` (`pixel_size`, `D_px`, `D_fov`).
- **Tier:** fast.
- **What it checks:** for EST at $h$ = 0, 5 and 20 km the DM grid step equals the pupil pixel $D/N$; the grid covers the metapupil with a symmetric whole-pixel margin, $0\le D_{px}\,p-D_{fov}<2p$; and $D_{px}-N$ is even.
- **Tolerance:** `dm.geometry.grid_step_rel` = $10^{-3}$ for the step; margin in $[-10^{-9},2-10^{-9}]$ pixels; parity exact.
- **References:** none.

## `test_misregistration_pixel_size_matches_dm_grid`
- **Module:** `DeformableMirror.misReg` (`MisRegistration.spatialSampling`), `high_res_coords`.
- **Tier:** fast.
- **What it checks:** the metre-to-pixel conversion used by `MisRegistration` equals the DM grid step measured on `high_res_coords`.
- **Tolerance:** `dm.geometry.grid_step_rel` = $10^{-3}$ (relative).
- **References:** none.

## `test_dynamics_matches_dlsim`
- **Module:** `DeformableMirror.applyDynamics`, `updateDMShape` with `dynamicModel` (HDF5 state-space).
- **Tier:** fast.
- **What it checks:** a 7x7 DM with a second-order model (800 Hz, damping 0.3, zero-order hold at the loop rate $T_s=0.5$ ms) responds to a 60-step command sequence (a step then random) exactly as `scipy.signal.dlsim`, actuator by actuator; and the OPD after the sequence is the independent actuator model applied to the dlsim output.
- **Theory / assumptions:** discrete model from ZOH discretisation of

$$\dot{\mathbf x}=\begin{pmatrix}0&1\\-\omega^2&-2\zeta\omega\end{pmatrix}\mathbf x+\begin{pmatrix}0\\\omega^2\end{pmatrix}u,\quad y=(1\ 0)\,\mathbf x,\quad \omega=2\pi\cdot800\ \mathrm{Hz},\ \zeta=0.3,$$

  with unit DC gain.
- **Tolerance:** `dm.dynamics.rel` = $10^{-12}$ (torch float64 vs scipy) for the dynamic output; `dm.model.rel` $10^{-10}$ for the OPD.
- **References:** none (standard discrete-time state-space simulation).

## `test_actuator_misregistration_matches_independent_model`
- **Module:** `DeformableMirror.set_actuator_misregistration(shift_x, shift_y, rotation, magnification)`.
- **Tier:** fast. Parametrised over `shift 0.3 pitch` (0.12, -0.08 m, no rotation), `magnification 0.95`, `rotation 7 deg`, `combined` (0.03, -0.02 m, 7 deg, 1.02).
- **What it checks:** the DM shape equals the independent model with actuators moved to $\mathbf x'=s\,R\,\mathbf x+\mathbf d$ ($R$ a clockwise rotation for a positive angle, x right / y up; Gaussian width scaled by $s$, so that $g=c$ at one scaled pitch; partition of unity over the moved actuators), as documented; resetting to $(0,0,0,1)$ restores the original shape. Errors are relative to the rms of the unmisregistered surface.
- **Tolerance:** `dm.misregistration.exact` = $10^{-10}$.
- **References:** none.

## `test_actuator_misregistration_moves_a_poke_as_specified`
- **Module:** `DeformableMirror.set_actuator_misregistration`.
- **Tier:** fast.
- **What it checks:** the sense of each parameter on a single poked actuator at (+2 pitch, 0): a shift $(0.2p,-0.1p)$ moves the surface centroid by that vector, a +90 deg rotation (clockwise) takes it to $(0,-2p)$, a -90 deg rotation to $(0,+2p)$, and a magnification 1.1 to $(2.2p,0)$. Centroid is that of the region above half the maximum.
- **Tolerance:** `dm.misregistration.centroid_m` = 8 mm (2 % of the pitch; bias of the half-maximum-region centroid estimator, up to 6.4 mm on the independent model).
- **References:** none.

## `test_regression_actuator_vs_map_misregistration_convention`
- **Module:** `DeformableMirror.set_actuator_misregistration` vs `MisRegistration.apply_misreg`.
- **Tier:** fast. Parametrised over `shift +1 px x`, `shift +2 px y`, `rotation +90 deg`, `rotation -90 deg` (shifts in DM grid pixels).
- **What it checks:** equal-path consistency (not a physics check; the physical sense is covered by the previous test): the actuator-level misregistration has the same sense as the OPD-map misregistration where the map interpolation is exact (integer-pixel shifts and $\pm90$ deg), interior difference below tolerance (relative to the rms); the opposite transformation differs by more than ten times that difference.
- **Tolerance:** `dm.misregistration.convention` = $10^{-9}$ of the rms; opposite-sense ratio at least 10 (hard-coded).
- **References:** none.

## `test_print_properties_runs`
- **Module:** `DeformableMirror.print_properties`.
- **Tier:** fast.
- **What it checks:** logging runs without error and a record containing "Controlled Actuators" is emitted. Smoke test, no physics.
- **Tolerance:** assertion on the presence of the message.
- **References:** none.
