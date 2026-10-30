# `test_atmosphere.py` — von Kármán screens, extrusion, layer geometry, footprints and wind (T1)

Source: `tests/test_atmosphere.py`

Covers `SAOS.infinitePhaseScreen.PhaseScreenVonKarman` (initial FFT screen with subharmonics, `add_row`/`add_col` extrusion, `phase_covariance`), `SAOS.Atmosphere` (`initializeAtmosphere`, `update`, `getOPD`, `get_pupil_footprint`, `save`/`load`, per-layer r0, altitude, size and pixel scale) and the pupil of `SAOS.Telescope`. The strategy is to compare ensemble statistics of generated screens (structure function, PSD, joint $r_0$/$L_0$ least-squares fit in $\log D$) against closed-form von Kármán expressions written independently in `tests/theory.py`, and to check the deterministic geometry (layer scaling, footprint displacement on a prescribed linear phase ramp, frozen-flow shifts, seeds, save/load) by exact or integer-pixel comparisons. The whole module is `fast`.

## `test_theory_references_self_consistent`
- **Module:** none from SAOS; checks the references of `tests/theory.py` (`vk_psd`, `vk_covariance`, `vk_structure`, `kolmogorov_structure`).
- **Tier:** fast (module-level mark).
- **What it checks:** that the closed-form covariance at $r = 0, 0.05, 0.5, 5$ m equals the numerical Hankel transform of the PSD (`scipy.integrate.quad`, $r_0 = 0.10$ m, $L_0 = 25$ m), and that $D_{vK}/D_{Kolmogorov}\to 1$ at $r = 1, 10$ mm for $L_0 = 10^5$ m.
- **Theory / assumptions:** isotropic 2-D field, so the covariance is the Hankel transform of the PSD:

$$C(r) = \int_0^\infty 2\pi f\,\Phi(f)\,J_0(2\pi f r)\,\mathrm{d}f .$$

  For $r\ll L_0$ the outer-scale correction to $D$ is $O((r/L_0)^{1/3})$.
- **Tolerance:** `atmosphere.theory.hankel` = $10^{-4}$ (quadrature accuracy); `atmosphere.theory.kolmogorov_limit` = 0.01 (leading correction about 0.7 %).
- **References:** Roddier 1981 (Prog. Opt. 19); Assémat, Wilson & Gendron 2006 for the covariance form.

## `test_saos_phase_covariance_matches_theory`
- **Module:** `PhaseScreenVonKarman.phase_covariance`.
- **Tier:** fast.
- **What it checks:** the SAOS covariance function (documented as Assémat & Wilson eq. 5) divided by the independent $C(r)$ for $r = 0$ and 30 log-spaced values between 1 mm and 100 m ($r_0 = 0.10$ m, $L_0 = 25$ m). It also records $C(0)/(L_0/r_0)^{5/3}$ as a metric.
- **Theory / assumptions:** von Kármán covariance $C(r)$ as above.
- **Tolerance:** relative $10^{-6}$ (hard-coded in the test; numerical agreement of two evaluations of the same closed form).
- **References:** Assémat, Wilson & Gendron 2006, Opt. Express 14:988.

## `test_initial_screen_structure_function`
- **Module:** `PhaseScreenVonKarman.initial_screen` (FFT screen, subharmonics, 4x oversized grid, cropped).
- **Tier:** fast.
- **What it checks:** the mean structure function along $x$ and $y$ of 200 screens of 64 px (4 m, $r_0 = 0.10$ m, $L_0 = 25$ m) against $D_{vK}$, in two ranges of lag (4 px to $n_x/4$ and $n_x/4$ to $n_x/2$); the mean $D_x/D_y$ anisotropy over 4 to 16 px; and the recovery of $r_0$ from a joint $r_0$/$L_0$ fit and from an $r_0$-only fit with $L_0$ fixed. $L_0$ from the joint fit and the ratio at 1 px are only recorded.
- **Theory / assumptions:** $D(r) = \langle[\varphi(\mathbf{x}+\mathbf{r})-\varphi(\mathbf{x})]^2\rangle$ equals $D_{vK}(r)$ (ergodic average over pixels and 200 realisations). The docstring notes that an FFT screen lacks power above Nyquist, so $D(1\,\mathrm{px})\approx 0.92\,D_{vK}$; small lags are therefore not asserted. A 4 m screen does not span the $L_0$ turnover, so $L_0$ is weakly constrained.
- **Tolerance:** `atmosphere.initial_screen`: `D_ratio_small_r` 0.05, `D_ratio_large_r` 0.08, `anisotropy` 0.12, `r0_joint` 0.04, `r0_fixed` 0.03 (absolute on ratios; about $3.5\sigma$ over 10 seeds, 2026-10-26).
- **References:** Lane, Glindemann & Dainty 1992 (subharmonic FFT screens); Johansson & Gavel 1994 (FFT screen generation).

## `test_extruded_screen_structure_function`
- **Module:** `PhaseScreenVonKarman.add_row` (Assémat–Wilson extrusion with far-row stencil).
- **Tier:** fast.
- **What it checks:** after a burn-in of $n_x$ rows, the structure function of a 32 px (2 m) screen sampled every $n_x$ rows over 4000 rows (about 124 samples) against $D_{vK}$ for 1 px to $n_x/2$, averaged over axes (two lag ranges) and per axis (across and along the motion), plus the same $r_0$ fits as above. $D_y/D_{vK}$ at $n_x/2$ is recorded as a trend metric.
- **Theory / assumptions:** the extruded rows are built from the sampled covariance, so they do not show the sub-Nyquist deficit of the FFT screen; the ensemble average replaces the stationary ensemble.
- **Tolerance:** `atmosphere.extruded_screen`: `D_ratio_small_r` 0.08, `D_ratio_large_r` 0.15, `D_ratio_axis` 0.20, `r0_joint` 0.06, `r0_fixed` 0.05 (about $3.5\sigma$ over 10 seeds; sigma 0.016 to 0.087).
- **References:** Assémat, Wilson & Gendron 2006, Opt. Express 14:988.

## `test_outer_scale_initial_screen`
- **Module:** `PhaseScreenVonKarman.initial_screen`.
- **Tier:** fast.
- **What it checks:** with a screen that spans the turnover of the spectrum (128 px of 0.5 m = 64 m = 2.6 $L_0$; $r_0 = 0.5$ m, $L_0 = 25$ m), the joint $r_0$/$L_0$ fit of the mean $D(r)$ of 40 screens (lags from 2 px to 32 m) returns the generation parameters. The ratio $D/D_{vK}$ at 32 m is recorded.
- **Theory / assumptions:** the flattening of $D_{vK}(r)$ towards $2C(0)$ for $r\gtrsim L_0$ determines $L_0$; the pixel is chosen so that pixel$/r_0 = 1$ (resolved), and lag 1 px is excluded from the fit because it is below the sampling of a pixelated screen.
- **Tolerance:** `atmosphere.outer_scale_initial`: `L0_atol` 0.075 and `r0_atol` 0.025 on the fitted/true ratios ($3.5\sigma$; measured sigma 0.021 and 0.007).
- **References:** Assémat, Wilson & Gendron 2006; Roddier 1981.

## `test_outer_scale_extruded_screen`
- **Module:** `PhaseScreenVonKarman.add_row`.
- **Tier:** fast.
- **What it checks:** the same outer-scale recovery on an extruded screen of 64 px of 1 m ($r_0 = 1$ m, $L_0 = 25$ m) sampled every $n_x$ rows over 20000 rows (312 samples), fit from 2 px to $n_x/2 = 32$ m.
- **Theory / assumptions:** as in the initial-screen version.
- **Tolerance:** `atmosphere.outer_scale_extruded`: `L0_atol` 0.035, `r0_atol` 0.014 ($3.5\sigma$; sigma 0.010 and 0.004 over 10 seeds).
- **References:** Assémat, Wilson & Gendron 2006.

## `test_extruded_screen_low_frequencies`
- **Module:** `PhaseScreenVonKarman.add_row`.
- **Tier:** fast.
- **What it checks:** $D(r)$ at $r = n_x/8, n_x/4, n_x/2$ px along ($y$) and across ($x$) the extrusion, for a 32 px (2 m) screen, averaged over 16 independent screens of 40000 rows each (sampled every $n_x$ rows), against $D_{vK}$. Per-lag standard errors of the mean are recorded.
- **Theory / assumptions:** the ensemble (standard error about 0.009) resolves whether the extrusion truncates low frequencies.
- **Tolerance:** `atmosphere.extruded_lowfreq.atol` = 0.031 = $3.5\times$ the largest standard error of the mean with 16 screens ($0.035/\sqrt{16}$ = 0.0088, $D_x$ at $n_x/2$); measured means 0.999 to 1.008.
- **References:** none specific.

## `test_extrusion_directions_are_mirror_images`
- **Module:** `PhaseScreenVonKarman.add_row`, `add_col` (both signs of motion).
- **Tier:** fast. Parametrised over `axis` = `row`, `col`.
- **What it checks:** two screens with the same seed, the second one initialised with the mirror image of the first along rows (or columns), are extruded for $3n_x$ steps in opposite directions (+1 and -1); the first must equal the mirror of the second. This verifies that stencils and the A/B matrices are consistent in both directions. Convention-only check, no physical theory.
- **Tolerance:** max absolute difference $\le 10^{-10}$ rad (hard-coded).
- **References:** none.

## `test_screen_psd`
- **Module:** `PhaseScreenVonKarman.initial_screen`.
- **Tier:** fast.
- **What it checks:** the radially averaged periodogram of 200 initial 64 px screens (Hann window, mean removed, normalised to rad$^2$ m$^2$ by $\Delta x^2/(n\sum w^2)$) against $\Phi(f)$ in the band $6/L \le f \le 0.3/\Delta x$ (amplitude ratio per bin) and the log-log slope against the slope of the theoretical PSD in the same band (asymptotically $-11/3$).
- **Theory / assumptions:** Welch-type periodogram with a window; the lowest bins are excluded because of window leakage and the upper limit stays below Nyquist to avoid aliasing.
- **Tolerance:** `atmosphere.psd`: `amplitude` 0.15 (ratio), `slope` 0.15 (absolute).
- **References:** Roddier 1981

## `test_layer_r0_and_altitude_scaling`
- **Module:** `Atmosphere.__init__` / `initializeAtmosphere` layer set-up.
- **Tier:** fast.
- **What it checks:** for a 3-layer profile (fractions 0.5/0.3/0.2, altitudes 0/4/12 km, zenith 30 deg, 2 m telescope, 40 arcsec field): layer $r_{0,i}$ (stored in `fractionalR0`) against theory, $\sum r_{0,i}^{-5/3}$ against $(r_0\cos^{0.6}z)^{-5/3}$, layer altitude against $h/\cos z$, layer size `D_fov` against $D + 2\tan(\mathrm{fov}/2)h_i$, and that $n_{pix}\times$ pixel exceeds the layer size by less than two pixels (symmetric whole-pixel margin). The layer-pixel to telescope-pixel deviation is recorded.
- **Theory / assumptions:** Cn$^2$ integral along a slanted path:

$$r_{0,i} = r_0\,(\cos z)^{3/5} f_i^{-3/5},\qquad \sum_i r_{0,i}^{-5/3} = \left(r_0 \cos^{3/5} z\right)^{-5/3},\qquad h_i' = h_i/\cos z .$$

- **Tolerance:** relative $10^{-12}$ for r0 and altitude; $10^{-5}$ for the layer size (arcsec to rad conversion with 206265); margin in $[-10^{-9}, 2]$ pixels.
- **References:** Roddier 1981.

## `test_layer_pixel_equals_pupil_pixel`
- **Module:** `Atmosphere` layer sampling (`spatial_res`, `D_fov`, `npix`).
- **Tier:** fast.
- **What it checks:** the layer pixel pitch equals the pupil pixel $D/\mathrm{res}$ for four layers at 0, 3, 7.5 and 12 km (2 m, 32 px, 40 arcsec). A geometry convention (otherwise the pupil cut from the layer would not be $D$ wide).
- **Tolerance:** relative $10^{-6}$ (hard-coded).
- **References:** none.

## `test_pupil_area`
- **Module:** `Telescope.pupil`.
- **Tier:** fast.
- **What it checks:** the pupil pixel count against the analytic area of an annulus, $\frac{\pi}{4}N^2(1-\epsilon^2)$ with $\epsilon = 0.3$ central obstruction, for $N = 64$ and 128.
- **Tolerance:** absolute 0.01 on the ratio (1 %, hard-coded; sampling error of the pixelated edge).
- **References:** none.

## `test_footprint_displacement_direction`
- **Module:** `Atmosphere.getOPD` / `get_pupil_footprint` (off-axis footprint on a layer).
- **Tier:** fast. Parametrised over azimuth `az` = 0, 90, 180, 270 deg.
- **What it checks:** a source at an angle giving $h\tan\theta = 0.2$ m at a 10 km layer ($\theta\approx 4.1$ arcsec) samples the layer 0.2 m away along its azimuth, $(\Delta_{col},\Delta_{row}) = d(\cos az,\sin az)$. Measured on a prescribed linear phase ramp $\varphi = g\,x$ (or $g\,y$, $g = 0.05$ rad/m): the mean OPD difference between the off-axis and on-axis source gives the displacement independently of SAOS's pixel policy. The layer is made exactly $2D$ wide by tuning the field of view. The displacement 0.2 m is two 0.1 m pixels, so the footprint is rounded to the nearest layer pixel.
- **Theory / assumptions:** frozen layer at altitude $h$ is seen along the line of sight displaced by $h\tan\theta$; a linear ramp makes the OPD difference proportional to the displacement, $\Delta\mathrm{OPD}\cdot 2\pi/\lambda_{atm} = g\,\Delta$.
- **Tolerance:** `atmosphere.footprint.atol_m` = $10^{-6}$ m (float precision of the OPD difference, about $10^{-5}$ px).
- **References:** none.

## `test_footprint_displacement_fractional`
- **Module:** `Atmosphere.getOPD` / `get_pupil_footprint`.
- **Tier:** fast. Parametrised over $(d, az)$ = (0.16, 0), (0.09, 90), (0.04, 37), (0.15, 215), (0.16, 127), (0.09, 300) (metres, degrees).
- **What it checks:** fractional footprint offsets (0.4 to 1.6 layer pixels of 0.1 m, no component within 0.05 px of a half pixel) follow the integer-pixel model, $\mathrm{round}(d\cos az/p)\,p$ and $\mathrm{round}(d\sin az/p)\,p$ with $p = 0.1$ m, exactly; and lie within half a pixel of the physical $h\tan\theta$ (unbiased rounding). Same linear-ramp measurement as above.
- **Tolerance:** `atmosphere.footprint.atol_m` = $10^{-6}$ m for the model comparison; $|\Delta-h\tan\theta|/p\le 0.5+10^{-6}$ for the physical comparison.
- **References:** none (SAOS convention documented in the test).

## `test_wind_motion`
- **Module:** `Atmosphere.update` (layer shift), `PhaseScreenVonKarman.add_row` / `add_col`.
- **Tier:** fast. Parametrised over (px/step, direction deg) = (0.5, 0), (0.5, 180), (0.5, 90), (0.5, 270), (0.5, 45), (2.5, 0).
- **What it checks:** after 10 snapshots (every $\max(1,\lfloor 2/v\rfloor)$ steps) the accumulated content shift, measured by least-squares integer shift between successive layer screens (`citools.integer_shift`, $\pm4$ px), equals $v\,N\,T_s/\mathrm{pixel}$ along the wind direction, in the same frame as source azimuths (0 deg to +column, 90 deg to +row).
- **Theory / assumptions:** Taylor frozen flow, $\varphi(\mathbf{x},t)=\varphi(\mathbf{x}-\mathbf{v}t,0)$, implemented with integer-pixel motion (documented 1 px/step cap in the SAOS memory); hence the comparison is within $\pm1$ px.
- **Tolerance:** absolute 1.0 px on rows and columns (hard-coded; integer-pixel motion).
- **References:** none specific (Taylor hypothesis).

## `test_same_seed_reproducible`
- **Module:** `Atmosphere.initializeAtmosphere(randomState)`, `Atmosphere.update`.
- **Tier:** fast.
- **What it checks:** two atmospheres with the same `randomState` produce identical OPD after 25 updates (single layer at 0 m, 3 m/s, 30 deg); a different seed gives a different OPD (rms of the difference over rms of OPD $\ge 0.5$).
- **Tolerance:** difference exactly 0 for the same seed; $\ge 0.5$ for the different seed (hard-coded).
- **References:** none.

## `test_save_load_initial_state`
- **Module:** `Atmosphere.save`, `Atmosphere.load` (HDF5).
- **Tier:** fast.
- **What it checks:** the layer screen of a saved atmosphere (5 km layer) and that of a fresh atmosphere that loads the file are identical.
- **Tolerance:** exact (max absolute difference 0).
- **References:** none.

## `test_save_load_continues_identically`
- **Module:** `Atmosphere.save`, `Atmosphere.load`, `Atmosphere.update`.
- **Tier:** fast.
- **What it checks:** after 7 updates the state is saved and loaded into a second atmosphere; after 20 further updates both give the same OPD, so the extrusion state (random stream, stencil buffers) is preserved.
- **Tolerance:** $\mathrm{rms}(\mathrm{OPD}_1-\mathrm{OPD}_2)/\mathrm{rms}(\mathrm{OPD})\le10^{-12}$.
- **References:** none.

## `test_save_load_many_layers_order`
- **Module:** `Atmosphere.save`, `Atmosphere.load`.
- **Tier:** fast.
- **What it checks:** with 11 layers (0 to 10 km) every loaded layer keeps its altitude, i.e. the layer order survives the round trip (more than 9 layers exercises double-digit layer indices).
- **Tolerance:** absolute $10^{-6}$ m on the altitude difference.
- **References:** none.

## `test_odd_resolution_and_zero_cn2_layer`
- **Module:** `Telescope` / `Atmosphere.getOPD` edge cases.
- **Tier:** fast.
- **What it checks:** with an odd pupil sampling (31 px) an off-axis source returns a $31\times31$ finite OPD, and a layer with zero Cn$^2$ fraction has a screen identically 0 and does not crash.
- **Tolerance:** exact (shape equality, finiteness, max $|\varphi| = 0$).
- **References:** none.

## `test_random_seed_when_not_given`
- **Module:** `Atmosphere.initializeAtmosphere`, attribute `randomState`.
- **Tier:** fast.
- **What it checks:** without `randomState` two atmospheres differ, and the stored `randomState` reproduces exactly the screen of the first one.
- **Tolerance:** exact.
- **References:** none.

## `test_chromatic_shift_is_a_common_vector`
- **Module:** `Atmosphere.get_pupil_footprint`, `Source(chromatic_shift=...)`.
- **Tier:** fast.
- **What it checks:** the chromatic shift is an on-sky vector per layer, added to the source direction: an on-axis source with shift (4", 0) on the 5 km layer has the same footprint as a source at 4" azimuth 0; a source at 4" azimuth 90 with that shift has the same footprint as one at $4\sqrt2$" azimuth 45. Convention check on integer footprints.
- **Tolerance:** exact array equality.
- **References:** none.
