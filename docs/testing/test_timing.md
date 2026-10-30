# `test_timing.py` — loop latency: impulse response of the AO chain (T1)

Source: `tests/test_timing.py`

Covers the time structure of the closed loop built from `SAOS.LightPath` (delay $d$, `propagate(temporal_tick=True)`), `SAOS.ShackHartmann` (6x6 subapertures, 48 px pupil), `SAOS.InteractionMatrixHandler` (zernike IM), `SAOS.Controller` (tikhonov reconstructor, leaky integrator, 12 modes, `delay` = $d+1$) and `SAOS.DeformableMirror` (7x7 actuators, `updateDMShape`). The atmosphere is replaced by a stand-in object whose `getOPD` returns a prescribed OPD. The strategy is an impulse-response experiment: a one-frame tilt is injected at frame $k_0$ and the run with the impulse minus the run without it is examined, which removes the constant WFS offset. The test is `fast`.

Loop definition used as expectation (SAOS convention, from the docstring): the WFS measures frame $k$; `computeControlAction` reads the measurement of frame $k-d$; `updateDMShape` makes the command shape the DM seen by the WFS from frame $k+1$ on. Hence the first change of the controller output is at call $k_0+d$ and the first change of the DM OPD in the WFS path is at frame $k_0+d+1$, a total pure delay of $d+1$ frames.

## `test_impulse_latency_is_delay_plus_one`
- **Module:** `LightPath.propagate` with `delay`, `Controller.computeControlAction` (`delay = d + 1`), `DeformableMirror.updateDMShape(dynamicResponse=False)`; helper `chain` fixture builds the IM.
- **Tier:** fast. Parametrised over the light-path delay `delay` = 0, 1, 2 frames ($K_0$ = 4, $K$ = 14 frames, gain 0.5, leaky decay 0.9).
- **What it checks:** with the full signed difference arrays (impulse run minus no-impulse run): slopes before frame $k_0$, controller outputs before call $k_0+d$ and DM OPD maps before frame $k_0+d+1$ are exactly zero; slopes at $k_0$, controller output at $k_0+d$ and the DM OPD map at $k_0+d+1$ exceed a small floor; the DM OPD map at $k_0+d+1$ equals an independent Gaussian-actuator DM model applied to the controller output of call $k_0+d$; and the pupil-mean $x$-gradient of that map equals $-g$ times the injected gradient.
- **Theory / assumptions:** independent DM model with Gaussian influence functions, mechanical coupling 0.6 at one pitch, partition of unity:

$$\mathrm{OPD}(\mathbf x) = \frac{\sum_a c_a\,g_a(\mathbf x)}{\sum_a g_a(\mathbf x)},\qquad g_a(\mathbf x) = \exp\!\left[\ln(0.6)\,\frac{|\mathbf x-\mathbf x_a|^2}{p^2}\right],$$

  with $p = D/(n_{act}-1)$ and valid actuators within $D/2 + p/2$. For a pure tilt that the 12 modes represent and a leaky integrator of gain $g$ at the first response frame, the correction is $-g$ times the injected tilt gradient (OPD tilt 2 nm per px, about 0.1 px of spot shift). The slope baseline is assumed to cancel in the difference.
- **Tolerance:** `timing.latency`: `zero_atol` 0.0 (exact zeros), `signal_min` $10^{-12}$ (response must exceed it), `model_rel` $10^{-9}$ (DM OPD against independent model, float64), `gain_atol` 0.05 (the gain statement ignores the edge droop of the Gaussian-actuator tilt, which is 6.5 % in the edge annulus fit, see test_dm; measured 0.018).
- **References:** none specific (loop-delay convention of SAOS).
