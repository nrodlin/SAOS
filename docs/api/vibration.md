# Vibration

## Theory and implementation

`Vibration` injects a telescope/instrument jitter as a pure tip-tilt wavefront error varying in time. The input is a pair of time series $\theta_x[k]$, $\theta_y[k]$ in **arcsec on sky** (H5: `x_axis/data`, `y_axis/data`, with the attribute `Ts` of `x_axis`), which must be sampled at the telescope sampling time (checked by exact equality, otherwise `ValueError`). Frame: $x$ = columns, $y$ = rows.

A wavefront tilt of $\theta$ rad is an OPD plane $\mathrm{OPD}(x,y)=\theta\,x$. With the pixel-centre coordinates normalised to the pupil radius, $\hat x_j = (j-(N-1)/2)\,2/N$ (so $\hat x = x/(D/2)$ exactly, with a linear ramp of slope $2/N$ per pixel), the implemented OPD at iteration $k$ is

$$
\mathrm{OPD}_k = \frac{D}{2}\,\frac{\theta_x[k]}{206265}\;\hat x\,P \;+\; \frac{D}{2}\,\frac{\theta_y[k]}{206265}\;\hat y\,P \quad[\mathrm{m}],
$$

with $P$ the telescope pupil mask. The OPD is wavelength independent and is converted to phase by the light path with the wavelength of each source. It is summed with the atmospheric and DM OPD in the light path, so the wavefront sensors see it (and the loop corrects it) like atmospheric tip-tilt; it is not applied while measuring interaction matrices. If the simulation is longer than the series the index wraps around (modulo the shorter of the two series) and a warning is logged at every call. The input is taken as-is: it is not generated, filtered or normalised by SAOS, and the amplitude is the one in the file (the sign convention is that of the OPD ramp above, $+x$ along the columns).

::: SAOS.Vibration.Vibration
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
