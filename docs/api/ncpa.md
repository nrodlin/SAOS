# NCPA

## Theory and implementation

Non-common-path aberrations (NCPA) are static aberrations that affect the science path but not the wavefront sensor path. In SAOS they are a fixed OPD map in **metres** on the pupil grid, initialised to zero. They can be loaded from an H5 file (`opd/data`, resampled to the pupil size with bilinear interpolation if the shape differs) or set with `setOPD` (shape must match the pupil, otherwise `ValueError`).

In the light path the NCPA are added after the wavefront-sensor branch, only to the science OPD:

$$
\mathrm{OPD}_{sci} = P\,\big(\mathrm{OPD}_{wfs} + \mathrm{OPD}_{NCPA}\big), \qquad \varphi_{sci} = \frac{2\pi}{\lambda}\,\mathrm{OPD}_{sci},
$$

so the AO loop does not see them and does not correct them (they are the part of the science wavefront the sensor is blind to). The map is the same for all sources, wavelength independent in OPD (the phase scales as $1/\lambda$), and static. No pupil mask is applied inside the class, and `getPhase` actually returns the OPD in m.

::: SAOS.NCPA.NCPA
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
