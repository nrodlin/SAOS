# Source

## Theory and implementation

`Source` models a **natural point source at infinity** (NGS). Any laser guide star parameter (finite altitude, sodium profile, launch coordinates, spot FWHM) raises `NotImplementedError`.

### Position and frames

`coordinates = [rho, theta]` is the direction of the source with respect to the telescope axis, $\rho$ in arcsec and $\theta$ in degrees, so that on the sky $x = \rho\cos\theta$ (columns of the phase screens) and $y = \rho\sin\theta$ (rows). The same frame is used by the atmosphere footprints, the DM projection and the wind directions (see [Atmosphere](atmosphere.md)). The optional `chromatic_shift` is an array of shape $(N_L, 2)$ with an extra $(dx, dy)$ in arcsec per atmospheric layer, added to the source direction when the footprint on each layer is computed (a model of atmospheric dispersion; it is common to all sources of that wavelength).

### Photometry

Each band of `PHOTOMETRY_BANDS` is a tuple $[\lambda_c,\ \Delta\lambda,\ Z]$ (central wavelength and bandwidth in m, zero-point flux). The photon flux at the telescope is

$$
\Phi = \frac{Z}{368}\;10^{-0.4\,m} \quad [\mathrm{photons\,m^{-2}\,s^{-1}}],
$$

with $m$ the apparent magnitude. The zero points of the table are divided by an empirical constant (368, inherited from the OOMAO photometry tables), which gives $\Phi\approx 9\times10^{9}$ photons m$^{-2}$ s$^{-1}$ for $m_V=0$. The bandwidth is stored but not used to compute the flux (it is folded in the zero point). The number of photons collected by a sensor in an integration time $T$ is computed by the sensor as $\Phi\,A_{eff}\,T\,\eta$ with $A_{eff}$ from the [Telescope](telescope.md) and $\eta$ the light ratio of the sensor. The table is a fixed set of broadband (U to M) and narrow bands (e.g. `Na`, `EOS`, `IR1310`) and is approximate: no colour, spectral type or atmospheric extinction is modelled. The source is monochromatic at $\lambda_c$ (phase scaled as $500\,\mathrm{nm}/\lambda_c$ in the atmosphere, no polychromatic propagation). `Source.print_available_bands()` lists the table.

::: SAOS.Source.Source
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true