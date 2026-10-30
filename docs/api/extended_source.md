# Extended Source

## Theory and implementation

`ExtendedSource` models a patch of the solar disc (a square of side `fov` arcsec) observed through the atmosphere. The atmosphere is anisoplanatic across the patch, so the patch is divided into $n\times n$ **sub-directions** (`nSubDirs`), each propagated through the atmosphere as an independent point source (a `Source` at the sub-direction centre, with the photometry of the extended source), and the resulting images are blended afterwards (see [Correlating Shack-Hartmann](correlating_shwfs.md)). With `nSubDirs=1` the patch is a single direction at `coordinates`.

### Photometry

Only the bands `V` (500 nm), `V1` (525 nm), `R` (680 nm) and `IR` (1300 nm) are available, all with $\Delta\lambda = 10$ nm. The photon flux in the patch is computed from the ASTM G173-03 reference spectrum (direct normal irradiance $E_\lambda$ in W m$^{-2}$ nm$^{-1}$, read with `pvlib`, linearly interpolated at $\lambda_c$):

$$
\Phi = \Delta\lambda_{nm}\; E_{\lambda_c}\;\frac{\Omega_{fov}}{\Omega_\odot}\;\frac{\lambda_c}{h\,c}\quad[\mathrm{photons\,m^{-2}\,s^{-1}}],
\qquad \Omega_{fov} = \left(\frac{\mathrm{fov}}{206265}\right)^2,\quad \Omega_\odot = \pi\left(\frac{960}{206265}\right)^2 .
$$

Model choices: the Sun is a uniform disc of radius 960 arcsec (no limb darkening and no granulation contrast in the flux), the spectrum is the ground-level AM1.5 one (includes a fixed atmospheric transmission), and the flux does not depend on the position on the disc nor on the zenith angle. Each sub-direction source receives $\mathrm{round}(\Phi/n^2)$.

### Solar image and patch

The scene is an H5 image (`image/data`, plate scale `image` attribute `resolution` in arcsec/px, or `img_PS`). By default `cont_500nm.h5` is used, `cont_680nm.h5` for 680 nm, and the 500 nm image (with a warning) for any other wavelength. The patch is cut around pixel $(\lfloor s_0/2\rfloor, \lfloor s_1/2\rfloor) +$ offset of `coordinates`, in two versions: without padding (width fov) and padded (width $\mathrm{fov} + \texttt{patch\_padding} + \texttt{subDir\_margin}$, arcsec), widths rounded to whole image pixels. The patch position is rounded to the pixel of the image (no sub-pixel cut). Note that, in this cut, the $\rho\cos\theta$ component of `coordinates` displaces the first (row) axis of the image and $\rho\sin\theta$ the second, whereas the sub-direction coordinates below take $\rho\cos\theta$ as the column offset; both frames coincide only for a patch centred on the axis. A `ValueError` is raised if the patch falls outside the image.

### Sub-directions

Let $W = \mathrm{fov} + \texttt{patch\_padding}$. For $n>1$ the sub-direction size is

$$
s = \frac{2W}{n+1},
$$

and the $n$ windows per axis are placed with stride $s/2$, i.e. **50 % overlap**, so that they cover exactly $(n+1)\,s/2 = W$. For $n=1$, $s=W$. The sub-image of direction $(i,j)$ is cut from the padded patch, with width $\mathrm{round}\big((s+\texttt{subDir\_margin})/\mathrm{PS}\big)$ pixels (the margin avoids border effects when the image is shifted by the atmosphere) and top-left corner $\lfloor i\,s/(2\,\mathrm{PS})\rfloor$ along rows and $\lfloor j\,s/(2\,\mathrm{PS})\rfloor$ along columns. Here $i$ indexes the image **rows** and $j$ the **columns**: the patch is cut at `[row, col]` and placed back at the same `[row, col]` by the camera.

The sky offset of the centre of sub-direction $(i,j)$ with respect to the patch centre is

$$
d_i = \frac{s}{2} + i\,\frac{s}{2} - \frac{W}{2},
$$

and its sky coordinates relative to the telescope axis are $x = d_j + x_c$ (column) and $y = d_i + y_c$ (row), with $(x_c, y_c) = \rho_c(\cos\theta_c, \sin\theta_c)$ the patch centre. They are stored as polar $[\rho, \theta]$ (arcsec, degrees) in `subDirs_coordinates[0:2]`, and $s$ in `subDirs_coordinates[2]`. The sub-direction sources are created in the order $i$ (row) outer, $j$ inner.

### Blending filters

To recombine the sub-direction images into a single patch image a 2D window per direction (`filter_2D`) is built, following the idea of the WideField module of DASP. With $w = \mathrm{round}(s/\mathrm{PS})$ and stride $m = \mathrm{round}(w/2)$ the separable triangular window is

$$
t_k = 1 - \frac{\left|k - (w-1)/2\right|}{w/2}, \quad k=0,\dots,w-1, \qquad T_{kl} = t_k\, t_l ,
$$

each window is placed at $(i\,m,\ j\,m)$ of a mosaic of size $(n-1)\,m + w$ and divided by the sum of all the placed windows:

$$
F_{ij}(\mathbf{p}) = \frac{T(\mathbf{p}-\mathbf{p}_{ij})}{\sum_{i'j'} T(\mathbf{p}-\mathbf{p}_{i'j'})},
$$

so that the filters add up to one at every pixel of the mosaic, including its border (where a single window contributes). For $n=1$ the filter is 1. The windows do not vanish at the edges (the tent is non-zero there), which is what keeps the partition of unity at the border.

### Limitations

- The atmosphere is sampled at $n^2$ discrete directions (maximum `maxnSubDirs` per axis, default 7); the anisoplanatism inside a sub-direction is neglected, and each sub-direction is a single line of sight (integer-pixel footprint, see [Atmosphere](atmosphere.md)).
- Flux model: uniform solar disc, ground-level direct spectrum, 10 nm bandwidth.
- Solar images exist at 500 nm and 680 nm (downloaded from Zenodo, see [Installation](../installation.md#solar-images)); other wavelengths reuse the 500 nm granulation.

::: SAOS.ExtendedSource.ExtendedSource
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true