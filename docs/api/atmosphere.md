# Atmosphere

## Theory and implementation

### Model

The atmosphere is a stack of $N_L$ thin turbulent layers with von Kármán statistics (see [Layer](layer.md) for the statistics of one layer). Layers are independent, horizontally stationary and move rigidly ("frozen flow", Taylor hypothesis) at a constant velocity. Propagation is geometric: the phase seen along a line of sight is the sum of the layer phases sampled at the point where the line crosses each layer (near-field approximation, no Fresnel propagation, no scintillation).

**Conventions.** The reference wavelength of the screens is $\lambda_0 = 500$ nm. Layer screens are stored as **phase in rad at 500 nm**; the value returned by `getOPD` is an **OPD in m**. Sky coordinates of a source are $[\rho, \theta]$ with $\rho$ in arcsec and $\theta$ in degrees, giving $x = \rho\cos\theta$ along the **columns** and $y = \rho\sin\theta$ along the **rows** of the screens (the same frame is used for wind directions, DM misregistration and the chromatic shift). The wind direction is the direction *towards which* the turbulence moves in that frame.

### Layer scaling

With the zenith angle $z$ and fractional weights $w_i$ ($\sum_i w_i = 1$, `fractionalR0`), layer $i$ is built with

$$
r_{0,i} = r_0\,\cos^{3/5}(z)\; w_i^{-3/5}, \qquad h_i' = \frac{h_i}{\cos z},
$$

so that $\sum_i r_{0,i}^{-5/3} = r_0^{-5/3}\cos^{-1}(z)$, i.e. the total Fried parameter is the zenith $r_0$ scaled to the airmass $1/\cos z$ (plane-parallel atmosphere). A layer with $w_i = 0$ gets $r_{0,i}=\infty$ and a null screen. The slant altitude $h_i'$ is used for the footprint geometry. The pupil is not foreshortened, and neither the wind speed nor $L_0$ are scaled with $z$. The attribute `layer.fractionalR0` stores the resulting $r_{0,i}$ in metres (not the fraction).

### Metapupil and sampling

The layer pixel is the pupil pixel, $\delta = D/N$, where $N$ is the telescope `resolution`. The part of the layer seen by the field of view (FoV, diameter in arcsec taken from the telescope) is

$$
D_{fov,i} = D + 2\,h_i'\tan(\mathrm{FoV}/2), \qquad
n_{pix,i} = N + 2\left\lceil \frac{D_{fov,i}-D}{2\delta} - 10^{-9} \right\rceil ,
$$

i.e. a square screen of $n_{pix}\times n_{pix}$ pixels whose margin around the pupil is rounded up to whole pixels and symmetric ($n_{pix}-N$ even), so that the on-axis footprint is centred on the screen. Each screen is a `PhaseScreenVonKarman` with `n_columns=2` and the default options (subharmonics, 4x oversized initial grid, far rows). Layer $i$ is seeded with `randomState + 1000 i`; if no seed is given a random one is drawn, logged and stored in `atm.randomState`. Layers are built (and updated) in a thread pool.

### Footprint of a line of sight

The direction of a source is the vector $\mathbf{v} = (\rho\cos\theta,\ \rho\sin\theta) + \mathbf{c}_i$ in arcsec, where $\mathbf{c}_i$ is the optional chromatic shift of layer $i$ (one $(dx, dy)$ per layer, common to all the sources of that wavelength, see [Source](source.md)). The footprint centre on layer $i$, in layer pixels from the screen centre, is

$$
\mathbf{p}_i = \frac{h_i'\,\tan(|\mathbf{v}|/206265)}{\delta}\;\frac{\mathbf{v}}{|\mathbf{v}|}, \qquad
\text{col}_0 = \mathrm{round}(p_{x}) + \lfloor n_{pix}/2\rfloor,\quad
\text{row}_0 = \mathrm{round}(p_{y}) + \lfloor n_{pix}/2\rfloor .
$$

The pupil is the $N\times N$ window of the screen starting at $(\text{row}_0 - \lfloor N/2\rfloor,\ \text{col}_0 - \lfloor N/2\rfloor)$. All sources are at infinity, so the footprint has the same size on every layer (no cone effect).

**Integer-pixel model (by design).** The footprint position is rounded to the nearest layer pixel (`np.round`), with no sub-pixel interpolation. The position error is at most $\delta/2$ per axis and is unbiased, but the angular separation between two sources is quantised to $\delta/h_i'$ on each layer. Interpolating would smooth the screens and alter their spectrum, which is why it is not done. There is no bounds check: a source outside the circle of the telescope FoV makes the window leave the screen and the extraction fails.

### Frozen flow with whole-pixel steps

Each AO step of duration $T_s$ (`telescope.samplingTime`) adds to two per-layer buffers

$$
s_x \mathrel{+}= \frac{v\cos\phi\; T_s}{\delta}, \qquad s_y \mathrel{+}= \frac{v\sin\phi\; T_s}{\delta},
$$

with $v$ the wind speed and $\phi$ the wind direction. The integer part $n = \mathrm{trunc}(s)$ is applied to the screen as $|n|$ successive extrusions (several per step for a fast layer, none if the displacement is below one pixel) and the fractional remainder stays in the buffer. Hence the mean velocity is exact, but the screen moves in whole-pixel jumps: within a step the instantaneous displacement can differ from the ideal one by up to $\delta$ (temporal jitter of the order of $\delta/v$). Sign conventions:

- $s_x>0$: `add_col(+1)` adds a new column at the left (index 0) and the pattern moves towards $+$column ($+x$).
- $s_y>0$: `add_row(-1)` adds a new row at the top (index 0) and the pattern moves towards $+$row ($+y$). The row sign is inverted with respect to the method (`add_row(+1)` appends at the bottom, i.e. moves the pattern towards $-$row).

The buffers and the random generator of the extrusion are the evolution state of a layer.

### OPD of a line of sight (`getOPD`)

For each line of sight, the phase at 500 nm of every layer is cut at its footprint, rescaled to the observing wavelength $\lambda_s$ under the non-dispersive assumption, and summed:

$$
\varphi_s = \sum_i \varphi_{500,i}\;\frac{\lambda_0}{\lambda_s}, \qquad
\mathrm{OPD} = \varphi_s\,\frac{\lambda_s}{2\pi} = \frac{\lambda_0}{2\pi}\sum_i \varphi_{500,i}\ [\mathrm{m}] .
$$

The OPD is therefore wavelength independent (refractive index of air constant with $\lambda$); only the phase in rad (computed later as $2\pi\,\mathrm{OPD}/\lambda_s$) depends on the band. The pupil mask is not applied here. For an extended source (`tag == 'sun'`) the OPD is computed for each sub-direction star and returned with an extra leading axis. The chromatic shift only displaces the footprint, i.e. it models differential atmospheric refraction between wavelengths as a per-layer shift of the line of sight.

### Save and load

`save` writes an H5 file with, per layer, the attributes (`seed`, `D_fov`, `spatial_res`, `npix`, `fractionalR0`, `windSpeed`, `windDirection`, `altitude`, `n_columns`), the extrusion matrices `A_vert`, `B_vert`, `A_horz`, `B_horz`, the current phase screen, the serialised state of the random generator and the two displacement buffers. `load` rebuilds the screens from these data without recomputing the covariance matrices or the initial screen, so a loaded atmosphere continues exactly as the saved one would have (files without the generator state restart it from the seed). Atmosphere parameters not stored in the file ($L_0$, sampling time) come from the object being loaded into.

### Limitations

- Integer-pixel footprints and whole-pixel motion (above); the pixel is $D/N$, so the resolution sets the position accuracy.
- Near-field, non-dispersive, plane-parallel layers; no scintillation, no cone effect, no $L_0$ or wind scaling with zenith angle.
- Extrusion statistics are approximate (stencil truncation, see [Layer](layer.md)).

### References

- Roddier, F. (1981), The effects of atmospheric turbulence in optical astronomy, *Progress in Optics* 19, 281.
- Assémat, F., Wilson, R. W., Gendron, E. (2006), Method for simulating infinitely long and non stationary phase screens with optimized memory storage, *Opt. Express* 14, 988.
- Sasiela, R. J. (2007), *Electromagnetic Wave Propagation in Turbulence*, SPIE Press.

::: SAOS.Atmosphere.Atmosphere
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true