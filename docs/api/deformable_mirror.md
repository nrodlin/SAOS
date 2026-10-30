# Deformable Mirror

## Theory and implementation

The DM is modelled as a thin, transmissive phase screen at the conjugation altitude $h$ (default 0): the OPD map is a linear combination of influence functions (IFs) sampled on the telescope pixel grid, with an optional per-actuator temporal response. Units: commands and OPD in metres of wavefront (no factor 2), phase in rad at the source wavelength; sky x is the column of the arrays and y the row.

### Grid and metapupil

The pixel grid has step $\Delta = D/N_{res}$ and cell-centred coordinates $x_i = (i-\frac{D_{px}-1}{2})\,\Delta$, the same ones that the pupil, the atmosphere and the WFS use. The DM layer covers the metapupil seen by the field of view,

$$
D_{fov} = D + 2\,h\,\tan\!\left(\frac{\mathrm{FoV}}{2}\right),
\qquad
D_{px} = N_{res} + 2\left\lceil \frac{D_{fov}-D}{2\Delta}\right\rceil ,
$$

(margin rounded up to whole pixels, symmetric, so the on-axis footprint is centred).

### Actuators and valid actuators

Cartesian DM: $n_{act}\times n_{act}$ actuators on `linspace(-D_fov/2, D_fov/2, nActs)` in each axis, i.e. from edge to edge of the metapupil with pitch $p = D_{fov}/(n_{act}-1)$ (or the pitch given). Radial DM: a triangular (hexagonal) lattice with pitch $p$, rows separated by $p\sqrt3/2$ and odd rows shifted by $p/2$. Custom DM: user coordinates in metres. An actuator is valid when its radius satisfies $r \le D_{fov}/2 + f\,p$, with $f$ = `validActThreshpercentage` (default 0), or `valid_act_thresh_outer` when given. Only valid actuators are driven; a command of length $n_{act}^2$ is reduced to the valid ones.

### Influence functions

For analytic IFs, actuator $a$ has a Gaussian (radial basis function) of the distance $r$ to the pixel

$$
g_a(\mathbf{x}) = \exp\!\left[-(\varepsilon\,\|\mathbf{x}-\mathbf{x}_a\|)^2\right],
\qquad
\varepsilon = \frac{\sqrt{-\ln c}}{p},
$$

where $c$ is the mechanical coupling (value of the IF at one pitch, default 0.6), and they are normalised with a partition of unity (PoU) so that they add up to 1 at every pixel,

$$
\mathrm{IF}_a(\mathbf{x}) = \frac{g_a(\mathbf{x})}{\sum_b g_b(\mathbf{x}) + 10^{-12}} .
$$

A uniform command thus produces a uniform OPD (pure piston), also beyond the outermost actuators, and no pixel has less than its share of the stroke; the price is that $\mathrm{IF}_a(\mathbf{x}_a) < 1$ and the effective coupling differs from $c$. Custom IFs (`modes`, $[D_{px}^2, n_{valid}]$) are normalised the same way by default (`normalize_IFs`). The OPD is

$$
\mathrm{OPD}(\mathbf{x}) = \sum_a \mathrm{IF}_a(\mathbf{x})\,u_a ,
$$

and it is clipped to $\pm$ `maxStrokePtV` (default 100 um) after the sum, not per actuator.

### Dynamic model

If `dynamicModel` (HDF5 with $A,B,C,D$ and the sampling time `Ts`, which must equal the simulation one) is given, every valid actuator follows the same discrete state-space system, evolving one step each time the shape is updated:

$$
x_{k+1} = A\,x_k + B\,u_k,
\qquad
y_k = C\,x_k + D\,u_k ,
$$

with $y_k$ replacing the command in the OPD sum above. The initial state is zero. The response of each actuator is independent (no inter-actuator dynamic coupling beyond the IF).

### Footprint for altitude DMs

For a source at angular distance $\theta$ and azimuth $\psi$, the beam footprint on a DM at altitude $h$ is displaced by $h\tan\theta$ metres, converted to pixels and rounded to the nearest integer (integer-pixel model, the same as the atmosphere, so the error is at most 0.5 px per axis and unbiased). A square window of $N_{res}$ pixels centred there is cut from the metapupil map and multiplied by the pupil. For the Sun only the central sub-direction is used.

### Misregistration and rotation

Two mechanisms exist. (i) `misReg` acts on the OPD map at run time: magnification (bilinear), rotation (cubic spline, `nearest` border) and shift (bilinear, shift in metres divided by the pixel size) with the border replicated; a positive rotation is clockwise with x to the right and y up. Its rates (`vx`, `vy`, `wz`) can move it in time. (ii) `set_actuator_misregistration` is exact for analytic IFs: the actuator coordinates are transformed, $\mathbf{x}'_a = s\,R(\vartheta)\,\mathbf{x}_a + (\Delta_x,\Delta_y)$, the Gaussian width is scaled by $s$ ($\varepsilon\to\varepsilon/s$) and the IFs and their PoU normalisation are recomputed on the grid, with no interpolation of the OPD map. The two must not be combined (use only one).

### Modal bases (`SAOS/modalBasis`)

All the bases are $n_{act}^2\times n_{modes}$ matrices of values sampled at the actuator coordinates (zero at invalid actuators), each column scaled to $[-1,1]$ (divided by its maximum absolute value), and by default $n_{modes} = n_{valid}-1$: the piston-free set.

- Zernike: Noll indexing, $j$ even gives $\cos(m\theta)$ and $j$ odd gives $\sin(|m|\theta)$; radial polynomials by the Jacobi relation $R_n^m(\rho) = (-1)^{(n-m)/2}\rho^{m}P^{(m,0)}_{(n-m)/2}(1-2\rho^2)$ with a three-term recurrence (stable for any order; the explicit factorial sum loses all precision above $n\sim40$). The radius is normalised by the outermost valid actuator. Each mode is sampled, the piston is removed (mean over the valid actuators) and it is scaled to unit rms over them (Noll 1976). The sampled modes are not exactly orthogonal on a discrete actuator set, and become rank-deficient for large order.
- Karhunen-Loeve: double diagonalisation. The sampled Zernike set is orthonormalised ($Z = QR$), the Kolmogorov covariance of the Zernike coefficients (Noll 1976; Roddier 1990) is transformed, $R\,C\,R^T$, and diagonalised; the KL modes are $QU$ ordered by decreasing variance. The amplitude is not scaled with $r_0$ or $D$.
- Disc harmonics: eigenfunctions of the Laplacian on the disc, $J_m(k_{nm}\rho)\{\cos,\sin\}(m\theta)$ with $J_m'(k_{nm})=0$ (Milton and Lloyd-Hart 2005), piston removed, unit rms, then Gram-Schmidt orthogonalised on the valid actuators.
- Zonal and Hadamard bases are also available.

### Limitations

Gaussian IFs are a model of the mirror, not a measured IF; the stroke is clipped on the sum rather than per actuator; there is no hysteresis, creep or actuator failure model; the footprint of an altitude DM uses whole pixels.

### References

- Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford University Press.
- Noll, R. J. (1976), "Zernike polynomials and atmospheric turbulence", J. Opt. Soc. Am. 66, 207.
- Roddier, N. (1990), "Atmospheric wavefront simulation using Zernike polynomials", Opt. Eng. 29, 1174.
- Milton, M. and Lloyd-Hart, M. (2005), "Disk harmonic functions for adaptive optics simulations", in *Adaptive Optics: Analysis and Methods*, OSA.

::: SAOS.DeformableMirror.DeformableMirror
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true