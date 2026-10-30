# Layer

## Theory and implementation

`LayerClass` is a plain container for the properties of one turbulent layer (identifier, seed, $D_{fov}$, pixel size, number of pixels, $r_{0,i}$, altitude, wind speed and direction, the sub-pixel displacement buffers and the `screen` object). All the numerics live in `PhaseScreenVonKarman` (`SAOS/infinitePhaseScreen.py`), the infinite phase screen of Assémat, Wilson & Gendron (2006) that each layer owns. How layers are scaled, placed and moved is described in [Atmosphere](atmosphere.md). The screen is a square array of $n\times n$ pixels of size $\delta$ (m), with **phase in rad at the wavelength at which $r_0$ is given** (500 nm in SAOS); multiply by $\lambda/2\pi$ for an OPD in m.

### Von Kármán statistics

The phase power spectral density (PSD) implemented in the FFT generator is

$$
\Phi_\phi(f) = 0.023\; r_0^{-5/3}\; \frac{\exp\!\left[-(f/f_m)^2\right]}{\left(f^2 + f_0^2\right)^{11/6}}, \qquad f_0 = \frac{1}{L_0},\quad f_m = \frac{5.92}{2\pi\, l_0},
$$

with $f$ in cycles/m. The inner scale is fixed to $l_0 = 10^{-10}$ m, so the exponential cut-off is effectively absent and only the pixel Nyquist frequency $1/2\delta$ limits the high frequencies. The extrusion uses the corresponding phase covariance (Assémat et al. 2006, eq. 5), as coded in `phase_covariance`,

$$
C_\phi(r) = \left(\frac{L_0}{r_0}\right)^{5/3}\frac{\Gamma(11/6)}{2^{5/6}\pi^{8/3}}\left[\frac{24}{5}\Gamma\!\left(\frac{6}{5}\right)\right]^{5/6}\left(\frac{2\pi r}{L_0}\right)^{5/6} K_{5/6}\!\left(\frac{2\pi r}{L_0}\right),
$$

where $K_{5/6}$ is the modified Bessel function of the second kind. It is finite at $r=0$ (a very small number is added to $r$ to avoid the singularity of $K_{5/6}$) and has no inner scale. The two descriptions are consistent for $l_0\to0$. The covariance is the one of the infinite continuous field, evaluated at the pixel positions (no pixel averaging).

### Initial screen

The first $n\times n$ screen is an FFT screen (Schmidt 2010 recipe): on a grid of $N_g$ points with $\Delta f = 1/(N_g\delta)$, complex Gaussian white noise is shaped by $\sqrt{\Phi_\phi}\,\Delta f$ (zero frequency set to 0) and transformed with an inverse FFT, keeping the real part:

$$
\phi(\mathbf{x}) = \mathrm{Re}\sum_{\mathbf{f}} \left(a + i b\right)_{\mathbf{f}}\,\sqrt{\Phi_\phi(\mathbf{f})}\;\Delta f\; e^{2\pi i \mathbf{f}\cdot\mathbf{x}}, \qquad a, b\sim\mathcal{N}(0,1).
$$

An FFT screen is periodic and lacks the scales larger than its own size. Two corrections are used (both on by default):

1. **Oversized grid.** $N_g = \lceil 4\,n \rceil$ (`initial_oversize=4`) and the central $n\times n$ part is cropped, so that the periodicity (which distorts focus and astigmatism for a grid of the pupil size) is moved far from the retained area.
2. **Subharmonics** (Lane, Glindemann & Dainty 1992; Johansson & Gavel 1994). Three levels $p=1,2,3$ of $3\times3$ frequencies with spacing $\Delta f/3^p$ (central one excluded) are added with the same PSD weighting, and the mean of the low-frequency sum is removed. Without them a screen the size of the pupil has about 1/4 of the tilt variance.

Scales larger than $1/(\Delta f/3^3)$ are not represented. The generator uses `numpy.random.default_rng(seed)`, so screens are reproducible for a given seed.

### Extrusion (Assémat–Wilson)

To move the screen by one pixel, a new line $\mathbf{X}$ ($n$ values) is generated conditionally on a set of $n_s$ existing pixels $\mathbf{Z}$ (the stencil),

$$
\mathbf{X} = A\,\mathbf{Z} + B\,\mathbf{b}, \qquad \mathbf{b}\sim\mathcal{N}(0, I_n),
$$

$$
A = C_{xz}\,C_{zz}^{-1}, \qquad B B^{T} = C_{xx} - A\,C_{zx}, \qquad B = U\,\Sigma^{1/2}\ \text{ with }\ U\,\Sigma\,U^{T} = \mathrm{svd}(BB^T),
$$

where the $C$ are blocks of the covariance matrix $C_\phi(|\mathbf{r}_a-\mathbf{r}_b|)$ between the new-line positions $x$ and the stencil positions $z$. $A$ is the minimum-variance linear predictor of the new line (a Gaussian conditional mean) and $BB^T$ the conditional covariance. $A$ is computed with a Cholesky solve of $C_{zz}$ (least squares with `rcond=1e-8` if it fails). Both the vertical (new row) and horizontal (new column) cases have their own matrices (`A_vert`, `B_vert`, `A_horz`, `B_horz`), computed once at construction. Adding a line removes the opposite one, so the screen always has $n\times n$ pixels and the state is just the current screen plus the random generator.

**Stencil.** The stencil is not the full screen. It contains:

- the `n_columns` lines adjacent to the new one (all $n$ pixels each; SAOS uses `n_columns=2`, Assémat et al. report that two are adequate);
- if `far_rows=True` (default) sparse far lines at distances (in lines from the screen edge) $d = \max(4, 2\,n_{col}), 2d, 4d, \dots < n$, each sampled every $\max(2, d/2)$ pixels across (edges included), plus the farthest line $n-1$. This follows the idea of the Fried & Clark (2008) stencil: with only adjacent lines the large scales along the direction of motion are lost (tilt of about 0.7 of the theory along it).

For motion in $-$row (new row at the top) the screen is used as is; for $+$row it is flipped, so that the same $A$, $B$ serve both signs (the covariance is stationary and isotropic). The same holds for the columns. Because the new line is predicted from a finite stencil, the conditional statistics are approximate: the spatial structure function of the extended screen is only approximately the theoretical one. Screens saved without far rows (A matrices of the old size) are loaded with `far_rows=False`.

### Cost and limitations

- Construction cost is dominated by the $(n_s + n)$-sized covariance, Cholesky and SVD of the two cases; extrusion costs one matrix-vector product ($n\times n_s$) per line.
- Only the whole-pixel motion is available at this level; sub-pixel displacement is handled by the buffers in `Atmosphere`.
- The statistics are Gaussian, stationary and isotropic by construction; the stencil introduces a small anisotropy between the direction of motion and the cross direction.

### References

- Assémat, F., Wilson, R. W., Gendron, E. (2006), Method for simulating infinitely long and non stationary phase screens with optimized memory storage, *Opt. Express* 14, 988.
- Fried, D. L., Clark, T. (2008), Extruding Kolmogorov-type phase screen ribbons, *J. Opt. Soc. Am. A* 25, 463.
- Lane, R. G., Glindemann, A., Dainty, J. C. (1992), Simulation of a Kolmogorov phase screen, *Waves in Random Media* 2, 209.
- Johansson, E. M., Gavel, D. T. (1994), Simulation of stellar speckle imaging, *Proc. SPIE* 2200, 372.
- Roddier, F. (1981), The effects of atmospheric turbulence in optical astronomy, *Progress in Optics* 19, 281.

::: SAOS.Layer.LayerClass
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
