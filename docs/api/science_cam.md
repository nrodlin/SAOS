# Science Camera

## Theory and implementation

`ScienceCam` computes the image on the science focal plane from the phase at the pupil. Conventions: sky x is the image column and y the row; phase in rad at the wavelength of the source; `plate_scale` in arcsec/px (206265 arcsec per rad).

### PSF by matrix Fourier transform

The pupil field $P\,e^{i\phi}$ is used on the native pupil grid of $n$ cells of side $\Delta x = D/n$, without resampling the phase (a resampling would low-pass the phase and bias the Strehl ratio upwards). With $F = \lambda/(D\,\mathrm{ps})$ the diffraction scale in camera pixels, the image is

$$
A_{km} = \exp\!\left[-2\pi i\,\frac{(k-\lfloor n_{pix}/2\rfloor)\,(m-\frac{n-1}{2})}{F\,n}\right],
\qquad
E = A\,(P\,e^{i\phi})\,A^{T},
\qquad
\mathrm{PSF} = \frac{|E|^2}{(F\,n)^2\,\sum P^2} .
$$

Pixel $k$ is at the angle $(k-\lfloor n_{pix}/2\rfloor)\,\mathrm{ps}$ exactly, for any plate scale. The PSF centre (zero tilt) is therefore at pixel $\lfloor n_{pix}/2\rfloor$ in both axes, the centre of a pixel, for odd and even sizes. The sign is that of an FFT: a phase growing along the columns moves the PSF towards larger column indices. The normalisation is such that the energy over one period of the image plane is 1 (Parseval); the cropped $n_{pix}\times n_{pix}$ camera frame sums to the fraction of that energy inside the field. `get_frame` then renormalises each frame to unit sum. The ideal PSF used as Strehl reference is computed once per wavelength.

### Band limit of the sampled pupil field

The pupil field is known only at its samples, so its spectrum is defined up to the Nyquist frequency $1/(2\Delta x)$, i.e. for

$$
|\theta_x|,\ |\theta_y| \le \frac{\lambda}{2\,\Delta x} = \frac{n\,\lambda}{2D}
\qquad (2.49'' \text{ at } 500\ \mathrm{nm},\ 3.38'' \text{ at } 680\ \mathrm{nm}\ \text{for}\ n = 200,\ D = 4.149\ \mathrm{m}).
$$

Beyond it the transform of the samples is periodic with period $\lambda/\Delta x$ and would only repeat the PSF (replicas
carrying 80–90 % of the central peak at $\pm\lambda/\Delta x$ when the half field exceeds that value). `compute_psf`
therefore sets the image to zero outside the band: the MFT kernel rows have weight $w(u) = 1$ for $|u| < 1/2$
(cycles per pupil sample), $1/\sqrt{2}$ at $|u| = 1/2$ (the same frequency as $-1/2$: its energy is shared) and $0$
beyond. This is the Whittaker–Shannon interpretation of the sampled field: inside the band the PSF is exactly the
transform of the samples, which is the true one when the phase is well sampled ($\Delta x \ll r_0$; the field
$e^{i\phi}$ is then band-limited to a good approximation, the criterion being $D_\phi(\Delta x) \ll 1\ \mathrm{rad}^2$).
The PSF energy over the band is 1 (Parseval over one period). The light that a continuous turbulent pupil scatters
beyond the band is not represented with this sampling; when the camera field exceeds the band a warning gives the
telescope resolution that would cover it ($n \ge 2\,\theta_{max} D/\lambda$, e.g. $n \ge 401$ for $\pm 4.99''$ at
500 nm).

Two alternatives were considered and rejected: interpolating the phase to a finer grid before the transform (the
behaviour before 2026-10-27) low-passes the phase and biases the Strehl ratio upwards (+5 to +16 %); a field constant
inside each pupil pixel (zero-order hold) suppresses the replicas through the pixel transfer function
$\mathrm{sinc}^2(\theta\,\Delta x/\lambda)$ but also attenuates the halo inside the band (×0.87 at 1″ and ×0.57 at 2″ in
V), independently of how well the phase is sampled. References: Bracewell, *The Fourier Transform and Its
Applications*; Goodman, *Introduction to Fourier Optics* (sampling theorem).

### Extended source (Sun)

For a solar source (`ExtendedSource`, $N_{sd}\times N_{sd}$ sub-directions with a pupil phase each; a 2D phase is repeated for all of them) the frame is built as in the [Correlating SH WFS](correlating_shwfs.md) page: for each sub-direction a PSF is computed on the grid of its patch, the solar patch $O_i$ (resampled bilinearly to the camera plate scale) is convolved with it through FFTs,

$$
I_i = \left|\mathcal{F}^{-1}\!\left[\mathcal{F}[O_i]\;\mathcal{F}[\mathrm{PSF}_i]\right]\right| ,
$$

with the PSF centre moved to the origin so that the circular convolution does not shift the object, and every $I_i$ rescaled to the energy of $O_i$. The patches are blended with separable triangular windows (50 % overlap, resized to the camera scale and renormalised to a partition of unity, with the sum of weights used as the denominator), placed in a mosaic of $(\mathrm{FoV}+\text{padding})/\mathrm{ps}$ pixels, and the central $n_{pix}$ pixels are cropped. The source field of view must be at least the camera one. This is a per-direction isoplanatic approximation: the PSF is constant inside each patch.

### Exposure

Each propagation step (`LightPath`) produces a noise-free short-exposure frame of unit sum. The frame is stored every `decimation` iterations, and the frames are accumulated for $\lceil T_{int}/(\mathrm{dec}\,T_s) \rceil$ frames (see [Light Path](light_path.md)); they are summed with equal weights, renormalised to unit energy, scaled to the photons $\Phi\,T_{int}\,A_{eff}\,\eta_{light}$ and, if noise is enabled, passed through the [Detector](detector.md). The long exposure is therefore the incoherent sum of short exposures at the sampling rate: the temporal evolution within one $T_s$ is not simulated.

### References

- Soummer, R. et al. (2007), "Fast computation of Lyot-style coronagraph propagation", Opt. Express 15, 15935.
- Goodman, J. W., *Introduction to Fourier Optics*.

::: SAOS.ScienceCam.ScienceCam
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
