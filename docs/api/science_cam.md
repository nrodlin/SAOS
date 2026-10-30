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

!!! warning "Replicas of the PSF: limit on the field of view (under correction)"
    The MFT is evaluated on the pupil sampled at $\Delta x = D/n$, so the computed PSF is periodic in angle with period $\lambda/\Delta x$ (in rad), not only a single Airy pattern. If the camera half field of view exceeds this period, replicas of the PSF appear inside the frame. The condition for a valid frame is
    
    $$
    \frac{\mathrm{FoV}}{2} < \frac{\lambda}{\Delta x} = \frac{n\,\lambda}{D}
    \qquad\Longleftrightarrow\qquad
    \mathrm{FoV}_{max} = \frac{2\,n\,\lambda}{D}\ \text{[rad]} = 2\times 206265\,\frac{n\,\lambda}{D}\ \text{[arcsec]},
    $$
    
    that is, the maximum valid field grows linearly with wavelength, $\mathrm{FoV}_{max}\approx 1.99\times10^{7}\,\lambda[\mathrm{m}]\,(n/200)(4.149/D)$ arcsec. Measured example: at 500 nm with $n = 200$ pixels over $D=4.149$ m, $\lambda/\Delta x = 4.97$ arcsec; with a 9.975 arcsec field, the replicas at $\pm 4.97$ arcsec carry 80 to 90 % of the central peak. The Strehl ratio, the encircled energy and the image-based metrics are not reliable in that regime, and the frame energy normalisation is distorted. Use a field smaller than $\mathrm{FoV}_{max}$ (or a higher pupil resolution) for each band. This limitation is known and is currently being corrected.

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
