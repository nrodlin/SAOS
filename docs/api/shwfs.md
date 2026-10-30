# Shack-Hartmann Wavefront Sensor

## Theory and implementation

This section describes what `ShackHartmann` computes. Conventions: sky x is the image column and y the image row; all detector quantities are in pixels (px), phases in radians (rad) at the wavelength of the source, OPD in metres. The pupil grid has `resolution` cells of side $D/N_{res}$ and every map is sampled at cell centres.

### Subaperture split and resampling

The pupil is divided into $n_s \times n_s$ square subapertures of width $d = D/n_s$ (`nSubap`). Each one is resampled to $n_p$ cell-centred samples per side (`npix_pupil`, a multiple of the pupil pixels per subaperture `npix_phase`), i.e. at

$$
x_j = \left(j + \tfrac12\right)\frac{d}{n_p}, \qquad j = 0,\dots,n_p-1 ,
$$

measured from the subaperture edge, by separable linear interpolation of the two nearest pupil cell centres. In the outermost half cell the line through the two edge samples is extrapolated, so a linear phase (a pure tilt) is reproduced exactly, with no shift of the subaperture centre. The pupil amplitude is resampled by area weighting: the amplitude of each fine sample is $\sqrt{c}$, with $c\in[0,1]$ the fraction of the sample covered by the pupil, so $\sum |a|^2$ equals the illuminated area of the subaperture. The phase uses a normalised interpolation, $\mathrm{interp}(P\,\phi)/\mathrm{interp}(P)$, so the points at the pupil edge are not biased by the zeros outside the pupil; no phase outside the pupil is ever used.

By default $n_p$ is the smallest value for which the bias of the spot second moment, caused by the periodic replicas of a field sampled at $n_p$ points, stays below 1 %. That bias is approximately $1.22\,(u/n_p)^2$ with $u$ the half-width of the detector window in $\lambda/d$ (an empirical calibration against a converged Fourier sum). `pupil_oversampling = k` forces $n_p = k\,n_{lenslet}$ instead.

### Spot formation

The field of a subaperture is $E_0(x,y) = a(x,y)\,e^{i\phi(x,y)}$. The spot is obtained with a matrix Fourier transform (MFT; Soummer et al. 2007), not with a padded FFT, so the detector plate scale is exact and independent of the pupil sampling. With $F = \lambda/(d\,\mathrm{ps})$ the diffraction width in detector pixels ($\mathrm{ps}$ = `plate_scale` in arcsec/px, $\lambda/d$ converted with 206265 arcsec/rad),

$$
A_{kj} = \exp\!\left[-2\pi i\,\frac{(j-\frac{n_p-1}{2})(k-c)}{F\,n_p}\right],
\qquad
E = A\,E_0\,A^{T},
\qquad
I = \frac{|E|^2}{(F\,n_p)^2},
$$

so image pixel $k$ is at the angle $(k-c)\,\mathrm{ps}$ exactly. The window has `npix_lenslet` pixels, $c = (n_{lenslet}-n_{subap})//2 + (n_{subap}-1)/2$, and a flat phase gives a spot centred in the window that is cut for the detector. Only the valid subapertures are computed.

`fieldOfView` sets the detector window `npix_subap` $= \mathrm{round}(\mathrm{FoV}/\mathrm{ps})$. The extra `fft_fieldOfView_oversampling` only enlarges the computed window (`npix_lenslet`), which is then cropped to the central `npix_subap` pixels; since the MFT pixels are evaluated independently, the cropped values do not depend on it and the extra pixels act as additional guard pixels between subapertures in the frame (`guardPx` more pixels are added, and the camera has $n_s\,n_{lenslet} + (n_s+1)\,\mathrm{guardPx}$ pixels per side). Photons outside the window are lost (the frame is renormalised inside it). The detector pixel is not integrated: the MFT samples the intensity at pixel centres (no pixel-MTF).

Because the pupil is sampled at $d/n_p$, the spot is periodic with period $F\,n_p$ px; the window must be much smaller than that period (see the default $n_p$ above). $F \ge 1$ is required (Nyquist sampling is $F\ge2$).

### Photons and noise

The noise-free frame is normalised to unit energy by the detector and scaled to $N_{ph} = \mathrm{round}(\Phi\,A_{eff}\,T_s\,\eta_{light})$ photons per frame (flux $\Phi$ of the source, effective area, sampling time, `lightRatio`). The light is shared among the subapertures in proportion to their illuminated area; the photon budget is that of the whole frame. Detector effects are described in [Detector](detector.md). Without `noiseFlag` the ideal frame is passed unchanged (the centroid is scale-invariant).

### Valid subapertures

The illumination of a subaperture is the mean of the (bilinearly resized) pupil mask over its $n_{phase}^2$ pupil pixels. It is valid when it is at least `validSubapThreshold` (default 0.5). Partially illuminated subapertures are therefore imaged through their illuminated part only and carry the corresponding fraction of the light. There are $2\,n_{valid}$ slopes, ordered x (columns) first and y (rows) after.

### Centroid

On the cropped window of each subaperture, with pixel intensities $I_p$ and coordinates $(x_p, y_p)$ (pixel index, column and row), the thresholded centre of gravity (CoG) is

$$
w_p = \max(I_p - T,\,0), \qquad
\bar{x} = \frac{\sum_p w_p\,x_p}{\sum_p w_p}, \qquad
\bar{y} = \frac{\sum_p w_p\,y_p}{\sum_p w_p},
$$

where $T$ is the $(n_b+1)$-th largest intensity of the subaperture and $n_b$ = `use_brightest` (default 50). A pixel enters or leaves the group with zero weight, so the response is continuous, including around zero tilt for a symmetric spot. With $n_b$ equal to the number of window pixels the plain CoG is used (recommended for linearity tests). Non-finite centroids (empty weights) are set to 0 with a warning. The thresholded CoG is biased towards the window centre for large displacements and by truncation of the wings; the response is linear only for small shifts compared with the spot size.

### Reference slopes and units

The slopes are $s = (\bar{x} - \bar{x}_{ref})/u$ for x and the same for y, with $\bar{x}_{ref}$ the centroid of the unaberrated spot of each subaperture (this removes the offsets of partially illuminated subapertures and of the window centre). With noise, the reference is the average of `n_reference_frames` frames (residual noise = single-frame noise divided by $\sqrt n$).

With `unit_in_rad=False`, $u = 1$ and slopes are in px. With `unit_in_rad=True`, $u$ (px/rad) is the slope of a linear fit of the mean measured x slope against the amplitude of a tilt phase ($a \cdot \mathrm{Tip}$, $a=-2,\dots,2$), where Tip is a ramp from $-\pi$ to $\pi$ over the full pupil grid and the abscissa is $a\,\mathrm{std}(\mathrm{Tip})$ with the standard deviation taken over the whole square grid ($\pi/\sqrt{3}$). The output unit is therefore "rad rms of the tilt over the square pupil grid", consistent for tip and tilt but not an rms over the circular pupil.

### Limitations

- Geometrical optics of the lenslet is not modelled: each spot is the diffraction pattern of the subaperture, with no lenslet-array aberrations, no crosstalk between neighbouring spots (the guard pixels separate them) and no pixel-MTF or charge diffusion.
- The phase within a subaperture is piecewise-bilinear interpolation of the pupil-resolution map: the spatial scale of the aberrations should be resolved by the pupil sampling.
- Only natural (point) sources are supported; see [Correlating SH WFS](correlating_shwfs.md) for the extended Sun.

### References

- Rousset, G. (1999), "Wave-front sensors", in Roddier (ed.), *Adaptive Optics in Astronomy*, Cambridge University Press.
- Soummer, R. et al. (2007), "Fast computation of Lyot-style coronagraph propagation", Opt. Express 15, 15935.
- Goodman, J. W., *Introduction to Fourier Optics*.

::: SAOS.ShackHartmann.ShackHartmann
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true