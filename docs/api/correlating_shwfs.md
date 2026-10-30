# Correlating Shack-Hartmann WFS

## Theory and implementation

`CorrelatingShackHartmann` models a Shack-Hartmann sensor observing the extended, low-contrast Sun. Each subaperture forms an image of a solar patch, and the shift of that image with respect to a reference is measured by cross-correlation (Michau, Rousset and Fontanella 1993; Rousset 1999; Löfdahl 2010). Conventions as in [Shack-Hartmann WFS](shwfs.md): sky x is the column and y the row, detector quantities in px, phases in rad. The subaperture resampling, the cell-centred sampling, the valid-subaperture selection and the MFT are the same as in `ShackHartmann`.

### Image of an extended source

Isoplanatism is not assumed across the field. The source (`ExtendedSource`) provides $N_{sd}\times N_{sd}$ sub-directions (`nSubDirs`), each with its own solar patch $O_i$ and its own pupil phase $\phi_i$ (when a 2D phase is given, for example the DM only, it is repeated for all sub-directions). For every valid subaperture and sub-direction the point-spread function is computed by MFT on the `npix_sun` grid of the patch, centred at pixel `npix_sun//2` and normalised to the energy of the subaperture (1 if fully illuminated, its illuminated area otherwise):

$$
\mathrm{PSF}_i = \frac{|\mathcal{M}[a\,e^{i\phi_i}]|^2}{\sum |\mathcal{M}[a\,e^{i\phi_i}]|^2}\; \langle a^2\rangle .
$$

The patch $O_i$ is resampled to the WFS plate scale (area averaging) and convolved with the PSF through FFTs, moving the PSF centre to the origin so that the circular convolution does not shift the patch:

$$
I_i = \left|\mathcal{F}^{-1}\!\left[\mathcal{F}[O_i]\;\mathcal{F}[\mathrm{PSF}_i]\right]\right| ,
$$

and each sub-direction image is rescaled so that its total energy over all subapertures equals that of $O_i$ (one common factor per sub-direction, preserving the relative flux between subapertures). The sub-directions are blended with separable triangular (tent) windows of width equal to the sub-direction size, placed with a stride of half a window (50 % overlap), each divided by the sum of all windows placed so that the weights form a partition of unity (also at the borders). The mosaic is cropped to the subaperture window. The circular convolution assumes the patch is large enough compared with the PSF wings: the source patch includes a margin (`subDir_margin`) and padding (`patch_padding`) that are discarded.

The photon number is $N_{ph} = \mathrm{round}\big(\Phi\,A_{eff}\,T_s\,\eta_{light}\,(\mathrm{FoV}_{WFS}/\mathrm{FoV}_{src})^2\big)$: the flux of the Sun is per unit of sky area, so it scales with the square of the field of view that the WFS actually observes.

### Reference and pseudo-reference

The correlation reference ("pseudo-reference") is the unaberrated image of one subaperture, the first fully illuminated one (`pseudoref_index`), acquired in the same conditions as the measurements: with noise, it is the average of `n_reference_frames` frames. Every subaperture is correlated with this single image, so the measured peak position of an unaberrated subaperture is not zero when its patch differs from the reference one. For that reason reference slopes are measured at null phase (again averaging `n_reference_frames` frames) and subtracted. `LightPath` acquires both at initialisation (`initialize_wfs`) for solar sources.

### Cross-correlation

With subaperture image $S$ and reference $R$, the circular cross-correlation is computed with FFTs,

$$
C = \left|\mathcal{F}^{-1}\!\left[\mathcal{F}[S]\;\overline{\mathcal{F}[R]}\right]\right| ,
$$

shifted so that zero displacement is at pixel `npix_subap//2`. If $S(\mathbf{x}) = R(\mathbf{x}-\mathbf{s})$ the peak is at $+\mathbf{s}$ (positive column shift: larger column index). The correlation is circular over the detector window, so shifts must stay well below half the window, and the modulus (real inputs give a real result, so this rectifies negative lobes to positive) is taken.

### Sub-pixel peak estimation

The peak position is the thresholded CoG of the correlation image, with the same weights as in `ShackHartmann`, $w_p = \max(C_p-T,0)$, but with $T$ the $n_b$-th largest value (default `use_brightest` $=9$), so the $n_b$-th pixel has zero weight and 8 pixels contribute. The slopes are $(\bar{x}-\bar{x}_{ref})/u$ in px or rad (for `unit_in_rad=True`, $u$ is calibrated like in `ShackHartmann` with tip/tilt ramps measured against the unaberrated pseudo-reference and reference slopes).

!!! note "Known gain below 1"
    The thresholded CoG over a few pixels around the correlation peak is not an unbiased estimator of the peak position: it is attracted towards the centre of the selected pixels and the measured shift is a fraction of the true one. Measured with this code the gain is between about 0.75 and 0.98 depending on the amplitude of the shift (it is closer to 1 for small amplitudes and falls for larger ones). The `unit_in_rad` calibration fits a single gain over its calibration amplitudes, so the residual gain at other amplitudes is not corrected. In closed loop the loop gain is scaled by this factor. The NGS `ShackHartmann` with all pixels does not have this issue.

### Limitations

- Isoplanatic patches are blended, not propagated through a full volume: the number of sub-directions limits the angular resolution of the anisoplanatism.
- The reference is one subaperture (spatial sampling of a different part of the Sun), as in operation.
- The detector window truncates the object: contrast, granulation structure and edge effects all depend on the field and on the margin.
- Lenslet aberrations, crosstalk and pixel-MTF are not modelled.

### References

- Michau, V., Rousset, G. and Fontanella, J. C. (1993), "Wavefront sensing from extended sources", in *Real Time and Post Facto Solar Image Correction*, NSO/SP Workshop.
- Rousset, G. (1999), "Wave-front sensors", in Roddier (ed.), *Adaptive Optics in Astronomy*, Cambridge University Press.
- Löfdahl, M. G. (2010), "Evaluation of image-shift measurement algorithms for solar Shack-Hartmann wavefront sensors", A&A 524, A90.

::: SAOS.CorrelatingShackHartmann.CorrelatingShackHartmann
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
