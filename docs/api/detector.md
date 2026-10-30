# Detector

## Theory and implementation

`Detector.integrate` converts a noise-free intensity frame into a digital frame in counts (DN). It is used by the WFS and the science camera. If `noiseFlag` is False the input frame is returned unchanged (float, not scaled to photons, no QE, no quantisation); the following applies when it is True. All the random draws use a `numpy` generator; each `LightPath` connected to the detector has its own stream (`register_stream`), seeded from the detector seed, so the noise does not depend on thread scheduling.

### Signal chain

For a noise-free frame $f$ and $N_{ph}$ photons per integration:

1. Normalisation and scaling: $P = \mathrm{round}\!\left(N_{ph}\,f/\sum f\right)$ photons per pixel. (The expected photon count is rounded to an integer before the Poisson draw, so pixels expecting fewer than 0.5 photons receive none.)
2. Shot noise and quantum efficiency: the detected electrons are a Poisson draw of mean $\eta P$ (the binomial thinning of a Poisson stream is Poisson), $n_e \sim \mathrm{Poisson}(\eta P)$. Without shot noise, $n_e = \eta P$.
3. Dark current: $n_e \mathrel{+}= \mathrm{Poisson}(i_d\,T_{int})$, with $i_d$ in e-/px/s and $T_{int}$ the integration time.
4. Full well: $n_e \leftarrow \min(n_e, N_{FW})$ (a warning is logged when saturated).
5. EMCCD only: multiplication gain $n_e \leftarrow g\,n_e$ before the read-out.
6. Read-out noise: $n_e \mathrel{+}= \mathcal{N}(0,\sigma_r^2)$ [e-].
7. CCD/CMOS: $n_e \leftarrow g\,n_e$ (after the read-out, so the read-out noise is also multiplied by the gain).
8. Quantisation: $\mathrm{DN} = \mathrm{round}(n_e/q)$ with the conversion $q = N_{FW}/2^{n_{bits}}$ e-/DN by default (or `quantization_conversion` if given; 1 e-/DN if the full well is infinite), clipped to $[0,\,2^{n_{bits}}-1]$. The quantisation noise is $q/\sqrt{12}$.
9. Dark subtraction: the mean of `darkCalibration` frames of zero input (shot noise disabled) is subtracted and the result is clipped to $\geq 0$. The output is unsigned integer (`uint8`/`uint16`/...) according to $n_{bits}$.

### Conventions and limitations

- There is no bias pedestal: all the negative values are clipped to 0 by convention (after the read-out noise in step 8 and after the dark subtraction in step 9), because unsigned output would otherwise wrap around. This clipping raises the mean of faint pixels, mostly when $\sigma_r$ is comparable to the signal; photon-starved pixels are biased upwards.
- The noise terms are independent and uncorrelated between pixels; there is no fixed-pattern noise, PRNU, dark-current non-uniformity, cross-talk, blooming or non-linearity. The full well is a hard clip.
- The quantum efficiency is a single constant (no wavelength dependence); `gain` is a pure multiplicative factor.
- Defaults of the WFS and science camera correspond to a pco.dimax 3.6 ST at 500 nm (QE 0.64, FWC 60 ke-, 250 e-/px/s, 60 e- rms).

### References

- Janesick, J. R. (2001), *Scientific Charge-Coupled Devices*, SPIE Press.
- Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford University Press.

::: SAOS.Detector.Detector
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
