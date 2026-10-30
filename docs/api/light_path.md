# Light Path

## Theory and implementation

A `LightPath` is one optical train: source, telescope and, optionally, vibrations, local seeing, atmosphere, deformable mirror(s), WFS, NCPA and science camera. It is the unit that the simulation advances in time. OPD is in metres, phase in rad at the wavelength $\lambda$ of the source of the path, $\phi = 2\pi\,\mathrm{OPD}/\lambda$.

### Propagation order

At each call to `propagate` the OPD terms are computed in this order:

1. Vibrations and local seeing for the current iteration index.
2. Atmosphere projected to the source (`Atmosphere.getOPD`).
3. Every DM `get_dm_opd(src)` (the DM is modelled as transmissive, so there is no factor 2 for the go-and-return path).
4. WFS OPD: $\mathrm{OPD}_{wfs} = P\,(\mathrm{OPD}_{vib}+\mathrm{OPD}_{ls}+\mathrm{OPD}_{atm}+\sum_k \mathrm{OPD}_{DM,k})$ with $P$ the pupil. The slopes are then measured.
5. Science OPD: $\mathrm{OPD}_{sci} = P\,(\mathrm{OPD}_{wfs}+\mathrm{OPD}_{NCPA})$: NCPA are seen by the science camera but not by the WFS.

During the measurement of the interaction matrix (`interaction_matrix=True`) the vibrations, local seeing and atmosphere are replaced by zeros, and the science frame is not generated.

### Delay buffer

The slopes measured at iteration $k$ (before the time advance) are stored in column $(k+1) \bmod (\mathrm{delay}+1)$ of a circular buffer of $\mathrm{delay}+1$ columns. `get_wavefront_error`, called after the propagation of iteration $k$ (the counter is then $k+1$), reads column $(k+1-\mathrm{delay}) \bmod (\mathrm{delay}+1)$, which holds the slopes measured at iteration $k-\mathrm{delay}$. With `delay=0` the measurement of the same step is returned; with `delay`$=D$ the controller sees slopes that are $D$ samples old. While the counter is smaller than the delay, a zero vector is returned. The delay is an integer number of samples (rounded); the sub-sample fractions of the frame latency are not modelled here.

### Science frames and long exposure

Short exposures are generated only when the time advances. With decimation $\mathrm{dec}$, the frame is taken at the last iteration of every window of $\mathrm{dec}$ iterations (iterations $\mathrm{dec}-1, 2\,\mathrm{dec}-1,\dots$). A long exposure of integration time $T_{int}$ accumulates

$$
N_{fr} = \max\!\left(1,\ \left\lceil \frac{T_{int}}{\mathrm{dec}\cdot T_s} - 10^{-6} \right\rceil\right)
$$

short frames, counted as an integer (accumulating float times could add one frame when rounding errors appear); a non-integer ratio is rounded up, so the exposure is never shorter than $T_{int}$. The frames are summed with equal weight, renormalised to unit energy and scaled to $\Phi\,T_{int}\,A_{eff}\,\eta_{light}$ photons (with detector noise if enabled; see [Science Camera](science_cam.md)).

### Threading

`parallel_dms=True` computes the DM projections in threads. Each light path registers its own noise stream on the WFS and science detectors, so the noise of one path does not depend on the thread scheduling. LGS sources are not supported.

::: SAOS.LightPath.LightPath
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true