# Savepoint

Saves the simulation data and metrics to an HDF5 file, one group per light path (`LightPath_<k>`), atmosphere
(`Atmosphere_<i>`) and deformable mirror (`DeformableMirror_<i>`).

## Theory and implementation

### What is saved and when

`Savepoint.save(objects, iteration)` is called by the simulation loop with the list of light paths, atmospheres or DMs. Every quantity has an integer flag: 0 disables it and $n>0$ saves it with decimation $n$. The group and datasets of a quantity are created at the first call of `save` (whatever the decimation), and then appended every time $(\mathrm{iteration}+1)\bmod n=0$, so with $n>1$ the first sample is that of the first call and the next ones are at iterations $n-1,\,2n-1,\dots$. Each quantity is a group with `iteration` (the iteration of each sample), its statistics, and `data` unless `only_metrics=True`.

| Flag | Group (in `LightPath_<k>`) | Content |
|------|----------------------------|---------|
| `atm`, `dm` | `Atmosphere_<i>/layer_<l>`, `DeformableMirror_<i>` | Phase screens per layer; DM commands of the valid actuators |
| `atm_per_dir` | `atm_opd`, `atm_phase` | Atmosphere OPD [m] and phase [rad] projected to the source |
| `dm_per_dir` | `dm_opd_<i>`, `dm_phase_<i>` | OPD and phase of each DM towards the source |
| `slopes` | `slopes_1D`, `slopes_2D` | WFS slopes (x block, then y block) |
| `error` | `error/slopes_1D` | `get_wavefront_error()`: the slopes seen by the controller, after the delay |
| `wfs`, `sci` | `wfs_opd/phase`, `sci_opd/phase` | OPD and phase at the WFS and at the science camera |
| `wfs_frame` | `wfs_frame` | WFS detector frame |
| `sci_frame` | `sci_frame_shortExp`, `sci_frame_longExp` | Short exposure (each frame) and long exposure frames |

The flags can be scalars, lists (one value per light path) or dicts keyed by light-path index with an optional `default`; `sci_frame` also accepts `{'short': n, 'long': m}` to decimate the short and long exposures independently (see [Light Path](light_path.md) for how the frames are accumulated). `only_metrics` can be resolved per light path in the same way. The file is opened in append mode at each call.

### Metrics

For OPD and phase maps the statistics are computed over the pupil pixels $\mathcal{P}$ of each sample: `min`, `max`, `mean`, `std` and $\mathrm{rms}=\sqrt{\langle\phi^2\rangle_{\mathcal P}}$ (so rms includes the mean, `std` does not). Layer screens use all the pixels and DM commands the valid actuators. Slopes statistics (`min`, `max`, `mean`, `std`, `rms`) are given as a pair (x block, y block) per sample; `slopes_2D` only has `min` and `max`. The units are those of the quantity (m, rad, px or rad for slopes).

**Strehl** (`strehl`, point sources). For a camera frame $I$ and the ideal (diffraction-limited) PSF $I_0$ of the camera at the same wavelength,

$$
\mathrm{SR}=\frac{\max I\,/\sum I}{\max I_0\,/\sum I_0},
$$

where the sums are over the pixels of the frame, i.e. the peak is normalised by the energy recorded in the camera field (the observational definition). With small fields the energy of the aberrated halo that leaves the field is not counted, which raises SR slightly. The peaks $\max I$ are the maxima of the band-limited (Fourier, sinc) interpolation of the image on a grid of 1/32 pixel within $\pm1$ pixel of the brightest pixel (`peak_subpixel`; never below the brightest pixel value), so that SR does not depend on where the peak falls with respect to the pixel grid. The interpolation is exact for frames sampled at Nyquist or better ($\lambda/D\ge2$ px) and an estimate otherwise. A frame with zero total energy gives SR $=0$.

**Solar sources** (`src.tag == 'sun'`). Two metrics on the frame $I$:

- `contrast` $=\sigma_I/\bar I$, the standard deviation over the mean of the frame (0 if $\bar I=0$).
- `MFGS` (median-filtered gradient similarity): the frame is normalised by its mean, $\tilde I=I/\bar I$, and compared with its $3\times3$ median filtered version $M$ (reflected borders) through their gradient magnitudes $g_I=\sqrt{(K_x*\tilde I)^2+(K_y*\tilde I)^2}$ and $g_M$ (the same with $M$), with the Scharr kernel $K_x=\begin{pmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{pmatrix}$ and $K_y$ its transpose (rotation by $90^\circ$). The similarity map is

$$
G=\frac{2\,g_I\,g_M+C}{g_I^2+g_M^2+C},\qquad \mathrm{MFGS}=\langle G\rangle,\quad C=10^{-10}.
$$

MFGS is $1$ for a frame that is not changed by the median filter and decreases as the high-frequency content (noise or fine structure) that the filter removes grows; the normalisation makes it independent of the frame units. It is a relative measure: it penalises both noise and genuine small-scale structure.

### Long and short exposures

The metrics of `sci_frame_shortExp` and `sci_frame_longExp` are computed on each saved frame with the formulas above: the long exposure frame is the accumulation of the short exposures over the integration time, so its SR is the long exposure Strehl. The two series have independent decimation (`{'short': n, 'long': m}`), and the long exposure group is created when the first long frame exists. `only_long_exp` is stored in the settings metadata but is not used by `save`.

## Metadata

See the section [Metadata](#metadata) below: `/metadata` is written at the first `save` with the version and git commit of SAOS (and whether the repository had uncommitted changes), library versions, host and command line, and `last_iteration` is refreshed at every call, so an interrupted run can be detected from the file.

## Metadata

Every file has a group `/metadata`, written at the first `save()`:

| Location | Content |
|----------|---------|
| `/metadata` attributes | `saos_version`, `saos_git_commit`, `saos_git_dirty` (uncommitted changes in the SAOS repository; empty when SAOS is installed from a wheel), Python and library versions (`numpy`, `scipy`, `h5py`, `torch`), `gil_enabled`, `cuda_device`, `hostname`, `user`, `platform`, `command_line`, `cwd`, `created_utc`, `savepoint_settings` (JSON), and `last_iteration` / `last_saved_utc`, updated at every save (an interrupted run can be detected) |
| `/metadata/user` | User metadata: `Savepoint(..., metadata={...})` and `Savepoint.add_metadata(**kw)` |
| `/metadata/telescope`, `/metadata/atmosphere`, `/metadata/dm_<i>` | Configuration of the modules of the first light path that has them, whatever is saved |
| `LightPath_<k>`, `Atmosphere_<i>`, `DeformableMirror_<i>` attributes | Configuration of the module: source (`src_*`), WFS (`wfs_*`), science camera (`sci_*`), delay, DM altitudes... |

Values are scalars, numeric arrays (up to 10000 elements) or JSON strings (dicts, mixed lists). They are attributes, except
those above 16 kB, which are stored as datasets of the same name in the same group (HDF5 limits the attributes of an
object to 64 kB).

```python
sp = Savepoint(file_path='res.h5', sci_frame={'long': 1}, only_metrics=True,
               metadata=dict(case=1, draw=2, spider='noSpider'))
sp.add_metadata(controller=dict(type='leaky', gain=[0.25, 0.2, 0.2]))

import h5py, json
with h5py.File('res.h5') as f:
    print(f['metadata'].attrs['saos_version'], f['metadata'].attrs['saos_git_commit'])
    print(json.loads(f['metadata/user'].attrs['controller']))
```

::: SAOS.Savepoint.Savepoint
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true

::: SAOS.Savepoint.module_metadata

::: SAOS.Savepoint.run_info
