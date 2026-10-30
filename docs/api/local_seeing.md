# Local Seeing

## Theory and implementation

`LocalSeeing` replays a pre-computed sequence of OPD maps (m) that represents an aberration common to all lines of sight and conjugated to the pupil (for instance dome or mirror seeing not included in the layered atmosphere). The input H5 contains `opd/data`, an array `[row, col, time]` in **metres** with the attribute `Ts` (sampling time, which must equal the telescope sampling time exactly, otherwise `ValueError`). SAOS does not generate the maps: they are an external input with their own statistics and dynamics.

At iteration $k$ the map $\mathrm{OPD}_k$ is returned and added, in the light path, to the atmospheric, vibration and DM OPD:

$$
\mathrm{OPD}_{wfs} = P\,\big(\mathrm{OPD}_{atm} + \mathrm{OPD}_{vib} + \mathrm{OPD}_{ls} + \mathrm{OPD}_{DM}\big),
$$

with $P$ the pupil mask applied once by the light path. The map is the same for every source and field direction (it is treated as pupil-conjugated, i.e. a ground-layer or in-dome contribution). It is not applied during interaction-matrix measurement. If the series is shorter than the simulation, the index wraps around with a warning.

If the first dimension of the array differs from the pupil resolution the maps are resampled to the pupil size with bilinear interpolation (`cv2.resize`) and multiplied by the pupil mask (a warning notes possible artefacts). Maps with the right resolution are used unchanged, without masking. Only the first dimension is compared, so square maps are assumed. The returned array is a view of the stored data and must not be modified in place.

::: SAOS.LocalSeeing.LocalSeeing
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
