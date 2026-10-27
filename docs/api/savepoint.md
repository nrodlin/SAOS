# Savepoint

Saves the simulation data and metrics to an HDF5 file, one group per light path (`LightPath_<k>`), atmosphere
(`Atmosphere_<i>`) and deformable mirror (`DeformableMirror_<i>`).

## Metadata

Every file has a group `/metadata`, written at the first `save()`:

| Location | Content |
|----------|---------|
| `/metadata` attributes | `saos_version`, `saos_git_commit`, `saos_git_dirty` (uncommitted changes in the SAOS repository; empty when SAOS is installed from a wheel), Python and library versions (`numpy`, `scipy`, `h5py`, `torch`), `gil_enabled`, `cuda_device`, `hostname`, `user`, `platform`, `command_line`, `cwd`, `created_utc`, `savepoint_settings` (JSON), and `last_iteration` / `last_saved_utc`, updated at every save (an interrupted run can be detected) |
| `/metadata/user` | User metadata: `Savepoint(..., metadata={...})` and `Savepoint.add_metadata(**kw)` |
| `/metadata/telescope`, `/metadata/atmosphere`, `/metadata/dm_<i>` | Configuration of the modules of the first light path that has them, whatever is saved |
| `LightPath_<k>`, `Atmosphere_<i>`, `DeformableMirror_<i>` attributes | Configuration of the module: source (`src_*`), WFS (`wfs_*`), science camera (`sci_*`), delay, DM altitudes... |

Values are scalars, small numeric arrays or JSON strings (dicts, mixed lists).

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
