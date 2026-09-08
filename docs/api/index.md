# API Reference

Complete reference documentation for all SAOS classes and methods.
All docstrings follow the **NumPy** format.

| Module | Class | Description |
|--------|-------|-------------|
| [Telescope](telescope.md) | `Telescope` | Pupil model and sampling parameters |
| [Source](source.md) | `Source` | Point source (guide star / science target) |
| [Extended Source](extended_source.md) | `ExtendedSource` | Extended solar source with sub-directions |
| [Layer](layer.md) | `LayerClass` | Single turbulent atmospheric layer |
| [Atmosphere](atmosphere.md) | `Atmosphere` | Multi-layer Von Kármán atmospheric model |
| [Light Path](light_path.md) | `LightPath` | Single line-of-sight propagation pipeline |
| [Deformable Mirror](deformable_mirror.md) | `DeformableMirror` | DM model (Cartesian, radial & custom geometry) |
| [Misregistration](misregistration.md) | `MisRegistration` | DM-WFS misalignment model |
| [NCPA](ncpa.md) | `NCPA` | Non-Common Path Aberrations |
| [SHWFS](shwfs.md) | `ShackHartmann` | Classical Shack-Hartmann WFS |
| [Correlating SHWFS](correlating_shwfs.md) | `CorrelatingShackHartmann` | Solar correlation-based SH-WFS |
| [Detector](detector.md) | `Detector` | Detector noise and quantization model |
| [Science Camera](science_cam.md) | `ScienceCam` | PSF / solar image science camera |
| [Interaction Matrix](interaction_matrix.md) | `InteractionMatrixHandler` | IM measurement and reconstruction |
| [Controller](controller.md) | `Controller` | Leaky integrator / PI AO controller |
| [Sharepoint](sharepoint.md) | `Sharepoint` | Real-time telemetry via ZeroMQ |
