# OOPAO: Object-Oriented Python Adaptive Optics

Welcome to the documentation for **OOPAO**, a modular, object-oriented simulation framework for adaptive optics systems.

This package allows you to model atmospheric turbulence, deformable mirrors, wavefront sensors, and extended or point sources — all using a composable LightPath abstraction.

---

## 📦 Core Components

- **[Atmosphere](atmosphere.md)** – Multi-layer turbulence modeling with von Kármán statistics.
- **[Source](source.md)** – Point sources with spectral/magnitude properties.
- **[Extended Source](extended_source.md)** – 2D image-based sources like the Sun.
- **[Deformable Mirror](deformable_mirror.md)** – Zonal and modal DM simulation with misregistration support.
- **[Shack-Hartmann WFS](shwfs.md)** – Diffractive and geometric SH-WFS with LGS spot elongation support.
- **[Interaction Matrix Handler](interaction_matrix.md)** – Tools for modal basis generation and interaction matrix acquisition.
- **[MisRegistration](misregistration.md)** – Parametric geometric transformations (rotation, shift, scaling).
- **[Light Path](light_path.md)** – High-level abstraction to simulate AO system propagation.
- **[Sharepoint](sharepoint.md)** – ZeroMQ publisher to stream AO simulation data.

---
