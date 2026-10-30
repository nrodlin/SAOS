# `test_packaging.py` — wheel contents and declared dependencies (C.13, nightly)

Source: `tests/test_packaging.py`

Covers the packaging of SAOS (`setup.cfg` / build metadata) rather than any physics class. A wheel is built from the repository (`pip wheel --no-deps --no-build-isolation`, module-scoped fixture) and inspected as a zip file. The check is independent of the packaging metadata: source packages are found by walking `SAOS/`, and third-party imports are read from the source with `ast` (excluding `sys.stdlib_module_names` and `SAOS`), with the import-name to distribution mapping `cv2` to `opencv-contrib-python` and `zmq` to `pyzmq`. No statistics or theory are involved. Whole module `nightly` (it builds a wheel).

## `test_wheel_contains_every_subpackage_and_the_images`
- **Module:** package data and `packages` declaration of the SAOS distribution.
- **Tier:** nightly.
- **What it checks:** every directory of `SAOS/` containing Python modules appears in the wheel as a package (a directory with `__init__.py`), and the solar images `SAOS/images/cont_500nm.h5` and `SAOS/images/cont_680nm.h5` are in the wheel.
- **Tolerance:** exact (zero missing subpackages, both images present).
- **References:** none.

## `test_wheel_requirements_match_the_imports`
- **Module:** wheel `METADATA` (`Requires-Dist`, `Requires-Python`).
- **Tier:** nightly.
- **What it checks:** the set of declared requirements equals the set of third-party modules imported by the package (nothing imported but undeclared, nothing declared but never imported), and `Requires-Python` is exactly `>=3.10`.
- **Tolerance:** exact (zero differences in both directions; exact string for the Python requirement).
- **References:** none.
