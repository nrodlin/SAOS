# Data files

The solar continuum images used by `ExtendedSource` (500 nm and 680 nm) are published in Zenodo,
[doi:10.5281/zenodo.23821204](https://doi.org/10.5281/zenodo.23821204) (CC BY 4.0), and are not part of the package.
`SAOS.data` keeps them in a local data directory (`$SAOS_DATA_DIR`, by default `~/.cache/SAOS`) and downloads a file the
first time it is needed, verifying its MD5 checksum. `saos-download-data` fetches all of them in advance.

::: SAOS.data
