# `test_data.py` — data directory and download of the solar images (T1)

Source: `tests/test_data.py`

Covers `SAOS.data`. The download is exercised against a local `file://` record holding one random file, with a registry entry built from its size and MD5, so no network is used. Whole module `fast`.

## `test_data_dir_from_environment`
- **Module:** `SAOS.data.data_dir`.
- **What it checks:** the data directory is `$SAOS_DATA_DIR` when set (and is created), and `$XDG_CACHE_HOME/SAOS` otherwise.
- **Tolerance:** exact.

## `test_registry_matches_the_zenodo_record`
- **Module:** `SAOS.data.REGISTRY`.
- **What it checks:** the registry lists `cont_500nm.h5` and `cont_680nm.h5`, each with a 32-digit hexadecimal MD5.
- **Tolerance:** exact.

## `test_download_verifies_and_reuses`
- **Module:** `SAOS.data.get_data_file`.
- **What it checks:** a missing file is downloaded into the data directory with the bytes of the record and no partial file left; a second call returns the same path without downloading.
- **Tolerance:** exact (byte equality).

## `test_checksum_mismatch_raises_and_leaves_nothing`
- **Module:** `SAOS.data.get_data_file`.
- **What it checks:** a download whose MD5 differs from the registry raises `IOError` and leaves the data directory empty; an unknown file name raises `KeyError`.
- **Tolerance:** exact.

## `test_truncated_file_is_downloaded_again`
- **Module:** `SAOS.data.get_data_file`.
- **What it checks:** a file whose size differs from the registry is replaced by the record.
- **Tolerance:** exact (byte equality).
