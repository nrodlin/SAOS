"""
External data files of SAOS (solar images), downloaded on demand from Zenodo.

The solar continuum images used by ExtendedSource are not distributed with the package. They are published in
Zenodo (doi:10.5281/zenodo.23821204, CC BY 4.0) and stored in a local data directory:

- ``$SAOS_DATA_DIR`` when the environment variable is set;
- otherwise ``$XDG_CACHE_HOME/SAOS`` (``~/.cache/SAOS`` by default).

``get_data_file(name)`` returns the path of a file, downloading it (and checking its MD5 checksum) when it is missing.
All the files can be fetched in advance with the command ``saos-download-data`` (or ``python -m SAOS.data``).
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
import urllib.request

ZENODO_DOI = '10.5281/zenodo.23821204'
ZENODO_RECORD = 'https://zenodo.org/api/records/23821204/files'

# File name -> (size [bytes], MD5 checksum)
REGISTRY = {
    'cont_500nm.h5': (240562900, 'dc2d763dbba1ff68c919a2b35749ff72'),
    'cont_680nm.h5': (130510576, '93fcf06b2f91caf1065c9cef6d45ae3b'),
}

_CHUNK = 1 << 20


def data_dir():
    """Local directory of the SAOS data files (created if needed)."""
    path = os.environ.get('SAOS_DATA_DIR') or os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'SAOS')
    os.makedirs(path, exist_ok=True)
    return path


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(_CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def _download(name, path, logger):
    size, md5 = REGISTRY[name]
    url = f'{ZENODO_RECORD}/{name}/content'
    tmp = f'{path}.part{os.getpid()}'
    logger.info(f'SAOS data - downloading {name} ({size / 1e6:.0f} MB) from doi:{ZENODO_DOI} to {path}')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, 'wb') as f:
            shutil.copyfileobj(response, f, _CHUNK)
        if _md5(tmp) != md5:
            raise IOError(f'SAOS data - checksum mismatch for {name} downloaded from {url}')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def get_data_file(name, logger=None):
    """
    Path of a SAOS data file, downloaded from Zenodo when it is not in the data directory.

    Parameters
    ----------
    name : str
        File name, one of REGISTRY.
    logger : logging.Logger, optional
        Logger for the download messages.

    Returns
    -------
    str
        Absolute path of the file.
    """
    if name not in REGISTRY:
        raise KeyError(f'SAOS data - unknown data file {name}; available: {sorted(REGISTRY)}')
    logger = logger or logging.getLogger(__name__)
    path = os.path.join(data_dir(), name)
    if not os.path.isfile(path) or os.path.getsize(path) != REGISTRY[name][0]:
        _download(name, path, logger)
    return path


def download_all(verify=False, logger=None):
    """
    Download every SAOS data file that is missing from the data directory.

    Parameters
    ----------
    verify : bool
        Also check the MD5 checksum of the files already present (re-downloading the ones that do not match).
    logger : logging.Logger, optional
        Logger for the messages.

    Returns
    -------
    list of str
        Paths of the files.
    """
    logger = logger or logging.getLogger(__name__)
    paths = []
    for name, (_, md5) in REGISTRY.items():
        path = os.path.join(data_dir(), name)
        if verify and os.path.isfile(path) and _md5(path) != md5:
            logger.warning(f'SAOS data - {name} does not match its checksum, downloading it again')
            os.remove(path)
        paths.append(get_data_file(name, logger))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(prog='saos-download-data', description=f'Download the SAOS data files (doi:{ZENODO_DOI}) to the data '
                                     'directory ($SAOS_DATA_DIR, default ~/.cache/SAOS)')
    parser.add_argument('--verify', action='store_true', help='Check the MD5 checksum of the files already present')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    for path in download_all(verify=args.verify):
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
