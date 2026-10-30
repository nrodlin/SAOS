"""
SAOS.data: location of the data directory, reuse of the files already present, download with checksum verification and
atomic placement. The download is exercised against a local file:// record (no network): the expected behaviour is that
of the documented contract (file in $SAOS_DATA_DIR, MD5 equal to the registry, no partial file left on failure).
"""
import hashlib
import os

import pytest

import SAOS.data as data

pytestmark = pytest.mark.fast


@pytest.fixture
def record(tmp_path, monkeypatch):
    """Local record with one file, a registry entry for it and an empty data directory."""
    src = tmp_path / 'record'
    (src / 'img.h5').mkdir(parents=True)
    payload = os.urandom(3 * (1 << 20) + 17)
    (src / 'img.h5' / 'content').write_bytes(payload)
    monkeypatch.setattr(data, 'ZENODO_RECORD', src.as_uri())
    monkeypatch.setattr(data, 'REGISTRY', {'img.h5': (len(payload), hashlib.md5(payload).hexdigest())})
    monkeypatch.setenv('SAOS_DATA_DIR', str(tmp_path / 'data'))
    return payload, tmp_path / 'data'


def test_data_dir_from_environment(tmp_path, monkeypatch, report):
    monkeypatch.setenv('SAOS_DATA_DIR', str(tmp_path / 'd'))
    report.check('data_dir() is $SAOS_DATA_DIR and exists', float(data.data_dir() == str(tmp_path / 'd') and (tmp_path / 'd').is_dir()),
                 expected=1.0, atol=0)
    monkeypatch.delenv('SAOS_DATA_DIR')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    report.check('data_dir() is $XDG_CACHE_HOME/SAOS without SAOS_DATA_DIR', float(data.data_dir() == str(tmp_path / 'cache' / 'SAOS')),
                 expected=1.0, atol=0)
    report.assert_ok()


def test_registry_matches_the_zenodo_record(report):
    report.check('registry entries (cont_500nm.h5, cont_680nm.h5)', float(sorted(data.REGISTRY) == ['cont_500nm.h5', 'cont_680nm.h5']),
                 expected=1.0, atol=0)
    report.check('every MD5 has 32 hex digits', float(all(len(m) == 32 and int(m, 16) >= 0 for _, m in data.REGISTRY.values())),
                 expected=1.0, atol=0)
    report.assert_ok()


def test_download_verifies_and_reuses(record, monkeypatch, report):
    payload, ddir = record
    path = data.get_data_file('img.h5')
    report.check('downloaded file in the data directory', float(path == str(ddir / 'img.h5')), expected=1.0, atol=0)
    report.check('downloaded bytes equal to the record', float(open(path, 'rb').read() == payload), expected=1.0, atol=0)
    report.check('no partial files left', float(len(os.listdir(ddir))), expected=1.0, atol=0)

    def no_download(*args, **kwargs):
        raise AssertionError('download not expected')
    monkeypatch.setattr(data, '_download', no_download)
    report.check('second call reuses the file', float(data.get_data_file('img.h5') == path), expected=1.0, atol=0)
    report.assert_ok()


def test_checksum_mismatch_raises_and_leaves_nothing(record, monkeypatch, report):
    payload, ddir = record
    monkeypatch.setattr(data, 'REGISTRY', {'img.h5': (len(payload), '0' * 32)})
    with pytest.raises(IOError):
        data.get_data_file('img.h5')
    report.check('files in the data directory after a failed download', float(len(os.listdir(ddir))), expected=0.0, atol=0)
    with pytest.raises(KeyError):
        data.get_data_file('unknown.h5')
    report.assert_ok()


def test_truncated_file_is_downloaded_again(record, report):
    payload, ddir = record
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / 'img.h5').write_bytes(payload[:100])
    path = data.get_data_file('img.h5')
    report.check('file of the wrong size replaced by the record', float(open(path, 'rb').read() == payload), expected=1.0, atol=0)
    report.assert_ok()
