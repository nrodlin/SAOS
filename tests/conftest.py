"""
Shared fixtures and result collection of the SAOS validation suite.

Tiers (markers, see pyproject.toml): fast (T1), nightly (T2), heavy (T3), gpu. Known SAOS bugs are labelled with
citools.known_issue; those tests fail until the bug is fixed. Every test can record metrics and figures with the `report` fixture; results go to
$SAOS_CI_OUT (default ~/simulations/results/ci/dev, never inside the repository): one folder per test plus
results.json, which ci/make_report.py turns into report.html.

SAOS objects get the external `logger` fixture: with logger=None SAOS starts a QueueListener per object whose
__del__ can block the interpreter at exit.
"""
import datetime
import glob
import json
import logging
import os
import platform
import subprocess
import time
import zlib

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import yaml

from citools import Report, _jsonable

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
OUT_DIR = os.environ.get('SAOS_CI_OUT', os.path.expanduser('~/simulations/results/ci/dev'))

_REPORTS = {}
_RESULTS = {}
_T0 = time.time()


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if not has_gpu:
        skip = pytest.mark.skip(reason='no CUDA device visible')
        for item in items:
            if 'gpu' in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope='session')
def logger():
    lg = logging.getLogger('saos-ci')
    lg.setLevel(logging.WARNING)
    return lg


@pytest.fixture(scope='session')
def tolerances():
    """Tolerances per area: tests/tolerances/<area>.yaml -> tolerances['<area>'][...]."""
    out = {}
    for path in sorted(glob.glob(os.path.join(TESTS_DIR, 'tolerances', '*.yaml'))):
        with open(path) as f:
            out[os.path.splitext(os.path.basename(path))[0]] = yaml.safe_load(f) or {}
    return out


@pytest.fixture
def seed(request):
    """Deterministic seed per test (stable across runs and machines); SAOS_CI_SEED_OFFSET=k gives another seed set, used to
    measure the estimator spread when calibrating tolerances."""
    return (zlib.crc32(request.node.nodeid.encode()) + 7919 * int(os.environ.get('SAOS_CI_SEED_OFFSET', 0))) % (2 ** 31)


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def report(request):
    rep = Report(request.node.nodeid, OUT_DIR)
    _REPORTS[request.node.nodeid] = rep
    yield rep
    rep.dump()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    ki = item.get_closest_marker('known_issue')
    r = _RESULTS.setdefault(item.nodeid, dict(
        nodeid=item.nodeid, module=item.module.__name__ if item.module else '', name=item.name,
        doc=' '.join((getattr(item, 'function', None).__doc__ or '').split()) if getattr(item, 'function', None) else '',
        markers=sorted({m.name for m in item.iter_markers()} - {'parametrize', 'usefixtures'}),
        known_issue=(f"[{ki.kwargs['issue_id']}] {ki.kwargs['description']}" + (f" ({ki.kwargs['location']})" if ki.kwargs.get('location') else '')) if ki else None, duration=0.0, outcome='passed', message=''))
    r['duration'] += rep.duration
    if rep.skipped and hasattr(rep, 'wasxfail'):
        r['outcome'] = 'xfailed'
    elif rep.skipped:
        if r['outcome'] == 'passed':
            r['outcome'] = 'skipped'
            r['message'] = str(rep.longrepr[-1]) if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
    elif rep.failed:
        text = rep.longreprtext or ''
        if rep.when == 'call' and 'XPASS(strict)' in text:
            r['outcome'] = 'xpassed'          # known issue no longer reproduced: remove the mark
        else:
            r['outcome'] = 'failed' if rep.when == 'call' else 'error'
        r['message'] = '\n'.join(text.splitlines()[-25:])


def _git(*args):
    try:
        return subprocess.run(['git', '-C', REPO_DIR, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    try:
        import torch
        torch_info = dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
                          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    except ImportError:
        torch_info = dict(torch='', cuda=False, gpu='')
    tests = []
    for nodeid, r in _RESULTS.items():
        rep = _REPORTS.get(nodeid)
        tests.append({**r, **(rep.to_dict() if rep else dict(metrics=[], figures=[], notes=[]))})
    info = dict(date=datetime.datetime.now().isoformat(timespec='seconds'), host=platform.node(),
                python=platform.python_version(), numpy=np.__version__, **torch_info,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                sha=_git('rev-parse', '--short', 'HEAD'), branch=_git('rev-parse', '--abbrev-ref', 'HEAD'),
                dirty=bool(_git('status', '--porcelain')), tier=os.environ.get('SAOS_CI_TIER', 'dev'),
                markexpr=session.config.option.markexpr, duration=time.time() - _T0, exitstatus=int(exitstatus))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(_jsonable(dict(info=info, tests=tests)), f, indent=1)
