"""
Packaging of SAOS: the wheel built from the repository must contain every subpackage and the solar images, and its
declared dependencies must be exactly the third-party modules imported by the package (independent check: the imports are
read from the source with ast, not from setup.cfg).
"""
import ast
import os
import subprocess
import sys
import sysconfig
import zipfile

import pytest

pytestmark = pytest.mark.nightly

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'SAOS')
# import name -> distribution name, where they differ
DIST = {'cv2': 'opencv-contrib-python', 'zmq': 'pyzmq'}


def source_packages():
    """Every directory of SAOS/ holding Python modules (excluding caches)."""
    out = set()
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        if any(f.endswith('.py') for f in files):
            out.add(os.path.relpath(root, REPO).replace(os.sep, '.'))
    return out


def third_party_imports():
    """Top-level modules imported anywhere in SAOS that are neither SAOS nor the standard library."""
    stdlib = set(sys.stdlib_module_names)
    mods = set()
    for root, _, files in os.walk(PKG):
        for f in files:
            if not f.endswith('.py'):
                continue
            tree = ast.parse(open(os.path.join(root, f), encoding='utf-8').read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods.add(node.module.split('.')[0])
    return {m for m in mods if m not in stdlib and m != 'SAOS'}


@pytest.fixture(scope='module')
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp('wheel')
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', REPO, '--no-deps', '--no-build-isolation', '-q', '-w', str(out)],
                   check=True, cwd=str(out))
    path = [os.path.join(out, f) for f in os.listdir(out) if f.endswith('.whl')][0]
    return zipfile.ZipFile(path)


def test_wheel_contains_every_subpackage_and_the_images(wheel, report):
    names = wheel.namelist()
    in_wheel = {os.path.dirname(n).replace('/', '.') for n in names if n.endswith('__init__.py')}
    expected = source_packages()
    report.note('subpackages in the source: ' + ', '.join(sorted(expected)))
    report.check('subpackages of the source missing from the wheel', len(expected - in_wheel), expected=0, atol=0)
    for img in ('cont_500nm.h5', 'cont_680nm.h5'):
        report.check(f'SAOS/images/{img} in the wheel', float(f'SAOS/images/{img}' in names), expected=1.0, atol=0)
    report.assert_ok()


def test_wheel_requirements_match_the_imports(wheel, report):
    meta = wheel.read([n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')][0]).decode()
    declared = {l.split(':', 1)[1].strip().split(';')[0].split()[0].lower() for l in meta.splitlines() if l.startswith('Requires-Dist:')}
    imported = {DIST.get(m, m).lower() for m in third_party_imports()}
    report.note('declared: ' + ', '.join(sorted(declared)))
    report.note('imported: ' + ', '.join(sorted(imported)))
    report.check('imported but not declared', len(imported - declared), expected=0, atol=0)
    report.check('declared but not imported', len(declared - imported), expected=0, atol=0)
    python_req = [l for l in meta.splitlines() if l.startswith('Requires-Python:')]
    report.check('Requires-Python is >=3.10', float(python_req == ['Requires-Python: >=3.10']), expected=1.0, atol=0)
    report.assert_ok()
