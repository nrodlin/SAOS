"""
Helpers of the SAOS validation suite: metric recording (`Report`), known-issue marks and small numerical tools shared by
the test modules (structure functions, centroids, shifts, FWHM). Physical laws live in theory.py.
"""
import json
import os
import re

import numpy as np
import pytest


def known_issue(issue_id, description, location=''):
    """
    Label of a documented SAOS bug. The test states the physically correct behaviour and FAILS while the bug exists;
    the label only groups the failure under the issue id in the report (no xfail: failures stay visible).
    """
    return pytest.mark.known_issue(issue_id=issue_id, description=description, location=location)


def slug(nodeid):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', nodeid.replace('::', '__'))[:180]


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


class Report:
    """
    Metrics and figures of one test. `check` records a value with its expected value and tolerance and returns whether
    it is within tolerance; `assert_ok` fails the test with every out-of-tolerance metric, so that all metrics are
    recorded before the test stops. Output: <out>/tests/<test>/ (metrics.json and figures).
    """

    def __init__(self, nodeid, out_dir):
        self.nodeid = nodeid
        self.dir = os.path.join(out_dir, 'tests', slug(nodeid))
        self.rel_dir = os.path.join('tests', slug(nodeid))
        self.metrics, self.figures, self.notes = [], [], []

    def check(self, name, value, expected=None, rtol=None, atol=None, lo=None, hi=None, unit='', trend=False):
        """
        Record `name` = value and compare it with the tolerance:
          |value - expected| <= atol + rtol * |expected|     (expected given)
          lo <= value <= hi                                    (bounds given; either may be None)
        Arrays are checked element-wise (all must pass). trend=True adds the metric to the history plots.
        """
        v = np.asarray(value, dtype=float)
        ok = np.ones(v.shape, dtype=bool)
        if expected is not None:
            e = np.asarray(expected, dtype=float)
            ok &= np.abs(v - e) <= np.asarray(0.0 if atol is None else atol, dtype=float) + np.asarray(0.0 if rtol is None else rtol, dtype=float) * np.abs(e)
        if lo is not None:
            ok &= v >= lo
        if hi is not None:
            ok &= v <= hi
        ok = bool(np.all(ok))
        self.metrics.append(_jsonable(dict(name=name, value=v if v.ndim else float(v), expected=expected, rtol=rtol,
                                           atol=atol, lo=lo, hi=hi, unit=unit, ok=ok, trend=trend and v.ndim == 0)))
        return ok

    def metric(self, name, value, unit='', trend=False):
        """Record a value without a pass/fail criterion (context for the report)."""
        v = np.asarray(value, dtype=float)
        self.metrics.append(_jsonable(dict(name=name, value=v if v.ndim else float(v), unit=unit, ok=None,
                                           trend=trend and v.ndim == 0)))

    def note(self, text):
        self.notes.append(str(text))

    def figure(self, fig, name):
        import matplotlib.pyplot as plt
        os.makedirs(self.dir, exist_ok=True)
        fname = re.sub(r'[^A-Za-z0-9_.-]+', '_', name) + '.png'
        fig.savefig(os.path.join(self.dir, fname), dpi=110, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(os.path.join(self.rel_dir, fname))

    def failures(self):
        return [m for m in self.metrics if m['ok'] is False]

    def assert_ok(self):
        bad = self.failures()
        assert not bad, 'out of tolerance: ' + '; '.join(_describe(m) for m in bad)

    def to_dict(self):
        return dict(metrics=self.metrics, figures=self.figures, notes=self.notes)

    def dump(self):
        if not (self.metrics or self.figures or self.notes):
            return
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, 'metrics.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=1)


def _describe(m):
    v = m['value']
    vs = f'{v:.4g}' if isinstance(v, float) else f'[{min(v):.4g} .. {max(v):.4g}]'
    crit = []
    if m.get('expected') is not None:
        e = m['expected']
        es = f'{e:.4g}' if isinstance(e, (int, float)) else 'array'
        crit.append(f'expected {es}' + (f' rtol {m["rtol"]}' if m.get('rtol') else '') + (f' atol {m["atol"]}' if m.get('atol') else ''))
    if m.get('lo') is not None or m.get('hi') is not None:
        crit.append(f'in [{m.get("lo")}, {m.get("hi")}]')
    return f'{m["name"]} = {vs} {m.get("unit", "")} ({", ".join(crit)})'


# ----------------------------------------------------------------------------------------------------------------------
# Numerical tools
# ----------------------------------------------------------------------------------------------------------------------

def structure_function(scrn, lags):
    """D(r) of a 2D map along x (columns) and y (rows): mean squared difference over all pairs at each integer lag."""
    dx = np.array([np.mean((scrn[:, l:] - scrn[:, :-l]) ** 2) for l in lags])
    dy = np.array([np.mean((scrn[l:, :] - scrn[:-l, :]) ** 2) for l in lags])
    return dx, dy


def centroid(img):
    """Centre of gravity (row, col) in pixels."""
    img = np.asarray(img, dtype=float)
    tot = img.sum()
    r, c = np.indices(img.shape)
    return (r * img).sum() / tot, (c * img).sum() / tot


def integer_shift(a, b, max_shift=4, margin=None):
    """
    Integer shift (drow, dcol) such that b[i, j] ~ a[i - drow, j - dcol] (the content of `a` moved by (drow, dcol)),
    by least squares over the overlapping region.
    """
    n0, n1 = a.shape
    m = max_shift if margin is None else margin
    best = None
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            bb = b[m:n0 - m, m:n1 - m]
            aa = a[m - dr:n0 - m - dr, m - dc:n1 - m - dc]
            e = np.mean((aa - bb) ** 2)
            if best is None or e < best[0]:
                best = (e, dr, dc)
    return best[1], best[2]


def fwhm_1d(profile, x=None):
    """FWHM of a peaked 1D profile by linear interpolation of the half-maximum crossings around the peak."""
    p = np.asarray(profile, dtype=float)
    x = np.arange(len(p), dtype=float) if x is None else np.asarray(x, dtype=float)
    k = int(np.argmax(p))
    half = p[k] / 2
    i = k
    while i > 0 and p[i - 1] > half:
        i -= 1
    j = k
    while j < len(p) - 1 and p[j + 1] > half:
        j += 1
    if i == 0 or j == len(p) - 1:
        return np.nan
    xl = x[i - 1] + (half - p[i - 1]) / (p[i] - p[i - 1]) * (x[i] - x[i - 1])
    xr = x[j] + (half - p[j]) / (p[j + 1] - p[j]) * (x[j + 1] - x[j])
    return xr - xl


def fwhm_2d(img, upsample=8):
    """
    FWHM (pixels) of a 2D PSF: mean of the x and y cuts through the peak after Fourier upsampling (band-limited
    images, so the interpolation is exact up to the frame edges).
    """
    img = np.asarray(img, dtype=float)
    n0, n1 = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    N0, N1 = n0 * upsample, n1 * upsample
    G = np.zeros((N0, N1), dtype=complex)
    o0, o1 = (N0 - n0) // 2, (N1 - n1) // 2
    G[o0:o0 + n0, o1:o1 + n1] = F
    up = np.real(np.fft.ifft2(np.fft.ifftshift(G))) * upsample ** 2
    r, c = np.unravel_index(np.argmax(up), up.shape)
    return 0.5 * (fwhm_1d(up[r, :]) + fwhm_1d(up[:, c])) / upsample
