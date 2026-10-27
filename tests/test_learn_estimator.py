"""
T1/T2 validation of SAOS.tomography.learnEstimator.LearnEstimator: von Karman structure-function series, theoretical
slope covariance matrix, Phase-1 atmosphere fit, gain conversion, tomographic reconstructor (static and predictive).

Stand-in: LearnEstimator only reads a few attributes of its light paths (wfs.subaperture_size, plate_scale, nSubap,
valid_subapertures; src.wavelength, coordinates [arcsec, deg]; tel.D, samplingTime; delay), so the tests use
SimpleNamespace objects with those attributes instead of full LightPaths (no optics are involved in this module).

Independent references (no SAOS code): von Karman covariance with scipy Bessel-K (theory.py); slope model
s = [phi(A) - phi(a)] / d between the midpoints of the subaperture edges, so that
  Cov(s_i, s_j) = sum_l w_l [C(A_i-B_j) - C(A_i-b_j) - C(a_i-B_j) + C(a_i-b_j)] / (d_i d_j)
with C the von Karman phase covariance of layer l (weight w_l, shifted by the layer position at the source direction).
Synthetic phase screens are generated here with an FFT generator with Lane subharmonics. Frozen flow:
phi(r, t) = phi0(r - V t), so Cov(phi(r_a, t + dt), phi(r_b, t)) = C(r_a - r_b - V dt).

Units: slopes in rad/m of phase gradient (phase in rad at the wavelength of r0), lengths in m.
"""
import logging
import os
import tempfile
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import theory
from citools import known_issue
from SAOS.tomography.learnEstimator import LearnEstimator

ARCSEC = 206265.0

# ----------------------------------------------------------------------------------------------------------------------
# Stand-ins and independent reference
# ----------------------------------------------------------------------------------------------------------------------


def make_lp(r_arcsec=0.0, theta_deg=0.0, n=6, D=3.0, ps=0.5, wavelength=640e-9, delay=2, Ts=1e-3):
    return SimpleNamespace(
        wfs=SimpleNamespace(subaperture_size=D / n, plate_scale=ps, nSubap=n, valid_subapertures=np.ones((n, n), dtype=bool)),
        src=SimpleNamespace(wavelength=wavelength, coordinates=[r_arcsec, theta_deg]),
        tel=SimpleNamespace(D=D, samplingTime=Ts), delay=delay)


def make_est(logger, meas, tgt, alts):
    est = LearnEstimator(alts, 0.0, meas, tgt, logger=logger)
    est.device = 'cpu'
    return est


def ref_rows(lps, alts, order='natural'):
    """
    Edge-midpoint pairs of the slopes of `lps` at each layer: A, a with shape (nLayers, N, 2) (x, y) [m], subaperture
    size d (N,). order 'natural': [X of WFS0, X of WFS1, ..., Y of WFS0, ...]; 'controller': [X0, Y0, X1, Y1, ...].
    """
    blocks = []
    for lp in lps:
        n, D = lp.wfs.nSubap, lp.tel.D
        d = D / n
        c = (np.arange(n) - (n - 1) / 2) * d
        cx, cy = np.meshgrid(c, c)                         # [row = y, column = x], flattened in C order
        cx, cy = cx.ravel(), cy.ravel()
        rad, th = lp.src.coordinates[0], np.deg2rad(lp.src.coordinates[1])
        blk = {}
        for ax, name in ((0, 'X'), (1, 'Y')):
            A = np.zeros((len(alts), n * n, 2))
            a = np.zeros_like(A)
            for l, h in enumerate(alts):
                off = h * rad / ARCSEC * np.array([np.cos(th), np.sin(th)])
                ctr = np.stack([cx, cy], -1) + off
                e = np.zeros(2)
                e[ax] = d / 2
                A[l], a[l] = ctr + e, ctr - e
            blk[name] = (A, a, np.full(n * n, d))
        blocks.append(blk)
    seq = [(b, k) for k in 'XY' for b in blocks] if order == 'natural' else [(b, k) for b in blocks for k in 'XY']
    A = np.concatenate([b[k][0] for b, k in seq], axis=1)
    a = np.concatenate([b[k][1] for b, k in seq], axis=1)
    d = np.concatenate([b[k][2] for b, k in seq])
    return A, a, d


def ref_cov(rows_t, rows_s, frac, r0, L0, shift=None):
    """Independent covariance between slope rows (targets) and slope columns (sensed); shift (nLayers, 2) is added to r_t - r_s."""
    At, at, dt_ = rows_t
    As, as_, ds = rows_s
    sh = np.zeros((len(frac), 1, 1, 2)) if shift is None else np.asarray(shift, float)[:, None, None, :]

    def C(P, Q):
        return theory.vk_covariance(np.linalg.norm(P[:, :, None, :] - Q[:, None, :, :] + sh, axis=-1), r0, L0)

    block = C(At, As) - C(At, as_) - C(at, As) + C(at, as_)
    return np.einsum('l,lij->ij', np.asarray(frac, float), block) / (dt_[:, None] * ds[None, :])


def vk_screens(n_pairs, N, h, r0, L0, rng):
    """
    FFT von Karman phase screens [rad], N x N pixels of h metres, with 3 levels of Lane subharmonics. Each complex
    realisation gives two independent screens (real and imaginary parts); returns (2 n_pairs, N, N).
    """
    L = N * h
    fx = np.fft.fftfreq(N, h)
    FX, FY = np.meshgrid(fx, fx)
    psd = theory.vk_psd(np.sqrt(FX ** 2 + FY ** 2), r0, L0)
    psd[0, 0] = 0.0
    df = 1.0 / L
    cn = (rng.standard_normal((n_pairs, N, N)) + 1j * rng.standard_normal((n_pairs, N, N))) * np.sqrt(psd) * df
    z = np.fft.ifft2(cn) * N * N
    x = (np.arange(N) - N / 2) * h
    X, Y = np.meshgrid(x, x)
    sub = np.zeros_like(z)
    for p in (1, 2, 3):
        dfp = 1.0 / (3 ** p * L)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    continue
                fxp, fyp = i * dfp, j * dfp
                amp = np.sqrt(theory.vk_psd(np.hypot(fxp, fyp), r0, L0)) * dfp
                c = (rng.standard_normal(n_pairs) + 1j * rng.standard_normal(n_pairs)) * amp
                sub += c[:, None, None] * np.exp(2j * np.pi * (fxp * X + fyp * Y))[None]
    z = z + sub - sub.mean(axis=(1, 2), keepdims=True)
    return np.concatenate([z.real, z.imag], axis=0)


def screen_slopes(screens, rows, N, h, shift_px=(0, 0)):
    """Slopes (n_screens, N_rows) from the screen nodes at the edge midpoints of `rows` (layer 0), window moved by shift_px (ix, iy)."""
    A, a, d = rows
    c = N // 2
    ia = np.rint(A[0] / h).astype(int) + c
    ib = np.rint(a[0] / h).astype(int) + c
    assert np.allclose(ia * h - c * h, A[0]) and np.allclose(ib * h - c * h, a[0]), 'midpoints must lie on screen nodes'
    sx, sy = shift_px
    return (screens[:, ia[:, 1] + sy, ia[:, 0] + sx] - screens[:, ib[:, 1] + sy, ib[:, 0] + sx]) / d


# ----------------------------------------------------------------------------------------------------------------------
# von Karman structure function series
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def est1(logger):
    return make_est(logger, [make_lp()], [make_lp()], [0.0])


def series_vs_exact(est, r, L0, r0=0.15):
    c = est.prepare_vk_constants_torch(torch.float64)
    D = est.vk_structure_torch(r0, L0, torch.as_tensor(r, dtype=torch.float64), c).numpy()
    return D, theory.vk_structure(r, r0, L0)


@pytest.mark.fast
@known_issue('LEARN-VK-SERIES', 'vk_structure_torch uses a 10-term power series that diverges for small L0 and large r '
             '(x = pi r / L0 > 1): D(r) wrong by orders of magnitude, e.g. x65 at r = 8 m, L0 = 5 m', 'SAOS/tomography/learnEstimator.py:161-200')
def test_vk_series_exact_grid(est1, report, tolerances):
    """
    von Karman phase structure function D(r) = 2 [C(0) - C(r)] with C from the Bessel-K formula (scipy) against
    vk_structure_torch, on r in [1 cm, 20 m] (log grid, 60 points) and L0 in {5, 10, 25, 50} m, r0 = 0.15 m.
    Criterion: relative difference <= rtol (reference precision) over the whole grid.
    """
    tol = tolerances['learn_estimator']['vk_series']
    r = np.geomspace(0.01, 20, 60)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for L0 in (5, 10, 25, 50):
        D, ref = series_vs_exact(est1, r, L0)
        rel = np.abs(D / ref - 1)
        report.check(f'max |D_series / D_exact - 1| (L0 = {L0} m)', rel.max(), 0.0, atol=tol['rtol'])
        k = int(np.argmax(rel))
        report.metric(f'  worst r (L0 = {L0} m)', r[k], 'm')
        ax.loglog(r, np.maximum(rel, 1e-16), label=f'L0 = {L0} m')
    ax.axhline(tol['rtol'], color='k', ls=':')
    ax.set_xlabel('r [m]')
    ax.set_ylabel('|D_series / D_exact - 1|')
    ax.legend(fontsize=7)
    report.figure(fig, 'vk_series_error')
    D, ref = series_vs_exact(est1, np.array([8.0]), 5.0)
    report.metric('D_series / D_exact at r = 8 m, L0 = 5 m', D[0] / ref[0])
    report.assert_ok()


@pytest.mark.fast
def test_vk_series_convergent_region(est1, report, tolerances):
    """
    Control: where the series converges (x = pi r / L0 <= 1) it must equal the exact Bessel-K D(r); r in [1 cm, L0/pi],
    L0 in {5, 10, 25, 50} m. D(0) = 0.
    """
    tol = tolerances['learn_estimator']['vk_series']
    for L0 in (5, 10, 25, 50):
        r = np.geomspace(0.01, L0 / np.pi * tol['convergent_x_max'], 40)
        D, ref = series_vs_exact(est1, r, L0)
        report.check(f'max |D_series / D_exact - 1| (L0 = {L0} m, x <= 1)', np.abs(D / ref - 1).max(), 0.0, atol=tol['rtol'])
    c = est1.prepare_vk_constants_torch(torch.float64)
    d0 = est1.vk_structure_torch(0.15, 25.0, torch.zeros(3, dtype=torch.float64), c).numpy()
    report.check('D(0)', np.abs(d0).max(), 0.0, atol=0.0)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Theoretical slope covariance matrix
# ----------------------------------------------------------------------------------------------------------------------

ALTS = [0.0, 4000.0, 10000.0]
FRAC = [0.5, 0.3, 0.2]
R0, L0_ = 0.12, 25.0


def saos_cov(est, lps_rows, r0, L0, frac, noise=None):
    """compute_covariance_matrix_torch for the measured WFS with the row layout of learn() (all X rows, then all Y rows)."""
    nW = len(est.measWFSparams)
    A = np.concatenate([est.measWFSmidpoint_X[j]['midPointA'] for j in range(nW)] + [est.measWFSmidpoint_Y[j]['midPointA'] for j in range(nW)], axis=1)
    a = np.concatenate([est.measWFSmidpoint_X[j]['midPointa'] for j in range(nW)] + [est.measWFSmidpoint_Y[j]['midPointa'] for j in range(nW)], axis=1)
    diff = torch.as_tensor(est.compute_coordinate_differences(A, a, A, a), dtype=torch.float64)
    d = torch.as_tensor(lps_rows[2], dtype=torch.float64)
    scale = 1.0 / (2.0 * d[:, None] * d[None, :])
    c = est.prepare_vk_constants_torch(torch.float64)
    kw = {}
    if noise is not None:
        n_sub = est.measWFSparams[0]['coordinates_per_layer'].shape[1]
        kw = dict(noise_var=torch.as_tensor(noise, dtype=torch.float64), row_sizes_tensor=torch.tensor([n_sub] * (2 * nW)),
                  diag_idx=torch.arange(2 * nW * n_sub))
    return est.compute_covariance_matrix_torch(diff, torch.tensor(r0, dtype=torch.float64), torch.tensor(L0, dtype=torch.float64),
                                               torch.as_tensor(frac, dtype=torch.float64), scale, c, **kw).numpy()


@pytest.fixture(scope='module')
def multi(logger):
    meas = [make_lp(0, 0), make_lp(30, 0), make_lp(30, 90)]
    tgt = [make_lp(10, 45), make_lp(0, 0)]
    est = make_est(logger, meas, tgt, ALTS)
    return est, meas, tgt


@pytest.mark.fast
def test_css_matches_analytic(multi, report, tolerances):
    """
    Theoretical slope covariance matrix (3 WFS 6x6, d = 0.5 m, 3 layers at 0 / 4 / 10 km, r0 = 0.12 m, L0 = 25 m, noise
    variance per WFS added on the diagonal) against the independent Bessel-K reference built from the WFS geometry.
    Criterion: max |C_saos - C_ref| / max |C_ref| <= rtol (float64).
    """
    tol = tolerances['learn_estimator']['matrix']
    est, meas, _ = multi
    rows = ref_rows(meas, ALTS)
    noise = np.array([0.5, 1.0, 2.0])
    C0 = ref_cov(rows, rows, FRAC, R0, L0_)
    Cn = np.diag(np.concatenate([np.repeat(noise, 36)] * 2))
    S = saos_cov(est, rows, R0, L0_, FRAC)
    Sn = saos_cov(est, rows, R0, L0_, FRAC, noise=noise)
    report.check('max |C_saos - C_ref| / max |C_ref|', np.abs(S - C0).max() / np.abs(C0).max(), 0.0, atol=tol['rtol'])
    report.check('with noise on the diagonal: max |C_saos - (C_ref + C_nn)| / max |C_ref|', np.abs(Sn - C0 - Cn).max() / np.abs(C0).max(), 0.0, atol=tol['rtol'])
    report.metric('rms of the slope covariance diagonal [(rad/m)^2]', np.sqrt(np.mean(np.diag(C0) ** 2)))
    report.assert_ok()


@pytest.mark.fast
def test_css_symmetric_positive_semidefinite(multi, report, tolerances):
    """C_ss (noise free) of the same configuration is symmetric and positive semi-definite (covariance of real slopes)."""
    tol = tolerances['learn_estimator']['matrix']
    est, meas, _ = multi
    S = saos_cov(est, ref_rows(meas, ALTS), R0, L0_, FRAC)
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    report.check('|C - C^T| / |C|', np.abs(S - S.T).max() / np.abs(S).max(), 0.0, atol=tol['asym_rtol'])
    report.check('lowest eigenvalue / largest eigenvalue', ev[0] / ev[-1], lo=-tol['psd_rtol'])
    report.assert_ok()


@pytest.mark.nightly
def test_css_vs_independent_screens(logger, report, tolerances, rng):
    """
    Theoretical slope covariance against the empirical covariance of slopes taken from independent von Karman screens
    (FFT + Lane subharmonics generated in this test, N = 256, h = d/4, r0 = 0.15 m, L0 = 10 m, one WFS 6x6 of d = 0.5 m,
    one layer, slopes between edge midpoints on screen nodes). With n independent zero-mean Gaussian realisations the
    sample covariance has Var(C^_ij) = (C_ii C_jj + C_ij^2) / n, so Q = sum (C^ - C)^2 / sum Var has expectation 1;
    criteria: Q within [q_lo, q_hi] (measured spread over seeds) and tr C^ / tr C within trace_atol.
    """
    tol = tolerances['learn_estimator']['screens']
    r0, L0, N, h = 0.15, 10.0, 256, 0.125
    est = make_est(logger, [make_lp()], [make_lp()], [0.0])
    rows = ref_rows([make_lp()], [0.0])
    S = saos_cov(est, rows, r0, L0, [1.0])
    n = tol['n_screens']
    sl = []
    for _ in range(n // 100):
        sl.append(screen_slopes(vk_screens(50, N, h, r0, L0, rng), rows, N, h))
    sl = np.concatenate(sl)
    Ce = sl.T @ sl / len(sl)
    dg = np.diag(S)
    var = (dg[:, None] * dg[None, :] + S ** 2) / len(sl)
    Q = np.sum((Ce - S) ** 2) / np.sum(var)
    report.check('Q = sum (C_emp - C)^2 / sum Var', Q, lo=tol['q_lo'], hi=tol['q_hi'])
    report.check('tr C_emp / tr C', np.trace(Ce) / np.trace(S), 1.0, atol=tol['trace_atol'])
    # subaperture-averaged slopes would differ from the midpoint model: not part of the criteria
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for a_, m, t in zip(ax, (S, Ce), ('theory', 'empirical (screens)')):
        im = a_.imshow(m, cmap='RdBu_r', vmin=-np.abs(S).max(), vmax=np.abs(S).max())
        a_.set_title(t, fontsize=8)
    report.figure(fig, 'css_theory_vs_screens')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Phase 1 fit and gain
# ----------------------------------------------------------------------------------------------------------------------

def write_slopes(path, slopes):
    with h5py.File(path, 'w') as f:
        f.create_dataset('/LightPath_0/slopes_1D/data', data=slopes)


def synth_slopes_px(lp, n_samples, r0, L0, rng):
    """Slopes [px] of one 6x6 WFS from independent screens of phase at 500 nm (r0 at 500 nm): OPD gradient = phase gradient
    [rad/m] * 500 nm / (2 pi), px = OPD gradient * 206265 / plate_scale (achromatic, independent of the WFS wavelength)."""
    N, h = 256, lp.tel.D / lp.wfs.nSubap / 4
    rows = ref_rows([lp], [0.0])
    sl = np.concatenate([screen_slopes(vk_screens(50, N, h, r0, L0, rng), rows, N, h) for _ in range(n_samples // 100)])
    return sl * 500e-9 / (2 * np.pi) * ARCSEC / lp.wfs.plate_scale


def run_learn(est, atm, tmp, spy=None, **kw):
    np.random.seed(0)
    return est.learn(atm, tmp, os.path.join(tmp, 'out'), selSubaps=1.0, selSamples=1.0, estimate_wind=False, **kw)


class _GainSpy:
    """Records the gain argument that learn() passes to compute_measurement_covariance (optionally replacing it)."""

    def __init__(self, est, replace=None):
        self.est, self.gains, self.replace = est, [], replace
        self.orig = est.compute_measurement_covariance
        est.compute_measurement_covariance = self

    def __call__(self, t, td, gain, dtype):
        self.gains.append(gain)
        return self.orig(t, td, gain if self.replace is None else self.replace, dtype)


@pytest.mark.fast
def test_gain_slopes_to_phase_gradient(logger, report, tolerances, rng):
    """
    A slope of s px moves the spot by s ps arcsec = s ps / 206265 rad, the wavefront (OPD) gradient (achromatic); the von Karman
    model is the phase structure function of r0 at 500 nm (Atmosphere convention), so the factor passed by learn() to
    compute_measurement_covariance must be 2 pi ps / (206265 x 500 nm), independent of the WFS wavelength (640 nm here),
    and of the subaperture size (0.5 m); plate scale 0.5 arcsec/px.
    """
    tol = tolerances['learn_estimator']['gain']
    lp = make_lp()
    est = make_est(logger, [lp], [lp], [0.0])
    spy = _GainSpy(est)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 's.h5')
        write_slopes(path, np.random.default_rng(1).standard_normal((50, 72)))
        np.random.seed(0)
        est.learn({'r0': 0.15, 'L0': 25.0, 'fractionalCn2': [1.0]}, path, tmp, selSubaps=1.0, selSamples=1.0, max_iters=1, estimate_wind=False)
    expected = 2 * np.pi * lp.wfs.plate_scale / (ARCSEC * 500e-9)
    report.metric('gain used by learn()', spy.gains[0], 'rad/m per px')
    report.check('gain / (2 pi ps / (206265 x 500 nm))', spy.gains[0] / expected, 1.0, rtol=tol['rtol'])
    report.assert_ok()


def phase1_fit(logger, rng, replace_gain, r0=0.15, L0=10.0, n_samples=1000):
    lp = make_lp()
    est = make_est(logger, [lp], [lp], [0.0])
    gain = 2 * np.pi * lp.wfs.plate_scale / (ARCSEC * 500e-9)
    spy = _GainSpy(est, replace=gain if replace_gain else None)
    sl_px = synth_slopes_px(lp, n_samples, r0, L0, rng)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 's.h5')
        write_slopes(path, sl_px)
        out = run_learn(est, {'r0': 1.3 * r0, 'L0': L0, 'fractionalCn2': [1.0]}, path, lr=2e-2, max_iters=1500, patience=400,
                        lr_patience=60, initial_noise=1e-3)
    return out


@pytest.mark.nightly
def test_phase1_recovers_r0(logger, report, tolerances, rng):
    """
    Phase 1 on slopes from independent von Karman screens (r0 = 0.15 m, L0 = 10 m, 1 layer, 1000 samples, noise free,
    start from 1.3 r0). To be independent of the gain test, the slope-to-gradient factor is replaced by the correct
    2 pi ps / (206265 x 500 nm) in this test. Criterion: |r0_fit / r0 - 1| <= r0_rtol (3.5 sigma of the estimator over seeds).
    """
    tol = tolerances['learn_estimator']['phase1']
    out = phase1_fit(logger, rng, True)
    report.metric('L0 fitted', out['L0'], 'm')
    report.metric('noise variance fitted', float(np.ravel(out['noise_var'])[0]))
    report.check('r0_fit / r0 - 1', out['r0'] / 0.15 - 1, 0.0, atol=tol['r0_rtol'])
    report.assert_ok()


@pytest.mark.nightly
def test_phase1_recovers_r0_end_to_end(logger, report, tolerances, rng):
    """Same as test_phase1_recovers_r0 with the unmodified gain of learn(): r0 must be recovered from slopes in px."""
    tol = tolerances['learn_estimator']['phase1']
    out = phase1_fit(logger, rng, False)
    report.check('r0_fit / r0 - 1', out['r0'] / 0.15 - 1, 0.0, atol=tol['r0_rtol'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Reconstructor
# ----------------------------------------------------------------------------------------------------------------------

NOISE = np.array([0.3, 0.6, 0.9])
REG = 1e-9
VX, VY = np.array([10.0, -6.0, 0.0]), np.array([3.0, 8.0, 0.0])


def expected_reconstructor(meas, tgt, shift_sign, dt, order='natural', noise=NOISE):
    """R = C_ts (C_ss + C_nn + reg)^-1 and the exact error-variance trace of a reconstructor, from the Bessel-K reference."""
    rs, rt = ref_rows(meas, ALTS, order), ref_rows(tgt, ALTS, order)
    nW, nsub = len(meas), 36
    nz = np.concatenate([np.repeat(noise, nsub)] * 2) if order == 'natural' else np.repeat(np.repeat(noise, nsub), 2)
    # controller order interleaves X0, Y0, X1, ...: noise per WFS repeated over 2 * nsub rows
    if order == 'controller':
        nz = np.concatenate([np.full(2 * nsub, v) for v in noise])
    Css = ref_cov(rs, rs, FRAC, R0, L0_) + np.diag(nz + REG)
    shift = shift_sign * dt * np.stack([VX, VY], -1)
    Cts = ref_cov(rt, rs, FRAC, R0, L0_, shift)
    Ctt = ref_cov(rt, rt, FRAC, R0, L0_)
    return Cts @ np.linalg.inv(Css), Cts, Css, Ctt, nz


def err_trace(R, Cts, Css_noisy, Ctt):
    """tr E[(R s - t)(R s - t)^T] with exact covariances (sensed noise included in Css_noisy; reg is negligible)."""
    return np.trace(Ctt - R @ Cts.T - Cts @ R.T + R @ Css_noisy @ R.T)


def build(est, **kw):
    return est.build_reconstructor(r0=R0, L0=L0_, fractionalCn2=FRAC, noise_var=NOISE, regularization=REG, **kw)


@pytest.mark.fast
@pytest.mark.parametrize('permute', [False, True])
def test_reconstructor_static(multi, permute, report, tolerances):
    """
    Non-predictive reconstructor R = C_ts (C_ss + C_nn)^-1 (3 WFS sensed, 2 target directions, 3 layers) against the
    independent Bessel-K reference, in the natural order [X of all WFS, Y of all WFS] and with permute_for_controller
    ([WFS0_X, WFS0_Y, WFS1_X, ...]). Criterion: max |R - R_ref| / max |R_ref| <= rtol.
    """
    tol = tolerances['learn_estimator']['reconstructor']
    est, meas, tgt = multi
    R = build(est, windSpeedX=None, windSpeedY=None, dt=0.0, permute_for_controller=permute)
    Rref = expected_reconstructor(meas, tgt, 1, 0.0, 'controller' if permute else 'natural')[0]
    report.check('max |R - R_ref| / max |R_ref|', np.abs(R - Rref).max() / np.abs(Rref).max(), 0.0, atol=tol['rtol'])
    report.assert_ok()


DT = 0.02


@pytest.mark.fast
def test_reconstructor_predicts_future(multi, report, tolerances):
    """
    Frozen flow phi(r, t) = phi0(r - V t): the covariance between the target at t + dt and the sensed slopes at t is
    C(r_t - r_s - V dt). Reconstructor built with the winds (10, 3), (-6, 8), (0, 0) m/s and dt = 20 ms.
    Criterion: with the exact covariances of the true future target, the error variance of R_saos equals that of the
    optimal R_ref built with -V dt (ratio 1 + err_ratio_atol), and R_saos is not the retrodiction one.
    """
    tol = tolerances['learn_estimator']['reconstructor']
    est, meas, tgt = multi
    R = build(est, windSpeedX=VX, windSpeedY=VY, dt=DT, permute_for_controller=False)
    Rf, Cts, Css, Ctt, nz = expected_reconstructor(meas, tgt, -1, DT)
    Rb, Ctsb, _, _, _ = expected_reconstructor(meas, tgt, +1, DT)
    Cssn = Css - REG * np.eye(len(Css))
    e_saos, e_opt = err_trace(R, Cts, Cssn, Ctt), err_trace(Rf, Cts, Cssn, Ctt)
    e_retro_opt = err_trace(Rb, Ctsb, Cssn, Ctt)
    report.metric('error variance of R_saos at t + dt', e_saos)
    report.metric('error variance of optimal R at t + dt', e_opt)
    report.metric('error variance of the retrodiction R (+V dt) at t + dt', err_trace(Rb, Cts, Cssn, Ctt))
    report.metric('error variance of R_saos at t - dt', err_trace(R, Ctsb, Cssn, Ctt))
    report.check('tr E[R_saos] / tr E[R_opt] - 1', e_saos / e_opt - 1, 0.0, atol=tol['err_ratio_atol'])
    report.check('max |R_saos - R_opt| / max |R_opt|', np.abs(R - Rf).max() / np.abs(Rf).max(), 0.0, atol=tol['rtol'])
    report.assert_ok()


@pytest.mark.fast
def test_reconstructor_zero_wind_ignores_dt(multi, report, tolerances):
    """Control: with zero winds the shift is zero, so any dt gives the static reconstructor (also == reference)."""
    tol = tolerances['learn_estimator']['reconstructor']
    est, meas, tgt = multi
    z = np.zeros(3)
    R = build(est, windSpeedX=z, windSpeedY=z, dt=DT, permute_for_controller=False)
    Rref = expected_reconstructor(meas, tgt, 1, 0.0)[0]
    report.check('max |R - R_static_ref| / max |R_ref|', np.abs(R - Rref).max() / np.abs(Rref).max(), 0.0, atol=tol['rtol'])
    report.assert_ok()


@pytest.mark.fast
def test_default_dt_is_delay_plus_one(logger, report, tolerances):
    """
    SAOS convention (tests/test_timing.py): a light path with delay d frames has a total pure delay (d + 1) Ts. With
    dt=None, build_reconstructor must predict (d + 1) Ts ahead: its result must equal the one with explicit
    dt = (d + 1) Ts (target light path delay d = 2, Ts = 5 ms, winds of the previous test).
    """
    tol = tolerances['learn_estimator']['reconstructor']
    d, Ts = 2, 5e-3
    meas, tgt = [make_lp(0, 0), make_lp(30, 0)], [make_lp(10, 45, delay=d, Ts=Ts)]
    est = make_est(logger, meas, tgt, ALTS)
    kw = dict(r0=R0, L0=L0_, fractionalCn2=FRAC, noise_var=NOISE[:2], regularization=REG, windSpeedX=VX, windSpeedY=VY, permute_for_controller=False)
    R_def = est.build_reconstructor(dt=None, **kw)
    R_exp = est.build_reconstructor(dt=(d + 1) * Ts, **kw)
    R_old = est.build_reconstructor(dt=d * Ts, **kw)
    report.metric('|R(d Ts) - R((d+1) Ts)| / |R|', np.abs(R_old - R_exp).max() / np.abs(R_exp).max())
    report.check('max |R(dt=None) - R(dt=(d+1) Ts)| / max |R|', np.abs(R_def - R_exp).max() / np.abs(R_exp).max(), 0.0, atol=tol['dt_rtol'])
    report.assert_ok()


@pytest.mark.nightly
def test_frozen_flow_cross_covariance_sign(report, tolerances, rng):
    """
    Reference check of the sign used above, from independent screens (not SAOS): one layer moving at V = +x with
    V dt = 1 m (8 px of h = 0.125 m), phi(x, t + dt) = phi0(x - V dt). The empirical covariance between the slopes of
    the same 6x6 WFS at t + dt (rows) and at t (columns) agrees with the theory with shift -V dt and not with +V dt.
    Criterion: relative Frobenius error vs -V dt <= err_hi; vs +V dt >= wrong_sign_lo.
    """
    tol = tolerances['learn_estimator']['frozen_flow']
    r0, L0, N, h = 0.15, 10.0, 256, 0.125
    lp = make_lp()
    rows = ref_rows([lp], [0.0])
    scr = [vk_screens(50, N, h, r0, L0, rng) for _ in range(tol['n_screens'] // 100)]
    s_t = np.concatenate([screen_slopes(s, rows, N, h) for s in scr])
    s_f = np.concatenate([screen_slopes(s, rows, N, h, shift_px=(-8, 0)) for s in scr])     # phi0(x - V dt): node index - 8
    Ce = s_f.T @ s_t / len(s_t)
    Cm = ref_cov(rows, rows, [1.0], r0, L0, [[-1.0, 0.0]])
    Cp = ref_cov(rows, rows, [1.0], r0, L0, [[+1.0, 0.0]])
    nrm = np.linalg.norm(Cm)
    report.check('||C_emp - C(-V dt)|| / ||C||', np.linalg.norm(Ce - Cm) / nrm, 0.0, atol=tol['err_hi'])
    report.check('||C_emp - C(+V dt)|| / ||C||', np.linalg.norm(Ce - Cp) / nrm, lo=tol['wrong_sign_lo'])
    report.assert_ok()
