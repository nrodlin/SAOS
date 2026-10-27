"""
T2 validation of the atmosphere at EST scale (D 4.149 m, 216 px, 1.92 cm/px, L0 25 m): r0 scan of the screen generators,
outer-scale recovery, Zernike variances on the pupil, Taylor frozen flow, anisoplanatism and the seeing-limited long
exposure with ScienceCam.

Every reference is independent of SAOS: von Karman formulas copied below from the literature (Noll 1976, Assemat 2006,
Tokovinin 2002), a Zernike basis and a pupil covariance built here. SAOS only provides the screens and images under test.
Tolerances (tests/tolerances/atmosphere_t2.yaml) are ~3.5 sigma of the estimator measured over seeds.
Phase screens in rad at 500 nm; OPD in m; the OPD in rad at 500 nm is OPD * 2 pi / 500 nm.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import gamma, kv
from math import factorial

from citools import fwhm_2d, known_issue, structure_function
from SAOS.Atmosphere import Atmosphere
from SAOS.ScienceCam import ScienceCam
from SAOS.Source import Source
from SAOS.Telescope import Telescope
from SAOS.infinitePhaseScreen import PhaseScreenVonKarman

pytestmark = pytest.mark.nightly

D, NX, L0, TS = 4.149, 216, 25.0, 1e-3
PIX = D / NX
ROW_SPEED = PIX / TS * (1 + 1e-9)     # wind speed of one screen row per update [m/s] (advance() counts rows as updates)
LAM = 500e-9
K_OPD = 2 * np.pi / LAM                      # OPD [m] -> phase [rad at 500 nm]
RAD2ARCSEC = 206265.0
R0S = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0]


# ----------------------------------------------------------------------------------------------------------------------
# Independent von Karman references (phase in rad at 500 nm, lengths in m)
# ----------------------------------------------------------------------------------------------------------------------

_KPSD = ((24 / 5) * gamma(6 / 5)) ** (5 / 6) * gamma(11 / 6) ** 2 / (2 * np.pi ** (11 / 3))   # 0.0229


def vk_var(r0, L0):
    """C(0) = 0.0229 r0^-5/3 pi (6/5) L0^5/3 = int 2 pi f PSD(f) df."""
    return (L0 / r0) ** (5 / 3) * _KPSD * np.pi * 6 / 5


def vk_cov(r, r0, L0):
    """C(r) = C(0) 2^(1/6)/Gamma(5/6) x^(5/6) K_5/6(x), x = 2 pi r / L0."""
    x = 2 * np.pi * np.maximum(np.asarray(r, dtype=float), 1e-12) / L0
    return vk_var(r0, L0) * 2 ** (1 / 6) / gamma(5 / 6) * x ** (5 / 6) * kv(5 / 6, x)


def vk_dstruct(r, r0, L0):
    return 2 * (vk_var(r0, L0) - vk_cov(r, r0, L0))


def fit_r0_L0(r, d, r0_guess, L0_guess, fix_L0=False):
    """Least squares in log D of r0 and L0 (or r0 only) to a measured structure function."""
    if fix_L0:
        s = least_squares(lambda p: np.log(vk_dstruct(r, np.exp(p[0]), L0_guess)) - np.log(d), [np.log(r0_guess)])
        return float(np.exp(s.x[0])), float(L0_guess)
    s = least_squares(lambda p: np.log(vk_dstruct(r, np.exp(p[0]), np.exp(p[1]))) - np.log(d), [np.log(r0_guess), np.log(L0_guess)],
                      bounds=([np.log(1e-3), np.log(0.1)], [np.log(10), np.log(1e4)]))
    return float(np.exp(s.x[0])), float(np.exp(s.x[1]))


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def lags_of(n_max):
    return np.unique(np.round(np.logspace(0, np.log10(n_max), 20)).astype(int))


def circle_mask(n):
    x = np.arange(n) - (n - 1) / 2
    return np.hypot(*np.meshgrid(x, x)) <= n / 2


def advance(atm, rows):
    """`rows` updates of every layer (Atmosphere.update without the thread-pool overhead); one row each at ROW_SPEED."""
    layers = [getattr(atm, f'layer_{i + 1}') for i in range(atm.nLayer)]
    for _ in range(rows):
        for L in layers:
            atm.updateLayer(L)


# ----------------------------------------------------------------------------------------------------------------------
# A2': r0 scan at pupil scale
# ----------------------------------------------------------------------------------------------------------------------

def a2_scan(source, seed, logger, n_maps, r0s=R0S):
    """{r0: dict(r, d, r0_joint, L0_joint, r0_fixed)} of mean D(r) of `n_maps` screens of 216 px (4.149 m) per r0."""
    lags = lags_of(NX // 2)
    r = lags * PIX
    out = {}
    for r0 in r0s:
        scr = PhaseScreenVonKarman(NX, PIX, r0, L0, random_seed=seed, logger=logger)
        if source == 'initial':
            maps = pmap(scr.initial_screen, [np.random.default_rng([seed, k]) for k in range(n_maps)])
        else:
            for _ in range(NX):                                    # burn-in: the initial screen leaves
                scr.add_row(1)
            maps = []
            for _ in range(n_maps):
                for _ in range(2 * NX):
                    scr.add_row(1)
                maps.append(np.array(scr.scrn, copy=True))
        d = np.mean(pmap(lambda m: 0.5 * np.add(*structure_function(m, lags)), maps), axis=0)
        r0j, L0j = fit_r0_L0(r, d, r0, L0)
        r0f, _ = fit_r0_L0(r, d, r0, L0, fix_L0=True)
        out[r0] = dict(r=r, d=d, r0_joint=r0j, L0_joint=L0j, r0_fixed=r0f)
    return out


@pytest.mark.parametrize('source', ['initial', 'extruded'])
def test_r0_scan_pupil_scale(logger, report, tolerances, seed, source):
    """
    Screens of the pupil size (216 px, 4.149 m, L0 25 m) for r0 = 0.05-1 m: the mean structure function of 32 independent
    screens (initial FFT screens, or one extruded screen sampled every 2 nx rows after a burn-in of nx rows) is fitted
    with the von Karman D(r) (joint r0/L0 fit and r0 fit with L0 fixed, lags 1 px - nx/2). The fitted r0 must equal the
    generation value within 3.5 sigma of the fit over seeds (per r0); the joint L0 within the same criterion on log L0.
    """
    tol = tolerances['atmosphere_t2']['r0_scan'][source]
    n_maps = tolerances['atmosphere_t2']['r0_scan']['n_maps']
    res = a2_scan(source, seed, logger, n_maps)
    rj, rf, lj = [], [], []
    for i, r0 in enumerate(R0S):
        v = res[r0]
        rj.append(v['r0_joint'] / r0); rf.append(v['r0_fixed'] / r0); lj.append(v['L0_joint'] / L0)
        report.check(f'r0 {r0}: fitted/true (joint r0, L0 fit)', rj[-1], expected=1, atol=tol['r0_joint'][i])
        report.check(f'r0 {r0}: fitted/true (L0 fixed)', rf[-1], expected=1, atol=tol['r0_fixed'][i])
        report.check(f'r0 {r0}: fitted L0/true (joint fit)', lj[-1], lo=np.exp(-tol['logL0'][i]), hi=np.exp(tol['logL0'][i]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.array(R0S)
    ax[0].errorbar(x, rj, yerr=tol['r0_joint'], fmt='o-', capsize=3, label='joint r0/L0 fit')
    ax[0].errorbar(x * 1.04, rf, yerr=tol['r0_fixed'], fmt='s--', mfc='none', capsize=3, label='L0 fixed')
    ax[0].axhline(1, color='k', lw=0.6); ax[0].set_xscale('log')
    ax[0].set_xlabel('true r0 (500 nm) [m]'); ax[0].set_ylabel('fitted r0 / true r0'); ax[0].legend(fontsize=7)
    ax[0].set_title('bars: test tolerance (3.5 sigma)', fontsize=8)
    ax[1].errorbar(x, np.array(lj) * L0, yerr=None, fmt='o-')
    ax[1].fill_between(x, L0 * np.exp(-np.array(tol['logL0'])), L0 * np.exp(np.array(tol['logL0'])), color='0.9')
    ax[1].axhline(L0, color='k', lw=0.6); ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('true r0 [m]'); ax[1].set_ylabel('fitted L0 [m] (joint)')
    for r0 in R0S:
        v = res[r0]
        ax[2].semilogx(v['r'], v['d'] / vk_dstruct(v['r'], r0, L0), label=f'r0 {r0}')
    ax[2].axhline(1, color='k', lw=0.6); ax[2].set_xlabel('r [m]'); ax[2].set_ylabel('D_SAOS / D_vK'); ax[2].legend(fontsize=6, ncol=2)
    fig.suptitle(f'r0 scan, {source} screens, 216 px, {n_maps} screens per r0, L0 {L0:g} m', fontsize=9)
    report.figure(fig, 'r0_scan')
    report.assert_ok()


def l0_scan(L0_true, seed, logger, n_maps=24, nx=256, pix=0.15, r0=0.3):
    """Joint fit of r0, L0 on screens of 38.4 m (256 px of 15 cm, r0 0.3 m) (initial FFT screens): lags 3 px - nx/2."""
    lags = np.unique(np.round(np.logspace(np.log10(3), np.log10(nx // 2), 22)).astype(int))
    r = lags * pix
    scr = PhaseScreenVonKarman(nx, pix, r0, L0_true, random_seed=seed, logger=logger)
    maps = pmap(scr.initial_screen, [np.random.default_rng([seed, k]) for k in range(n_maps)], workers=6)
    d = np.mean(pmap(lambda m: 0.5 * np.add(*structure_function(m, lags)), maps), axis=0)
    r0j, L0j = fit_r0_L0(r, d, r0, L0_true)
    return dict(r=r, d=d, r0=r0, r0_joint=r0j, L0_joint=L0j)


@pytest.mark.parametrize('L0_true', [5.0, 12.0, 25.0])
def test_outer_scale_recovery(logger, report, tolerances, seed, L0_true):
    """
    Screens of 38.4 m (256 px of 15 cm, r0 0.3 m) that span the outer-scale turnover of D(r) (L0 = 5, 12, 25 m): the joint
    r0/L0 fit of the mean structure function of 24 initial screens (lags 0.45 m - 19 m) recovers L0 and r0 within 3.5
    sigma of the fit over seeds.
    """
    tol = tolerances['atmosphere_t2']['outer_scale'][f'L0_{L0_true:g}']
    v = l0_scan(L0_true, seed, logger)
    report.check('fitted L0 / true', v['L0_joint'] / L0_true, lo=np.exp(-tol['logL0']), hi=np.exp(tol['logL0']))
    report.check('fitted r0 / true', v['r0_joint'] / v['r0'], expected=1, atol=tol['r0'])
    r = np.logspace(np.log10(v['r'][0]), np.log10(v['r'][-1]), 100)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(v['r'], v['d'], 'o', ms=4, label='SAOS (24 screens)')
    ax[0].loglog(r, vk_dstruct(r, v['r0'], L0_true), 'k', label=f'von Karman L0 {L0_true:g} m')
    ax[0].loglog(r, vk_dstruct(r, v['r0_joint'], v['L0_joint']), '--', color='C3', label=f'fit r0 {v["r0_joint"]:.3f}, L0 {v["L0_joint"]:.1f}')
    ax[0].loglog(r, vk_dstruct(r, v['r0'], 1e4), ':', color='0.5', label='Kolmogorov')
    ax[0].set_xlabel('r [m]'); ax[0].set_ylabel('D(r) [rad^2]'); ax[0].legend(fontsize=7)
    ax[1].semilogx(v['r'], v['d'] / vk_dstruct(v['r'], v['r0'], L0_true), 'o-', ms=4)
    ax[1].axhline(1, color='k', lw=0.6); ax[1].set_xlabel('r [m]'); ax[1].set_ylabel('D_SAOS / D_vK')
    report.figure(fig, 'outer_scale')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# A5: Zernike variances on the pupil
# ----------------------------------------------------------------------------------------------------------------------

def _noll_nm(j):
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def _zernike(j, rho, th):
    """Noll-normalised Zernike (unit variance on the unit disk); even j: cos, odd j: sin."""
    n, m = _noll_nm(j)
    am = abs(m)
    rad = sum((-1) ** s * factorial(n - s) / (factorial(s) * factorial((n + am) // 2 - s) * factorial((n - am) // 2 - s))
              * rho ** (n - 2 * s) for s in range((n - am) // 2 + 1))
    if m == 0:
        return np.sqrt(n + 1) * rad
    return np.sqrt(2 * (n + 1)) * rad * (np.cos(am * th) if j % 2 == 0 else np.sin(am * th))


def zernike_basis(n=NX, jmax=11):
    """Orthonormalised (on the sampled pupil, Gram-Schmidt in Noll order) Zernikes Z1..Zjmax, unit mean square, [jmax, npix]."""
    x = (np.arange(n) - (n - 1) / 2) / (n / 2)
    X, Y = np.meshgrid(x, x)
    mask = np.hypot(X, Y) <= 1
    rho, th = np.hypot(X, Y)[mask], np.arctan2(Y, X)[mask]
    Z = np.stack([_zernike(j, rho, th) for j in range(1, jmax + 1)], axis=1)
    Q, R = np.linalg.qr(Z)
    Q = Q * np.sign(np.diag(R))
    return mask, (Q * np.sqrt(mask.sum())).T


def zernike_covariance(r0, L0, n=NX, jmax=11):
    """Exact covariance B^T C B of the Z2..Zjmax coefficients (a_j = mean_pupil(phi Z_j)) on the sampled pupil [rad^2]."""
    mask, Z = zernike_basis(n, jmax)
    lag = np.arange(-(n - 1), n) * PIX
    kern = vk_cov(np.hypot(*np.meshgrid(lag, lag)), r0, L0)
    N = mask.sum()
    F = []
    for j in range(1, jmax):
        f = np.zeros((n, n)); f[mask] = Z[j]
        F.append((f, fftconvolve(kern, f, mode='valid')))
    G = np.array([[np.sum(fi[mask] * cj[mask]) for (_, cj) in F] for (fi, _) in F]) / N ** 2
    return mask, Z, G


def zernike_coeffs(maps, mask, Z):
    """a_j = mean over the pupil of phi Z_j, j = 2..11; maps [n, 216, 216] in rad."""
    return np.asarray(maps)[:, mask] @ Z[1:].T / mask.sum()


def zernike_initial(seed, n, logger, r0=0.10):
    scr = PhaseScreenVonKarman(NX, PIX, r0, L0, random_seed=seed, logger=logger)
    mask, Z = zernike_basis()
    maps = pmap(scr.initial_screen, [np.random.default_rng([seed, k]) for k in range(n)])
    return zernike_coeffs(maps, mask, Z)


A5_PROFILE = dict(r0=0.12, fractionalR0=[0.6, 0.3, 0.1], altitude=[0.0, 5000.0, 12000.0], windDirection=[0, 90, 0],
                  windSpeed=[ROW_SPEED] * 3)


def zernike_atmosphere(seed, n, logger, n_inst=4, spacing=1000):
    """
    Z2-Z11 coefficients of the on-axis OPD of the 3-layer Atmosphere: n_inst independent atmospheres, burn-in `spacing`
    rows (19 m: the initial screens leave, successive samples decorrelated beyond L0), then n/n_inst samples each.
    """
    mask, Z = zernike_basis()
    tel = Telescope(resolution=NX, diameter=D, samplingTime=TS, fov=0, logger=logger)
    src = Source('V0', 0, coordinates=[0, 0], logger=logger)

    def one(i):
        atm = Atmosphere(L0=L0, telescope=tel, zenith=0, logger=logger, **A5_PROFILE)
        atm.initializeAtmosphere(randomState=seed + 7919 * i)
        out = []
        advance(atm, spacing)
        for _ in range(n // n_inst):
            advance(atm, spacing)
            out.append(np.array(atm.getOPD(src)) * K_OPD)
        return out
    maps = [m for chunk in pmap(one, range(n_inst), workers=n_inst) for m in chunk]
    return zernike_coeffs(maps, mask, Z)


def _zernike_report(report, a, G, tol, label):
    n = a.shape[0]
    S = a.T @ a / n                                          # mean zero: variance estimator
    ratio = np.diag(S) / np.diag(G)
    L = np.linalg.cholesky(G)
    w = a @ np.linalg.inv(L).T                               # whitened: identity covariance in theory
    Sw = w.T @ w / n
    names = [f'Z{j}' for j in range(2, 12)]
    for name, rt in zip(names, ratio):
        report.check(f'{name} variance / von Karman projection ({label})', rt, expected=1, atol=tol['mode'])
    report.check(f'mean over Z2-Z11 of variance ratios ({label})', ratio.mean(), expected=1, atol=tol['mean'])
    report.check(f'whitened covariance: max |offdiag| ({label})', np.max(np.abs(Sw - np.diag(np.diag(Sw)))), lo=0, hi=tol['offdiag'])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].bar(np.arange(10) - 0.2, np.diag(G), 0.4, label='theory (B^T C B)')
    ax[0].bar(np.arange(10) + 0.2, np.diag(S), 0.4, label=f'SAOS ({n} realisations)')
    ax[0].set_yscale('log'); ax[0].set_xticks(range(10)); ax[0].set_xticklabels(names, fontsize=7); ax[0].set_ylabel('variance [rad^2]'); ax[0].legend(fontsize=7)
    ax[1].axhspan(1 - tol['mode'], 1 + tol['mode'], color='0.9')
    ax[1].plot(ratio, 'o'); ax[1].axhline(1, color='k', lw=0.6)
    ax[1].set_xticks(range(10)); ax[1].set_xticklabels(names, fontsize=7); ax[1].set_ylabel('variance SAOS / theory')
    im = ax[2].imshow(Sw, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax[2].set_title('whitened covariance (identity expected)', fontsize=8); fig.colorbar(im, ax=ax[2])
    fig.suptitle(label, fontsize=9)
    report.figure(fig, 'zernike_variances')
    report.metric('Z2 variance ratio', ratio[0]); report.metric('Z4 variance ratio', ratio[2])
    report.assert_ok()


def test_zernike_variances_initial_screen(logger, report, tolerances, seed):
    """
    Z2-Z11 variances (and covariances) of 600 initial 216-px screens (r0 0.10 m, L0 25 m) on the circular pupil against
    the exact von Karman projection B^T C B built here on the sampled pupil (FFT convolution of the sampled covariance
    with each basis vector). Tolerance on each variance ratio: 3.5 sqrt(2/n) (Gaussian estimator), verified over seeds.
    """
    tol = tolerances['atmosphere_t2']['zernike']['initial']
    mask, Z, G = zernike_covariance(0.10, L0)
    a = zernike_initial(seed, tol['n'], logger)
    _zernike_report(report, a, G, tol, f'initial screens, r0 0.10, {tol["n"]} screens')


def test_zernike_variances_atmosphere(logger, report, tolerances, seed):
    """
    Same as the initial-screen test for the full 3-layer Atmosphere (r0 0.12 m total, layers 0/5/12 km with 60/30/10% Cn2,
    on-axis OPD after burn-in; samples 19 m apart along the layer motion). The sum of independent layers with
    sum r0_i^-5/3 = r0^-5/3 and the same L0 has the von Karman covariance of r0.
    """
    tol = tolerances['atmosphere_t2']['zernike']['atmosphere']
    mask, Z, G = zernike_covariance(A5_PROFILE['r0'], L0)
    a = zernike_atmosphere(seed, tol['n'], logger)
    _zernike_report(report, a, G, tol, f'Atmosphere 3 layers, r0 {A5_PROFILE["r0"]}, {tol["n"]} samples')


# ----------------------------------------------------------------------------------------------------------------------
# A9: Taylor frozen flow
# ----------------------------------------------------------------------------------------------------------------------

def frozen_flow(k, direction, seed, logger, steps=2000, r0=0.10, tau_max=108):
    """
    Temporal structure function of the on-axis OPD of a single layer moving k px/step (integer): ratio measured / D_phi(k tau px)
    at steps tau. Returns (tau, r [m], D measured [rad^2], D theory [rad^2]).
    """
    tel = Telescope(resolution=NX, diameter=D, samplingTime=TS, fov=0, logger=logger)
    atm = Atmosphere(r0=r0, L0=L0, fractionalR0=[1.0], altitude=[0.0], windDirection=[direction],
                     windSpeed=[k * PIX / TS], telescope=tel, zenith=0, logger=logger)
    atm.initializeAtmosphere(randomState=seed)
    src = Source('V0', 0, coordinates=[0, 0], logger=logger)
    tau = np.array([t for t in (1, 2, 3, 5, 8, 13, 21, 34, 54, 80, 108) if k * t <= 108 and t <= tau_max])
    for _ in range(2 * NX):
        atm.update()
    buf = []
    acc, cnt = np.zeros(len(tau)), 0
    for t in range(steps):
        atm.update()
        buf.append(np.array(atm.getOPD(src), dtype=np.float32) * np.float32(K_OPD))
        if len(buf) > tau[-1] + 1:
            buf.pop(0)
        if len(buf) == tau[-1] + 1:
            acc += [np.mean((buf[-1] - buf[-1 - s]) ** 2, dtype=np.float64) for s in tau]
            cnt += 1
    r = k * tau * PIX
    return tau, r, acc / cnt, vk_dstruct(r, r0, L0)


@pytest.mark.parametrize('k,direction', [
    (1, 0), (1, 90),
    (2, 0)])
def test_taylor_frozen_flow(logger, report, tolerances, seed, k, direction):
    """
    Frozen flow at k integer px/step (v = k px / Ts): the temporal structure function of the on-axis OPD of one layer,
    D(tau) = <[phi(t + tau) - phi(t)]^2> = D_phi(v tau Ts), averaged over the 216 x 216 px and 2000 steps after a burn-in
    of 2 nx steps, equals the von Karman D(r) at r = k tau px (r up to 2.07 m = nx/2 px). The direction of motion is
    irrelevant for D(tau). Tolerance per range of r (<= nx/4 px and above): 3.5 sigma over seeds.
    """
    tol = tolerances['atmosphere_t2']['frozen_flow']
    tau, r, dm, dt = frozen_flow(k, direction, seed, logger)
    ratio = dm / dt
    small = r / PIX <= NX // 4
    # Per-delay tolerances calibrated with k = 1 (the larger estimator spread); indexed by delay tau, conservative for k > 1
    tol_tau = np.r_[tol['small_r'], tol['large_r']][:len(ratio)]
    report.check('D(tau)/D_vK(v tau), r <= nx/4 px', ratio[small], expected=1, atol=tol_tau[small])
    report.check('D(tau)/D_vK(v tau), nx/4 < r <= nx/2 px', ratio[~small], expected=1, atol=tol_tau[~small])
    report.metric('effective speed / requested (from the mean ratio on r <= nx/4, D ~ v^5/3)', np.mean(ratio[small]) ** 0.6)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    rr = np.logspace(np.log10(r[0]), np.log10(r[-1]), 80)
    ax[0].loglog(tau * TS * 1e3, dm, 'o', label='SAOS')
    ax[0].loglog(rr / (k * PIX) * TS * 1e3, vk_dstruct(rr, 0.10, L0), 'k', label=f'D_vK(v tau), v = {k} px/step')
    ax[0].set_xlabel('tau [ms]'); ax[0].set_ylabel('D(tau) [rad^2]'); ax[0].legend(fontsize=7)
    ax[1].semilogx(tau * TS * 1e3, ratio, 'o-')
    ax[1].axhline(1, color='k', lw=0.6)
    ax[1].fill_between(tau * TS * 1e3, 1 - tol_tau, 1 + tol_tau, color='0.92', label='tolerance')
    ax[1].set_xlabel('tau [ms]'); ax[1].set_ylabel('D_SAOS / D_vK(v tau)')
    ax[1].set_title(f'{k} px/step towards {direction} deg', fontsize=9)
    report.figure(fig, 'frozen_flow')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# A11: anisoplanatism
# ----------------------------------------------------------------------------------------------------------------------

A11 = dict(r0=0.12, frac=[0.5, 0.2, 0.2, 0.1], theta1=5.0, n_px=[0, 2, 5, 10], theta_mult=[1, 2, 3, 4], frac_px=0.02)


def a11_profile():
    """
    Layer altitudes chosen so that the footprint offset of the 5 arcsec source, h tan(theta) / (D/N) = n + 0.02 px with
    n = 0, 2, 5, 10, so that the rounding of the offsets to the layer pixel (integer-pixel footprint model, tested in
    test_atmosphere.py) stays below 0.02 j px for theta = j x 5 arcsec (j <= 4) and does not mix with the anisoplanatic
    physics tested here.
    """
    t1 = A11['theta1'] / RAD2ARCSEC
    return [(n + A11['frac_px']) * PIX / np.tan(t1) if n else 0.0 for n in A11['n_px']]


def aniso_theory(alt, theta_arcsec, r0, frac, L0):
    """Pointwise variance and pupil-piston-removed variance of the OPD difference (rad^2 at 500 nm), shift along +x."""
    mask = circle_mask(NX)
    N = mask.sum()
    Ac = fftconvolve(mask.astype(float), mask[::-1, ::-1].astype(float))            # pairs of pupil pixels at each lag
    lag = np.arange(-(NX - 1), NX) * PIX
    LY, LX = np.meshgrid(lag, lag, indexing='ij')                                   # (row lag, col lag)
    out_pw, out_pr = [], []
    for th in theta_arcsec:
        pw = pm = 0.0
        for h, f in zip(alt, frac):
            r0i = r0 * f ** (-3 / 5)
            s = h * np.tan(th / RAD2ARCSEC)
            pw += vk_dstruct(s, r0i, L0) if s > 0 else 0.0
            if s > 0:
                cov = 2 * vk_cov(np.hypot(LY, LX), r0i, L0) - vk_cov(np.hypot(LY, LX + s), r0i, L0) - vk_cov(np.hypot(LY, LX - s), r0i, L0)
                pm += np.sum(Ac * cov) / N ** 2
        out_pw.append(pw); out_pr.append(pw - pm)
    return np.array(out_pw), np.array(out_pr)


def aniso_measure(seed, n, logger, n_inst=4, spacing=1000):
    """Mean over `n` samples and the circular pupil of dOPD^2 (pointwise, and after removing the pupil piston) [rad^2 at 500 nm]."""
    alt = a11_profile()
    thetas = [A11['theta1'] * j for j in A11['theta_mult']]
    fov = 2 * max(thetas) + 10.0
    tel = Telescope(resolution=NX, diameter=D, samplingTime=TS, fov=fov, logger=logger)
    mask = circle_mask(NX)
    src0 = Source('V0', 0, coordinates=[0, 0], logger=logger)
    srcs = [Source('V0', 0, coordinates=[th, 0], logger=logger) for th in thetas]

    def one(i):
        atm = Atmosphere(r0=A11['r0'], L0=L0, fractionalR0=A11['frac'], altitude=alt, windDirection=[0, 90, 0, 90],
                         windSpeed=[ROW_SPEED] * 4, telescope=tel, zenith=0, logger=logger)
        atm.initializeAtmosphere(randomState=seed + 104729 * i)
        advance(atm, spacing)
        pw, pr = [], []
        for _ in range(n // n_inst):
            advance(atm, spacing)
            o0 = np.array(atm.getOPD(src0)) * K_OPD
            dd = [(np.array(atm.getOPD(s)) * K_OPD - o0)[mask] for s in srcs]
            pw.append([np.mean(x ** 2) for x in dd])
            pr.append([np.mean((x - x.mean()) ** 2) for x in dd])
        return pw, pr
    res = pmap(one, range(n_inst), workers=n_inst)
    pw = np.concatenate([r[0] for r in res]); pr = np.concatenate([r[1] for r in res])
    return np.array(thetas), pw.mean(0), pr.mean(0), alt


def test_anisoplanatism(logger, report, tolerances, seed):
    """
    OPD difference between the on-axis direction and directions at 5, 10, 15, 20 arcsec (4-layer atmosphere, r0 0.12 m,
    layers at 0, ~1.6, ~4, ~8 km): <dOPD^2> = sum_i D_i(h_i tan theta) = sum_i 2 [C_i(0) - C_i(h_i theta)] with
    r0_i = r0 f_i^-3/5; after removing the pupil mean (piston) the variance is reduced by the pupil average of
    sum_i [2 C_i(d) - C_i(d + s_i) - C_i(d - s_i)], computed here with the autocorrelation of the circular pupil.
    Altitudes tuned to integer-pixel footprint offsets (see a11_profile). Tolerance: 3.5 sigma over seeds.
    """
    tol = tolerances['atmosphere_t2']['aniso']
    th, pw, pr, alt = aniso_measure(seed, tol['n'], logger)
    tpw, tpr = aniso_theory(alt, th, A11['r0'], A11['frac'], L0)
    for t, a, b, c, d in zip(th, pw, tpw, pr, tpr):
        report.check(f'theta {t:g}": pointwise <dOPD^2> / theory', a / b, expected=1, atol=tol['rel'])
        report.check(f'theta {t:g}": piston-removed <dOPD^2> / theory', c / d, expected=1, atol=tol['rel'])
    report.metric('layer altitudes [m]', alt)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(th, tpw, 'k-o', label='theory, pointwise'); ax[0].loglog(th, tpr, 'k--s', mfc='none', label='theory, piston removed')
    ax[0].loglog(th, pw, 'C0^', label='SAOS pointwise'); ax[0].loglog(th, pr, 'C3v', label='SAOS piston removed')
    ax[0].set_xlabel('theta [arcsec]'); ax[0].set_ylabel('<dOPD^2> [rad^2 at 500 nm]'); ax[0].legend(fontsize=7)
    ax[1].semilogx(th, pw / tpw, 'C0^-', label='pointwise'); ax[1].semilogx(th, pr / tpr, 'C3v-', label='piston removed')
    ax[1].axhspan(1 - tol['rel'], 1 + tol['rel'], color='0.92'); ax[1].axhline(1, color='k', lw=0.6)
    ax[1].set_xlabel('theta [arcsec]'); ax[1].set_ylabel('SAOS / theory'); ax[1].legend(fontsize=7)
    report.figure(fig, 'anisoplanatism')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# A12: seeing-limited long exposure
# ----------------------------------------------------------------------------------------------------------------------

FW = 1.5                                                     # pixels per lambda/D of the science camera (216 px: 3.6 arcsec field)


def fwhm_radial(img, guess_px, centre=None, dr=1.0):
    """
    FWHM [px] of a centred, isotropic long exposure from its azimuthal mean about the optical axis (pixel nx/2): the peak is
    the quadratic fit of the profile for r < 0.35 guess (robust to speckle noise, unlike the maximum pixel), the half maximum
    is crossed by linear interpolation. The same estimator is applied to the exact expectation and to the SAOS frames.
    """
    n0, n1 = img.shape
    c = (n0 // 2, n1 // 2) if centre is None else centre
    rr = np.hypot(*np.indices(img.shape) - np.array(c)[:, None, None]).ravel()
    edges = np.arange(-dr / 2, 4 * guess_px, dr)
    idx = np.digitize(rr, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    prof = np.bincount(idx[ok], img.ravel()[ok], len(edges) - 1) / np.bincount(idx[ok], minlength=len(edges) - 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    sel = rc < 0.35 * guess_px
    i0 = np.polyfit(rc[sel] ** 2, prof[sel], 1)[1]
    k = np.argmax(prof < i0 / 2)
    return 2 * (rc[k - 1] + (prof[k - 1] - i0 / 2) / (prof[k - 1] - prof[k]) * dr)


def tokovinin_fwhm(r0, L0, lam=LAM):
    """Long-exposure seeing FWHM [arcsec]: 0.976 lambda/r0 sqrt(1 - 2.183 (r0/L0)^0.356) (Tokovinin 2002)."""
    return 0.976 * lam / r0 * np.sqrt(1 - 2.183 * (r0 / L0) ** 0.356) * RAD2ARCSEC


def le_reference_fwhm(r0, L0):
    """
    FWHM [arcsec] of the expected long exposure of the finite circular aperture, on the camera grid: FFT of
    T_tel(rho) exp(-D_vK(rho)/2) with the pupil autocorrelation T_tel of the sampled 216 px disk.
    """
    nfft = int(round(FW * NX))
    mask = circle_mask(NX).astype(float)
    T = fftconvolve(mask, mask[::-1, ::-1]) / mask.sum()
    lag = np.arange(-(nfft // 2), nfft - nfft // 2)
    otf = np.zeros((nfft, nfft))
    LY, LX = np.meshgrid(lag, lag, indexing='ij')
    inside = (np.abs(LY) < NX) & (np.abs(LX) < NX)
    otf[inside] = T[LY[inside] + NX - 1, LX[inside] + NX - 1]
    otf *= np.exp(-0.5 * vk_dstruct(np.hypot(LY, LX) * PIX, r0, L0))
    psf = np.fft.fftshift(np.real(np.fft.fft2(np.fft.ifftshift(otf))))
    s = (nfft - NX) // 2
    psf = psf[s:s + NX, s:s + NX]
    ps = LAM / D * RAD2ARCSEC / FW
    return fwhm_radial(psf, tokovinin_fwhm(r0, L0) / ps) * ps


def seeing_le(r0, seed, logger, n_chunks=6, n_per=80, spacing=800):
    """Long-exposure FWHM [arcsec] of n_chunks x n_per ScienceCam frames of one layer (r0 at 500 nm), no AO; returns (fwhm, le frame)."""
    tel = Telescope(resolution=NX, diameter=D, samplingTime=TS, fov=0, logger=logger)
    src = Source('V0', 0, coordinates=[0, 0], logger=logger)
    ps = src.wavelength / D * RAD2ARCSEC / FW

    def one(i):
        atm = Atmosphere(r0=r0, L0=L0, fractionalR0=[1.0], altitude=[0.0], windDirection=[0], windSpeed=[ROW_SPEED],
                         telescope=tel, zenith=0, logger=logger)
        atm.initializeAtmosphere(randomState=seed + 15485863 * i)
        cam = ScienceCam(fieldOfView=NX * ps, plate_scale=ps, samplingTime=TS, telescope=tel, logger=logger, device='cpu')
        advance(atm, spacing)
        acc = np.zeros((cam.nPix, cam.nPix))
        for _ in range(n_per):
            advance(atm, spacing)
            acc += cam.get_frame(src, np.array(atm.getOPD(src)) * (2 * np.pi / src.wavelength))
        return acc
    le = np.sum(pmap(one, range(n_chunks), workers=n_chunks), axis=0) / (n_chunks * n_per)
    return fwhm_radial(le, tokovinin_fwhm(r0, L0) / ps) * ps, le


@pytest.mark.parametrize('r0', [0.10, 0.15, 0.25])
def test_seeing_limited_fwhm(logger, report, tolerances, seed, r0):
    """
    No AO: the long exposure of 480 ScienceCam frames (one layer, L0 25 m, source at 500 nm, 1.5 px per lambda/D, 3.6 arcsec
    field) has the FWHM of Tokovinin (2002), 0.976 lambda/r0 sqrt(1 - 2.183 (r0/L0)^0.356), valid for r0 << L0. Frames are
    taken 16 m apart along the motion (tilt wander included). The expected long exposure of the finite aperture
    (FFT of T_tel exp(-D_vK/2)) is computed independently; the Tokovinin formula is checked against it first
    (accuracy of the reference), then SAOS against the formula within 3.5 sigma over seeds.
    """
    tol = tolerances['atmosphere_t2']['seeing']
    ref = le_reference_fwhm(r0, L0)
    tok = tokovinin_fwhm(r0, L0)
    report.check('reference: Tokovinin / finite-aperture expectation', tok / ref, expected=1, atol=tol['formula_accuracy'])
    fw, le = seeing_le(r0, seed, logger)
    report.metric('FWHM SAOS / Tokovinin', fw / tok)
    report.check('FWHM SAOS / finite-aperture expectation', fw / ref, expected=1, atol=tol['rel'])
    report.metric('FWHM SAOS [arcsec]', fw, unit='arcsec'); report.metric('FWHM Tokovinin [arcsec]', tok, unit='arcsec')
    report.metric('FWHM finite-aperture expectation [arcsec]', ref, unit='arcsec')
    ps = LAM / D * RAD2ARCSEC / FW
    c = np.unravel_index(np.argmax(le), le.shape)
    x = (np.arange(le.shape[1]) - c[1]) * ps
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].imshow(le, extent=[x[0], x[-1], x[-1], x[0]], origin='upper'); ax[0].set_xlabel('arcsec'); ax[0].set_title('long exposure', fontsize=9)
    ax[1].plot(x, le[c[0]] / le.max(), label='SAOS (row through peak)')
    for v, ls, lab in ((tok, '--', 'Tokovinin'), (ref, ':', 'finite aperture')):
        ax[1].plot([-v / 2, v / 2], [0.5, 0.5], ls, lw=2, label=f'{lab} FWHM {v:.3f}"')
    ax[1].plot([-fw / 2, fw / 2], [0.45, 0.45], 'k-', lw=2, label=f'SAOS FWHM {fw:.3f}"')
    ax[1].set_xlabel('arcsec'); ax[1].legend(fontsize=7); ax[1].set_title(f'r0 {r0} m', fontsize=9)
    report.figure(fig, 'seeing_le')
    report.assert_ok()
