"""
T1 validation of the atmosphere: von Karman phase-screen statistics (covariance, structure function, PSD, r0/L0 fit),
extrusion, layer scaling with the profile and zenith angle, screen motion (speed and direction), footprints of
off-axis sources, reproducibility and save/load.

Modules under test: SAOS.infinitePhaseScreen.PhaseScreenVonKarman, SAOS.Atmosphere, SAOS.Telescope (pupil).
Phase screens are in rad at 500 nm; OPD in m.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

import theory
from citools import integer_shift, known_issue, structure_function
from SAOS.Atmosphere import Atmosphere
from SAOS.infinitePhaseScreen import PhaseScreenVonKarman
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

R0, L0 = 0.10, 25.0


def _lags(n_max):
    return np.unique(np.round(np.logspace(0, np.log10(n_max), 20)).astype(int))


# ----------------------------------------------------------------------------------------------------------------------
# Phase-screen statistics
# ----------------------------------------------------------------------------------------------------------------------

def test_theory_references_self_consistent(report, tolerances):
    """
    The analytic references used by the suite agree with each other: the closed-form von Karman covariance equals the
    Hankel transform of the PSD 0.0229 r0^-5/3 (f^2 + L0^-2)^-11/6, and D(r) tends to 6.88 (r/r0)^5/3 for r << L0.
    """
    tol = tolerances['atmosphere']['theory']
    for r in (0.0, 0.05, 0.5, 5.0):
        num = quad(lambda f: 2 * np.pi * f * theory.vk_psd(f, R0, L0) * j0(2 * np.pi * f * r), 0, np.inf, limit=2000)[0]
        report.check(f'C({r} m) Hankel / closed form', num / theory.vk_covariance(r, R0, L0), expected=1, atol=tol['hankel'])
    r = np.array([1e-3, 1e-2])
    report.check('D_vK / D_Kolmogorov (r << L0)', theory.vk_structure(r, R0, 1e5) / theory.kolmogorov_structure(r, R0),
                 expected=1, atol=tol['kolmogorov_limit'])
    report.assert_ok()


def test_saos_phase_covariance_matches_theory(logger, report):
    """PhaseScreenVonKarman.phase_covariance (Assemat & Wilson eq. 5) equals the von Karman covariance (rtol 1e-6)."""
    scr = PhaseScreenVonKarman(8, 0.1, R0, L0, random_seed=1, logger=logger)
    r = np.r_[0.0, np.logspace(-3, 2, 30)]
    ratio = scr.phase_covariance(r.copy(), R0, L0) / theory.vk_covariance(r, R0, L0)
    report.check('C_SAOS / C_vK', ratio, expected=1, rtol=1e-6)
    report.metric('C(0) / (L0/r0)^5/3', theory.vk_variance(R0, L0) / (L0 / R0) ** (5 / 3))
    report.assert_ok()


def _fit_and_plot(report, tol, label, r, d, n_real, extra_curves=None):
    ratio = d / theory.vk_structure(r, R0, L0)
    r0j, L0j = theory.fit_r0_L0(r, d, R0, L0)
    r0f, _ = theory.fit_r0_L0(r, d, R0, L0, fix_L0=True)
    report.check('fitted r0 / true (joint r0, L0 fit)', r0j / R0, expected=1, atol=tol['r0_joint'], trend=True)
    report.metric('fitted L0 / true (joint fit; weakly constrained by a screen of ~L0 size, see the outer-scale tests)', L0j / L0)
    report.check('fitted r0 / true (L0 fixed)', r0f / R0, expected=1, atol=tol['r0_fixed'], trend=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    rr = np.logspace(np.log10(r[0]), np.log10(r[-1]), 100)
    ax[0].loglog(r, d, 'o', ms=4, label=f'SAOS ({n_real} realisations)')
    ax[0].loglog(rr, theory.vk_structure(rr, R0, L0), 'k', label=f'von Karman r0 {R0}, L0 {L0:g}')
    ax[0].loglog(rr, theory.vk_structure(rr, r0j, L0j), '--', color='C3', label=f'fit r0 {r0j:.4f}, L0 {L0j:.1f}')
    ax[0].set_xlabel('r [m]'); ax[0].set_ylabel('D(r) [rad^2]'); ax[0].legend(fontsize=7)
    ax[1].semilogx(r, ratio, 'o-', ms=4, label='mean of x and y')
    for name, curve in (extra_curves or {}).items():
        ax[1].semilogx(r, curve, '.', ms=3, label=name)
    ax[1].axhspan(1 - tol['D_ratio_small_r'], 1 + tol['D_ratio_small_r'], color='0.9', label='tolerance (small r)')
    ax[1].axhline(1, color='k', lw=0.6); ax[1].set_xlabel('r [m]'); ax[1].set_ylabel('D_SAOS / D_vK'); ax[1].legend(fontsize=7)
    fig.suptitle(label, fontsize=9)
    report.figure(fig, 'structure_function')
    return ratio


def test_initial_screen_structure_function(logger, report, tolerances, seed):
    """
    Initial FFT screen (subharmonics, 4x oversized grid, cropped): the mean D(r) of 200 screens of 64 px (4 m) equals
    the von Karman D(r) within tolerance for r >= 4 px, and the joint r0/L0 fit recovers the generation parameters.
    L0 is only recorded here (a 4 m screen does not span the turnover; see test_outer_scale_*). Small lags are recorded only: an FFT screen lacks the power above Nyquist, so D(1 px) is ~0.92 of the continuous
    theory (the extruded rows, built from the sampled covariance, do not have this deficit).
    """
    tol = tolerances['atmosphere']['initial_screen']
    nx, pix, n = 64, 4.0 / 64, 200
    scr = PhaseScreenVonKarman(nx, pix, R0, L0, random_seed=seed, logger=logger)
    rng = np.random.default_rng(seed)
    lags = _lags(nx // 2)
    dx, dy = np.mean([structure_function(scr.initial_screen(rng), lags) for _ in range(n)], axis=0)
    r = lags * pix
    ratio = _fit_and_plot(report, tol, f'Initial screen: {nx} px, {pix * 100:.2f} cm/px, {n} realisations', r,
                          0.5 * (dx + dy), n, {'x': dx / theory.vk_structure(r, R0, L0), 'y': dy / theory.vk_structure(r, R0, L0)})
    small, large = (lags >= 4) & (lags <= nx // 4), lags > nx // 4
    report.check('D/D_vK, 4 px <= r <= nx/4', ratio[small], expected=1, atol=tol['D_ratio_small_r'])
    report.check('D/D_vK, nx/4 < r <= nx/2', ratio[large], expected=1, atol=tol['D_ratio_large_r'])
    report.check('x/y anisotropy: mean D_x/D_y, 4 px <= r <= nx/4', np.mean((dx / dy)[small]), expected=1, atol=tol['anisotropy'])
    report.metric('D/D_vK at 1 px', ratio[0])
    report.assert_ok()


def test_extruded_screen_structure_function(logger, report, tolerances, seed):
    """
    Screen extruded row by row (Assemat & Wilson with far-row stencil): after a burn-in of nx rows, D(r) along and
    across the motion, sampled every nx rows over 4000 rows of a 32 px screen, equals the von Karman D(r) for
    1 px <= r <= nx/2.
    """
    tol = tolerances['atmosphere']['extruded_screen']
    nx, pix, n_rows = 32, 2.0 / 32, 4000
    scr = PhaseScreenVonKarman(nx, pix, R0, L0, random_seed=seed, logger=logger)
    for _ in range(nx):
        scr.add_row(1)
    lags = _lags(nx // 2)
    samples = []
    for k in range(n_rows):
        scr.add_row(1)
        if (k + 1) % nx == 0:
            samples.append(structure_function(np.array(scr.scrn), lags))
    dx, dy = np.mean(samples, axis=0)
    r = lags * pix
    ratio = _fit_and_plot(report, tol, f'Extruded screen: {nx} px, {n_rows} rows, {len(samples)} samples', r,
                          0.5 * (dx + dy), len(samples),
                          {'x (along rows)': dx / theory.vk_structure(r, R0, L0), 'y (motion)': dy / theory.vk_structure(r, R0, L0)})
    report.check('D/D_vK, 1 px <= r <= nx/4', ratio[lags <= nx // 4], expected=1, atol=tol['D_ratio_small_r'])
    report.check('D/D_vK, nx/4 < r <= nx/2', ratio[lags > nx // 4], expected=1, atol=tol['D_ratio_large_r'])
    report.check('D_x/D_vK (across the motion)', dx / theory.vk_structure(r, R0, L0), expected=1, atol=tol['D_ratio_axis'])
    report.check('D_y/D_vK (along the motion)', dy / theory.vk_structure(r, R0, L0), expected=1, atol=tol['D_ratio_axis'])
    report.metric('D_y/D_vK at nx/2', dy[-1] / theory.vk_structure(r[-1], R0, L0), trend=True)
    report.assert_ok()


def _outer_scale_fit(report, tol, label, r, d, r0, L0, n_real):
    """Joint r0/L0 fit of a structure function from lag 2 px (1 px is below the sampling of a pixelated screen)."""
    r0f, L0f = theory.fit_r0_L0(r[1:], d[1:], r0, L0)
    report.check('fitted L0 / true', L0f / L0, expected=1, atol=tol['L0_atol'], trend=True)
    report.check('fitted r0 / true', r0f / r0, expected=1, atol=tol['r0_atol'], trend=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    rr = np.logspace(np.log10(r[0]), np.log10(r[-1]), 100)
    ax.loglog(r, d, 'o', ms=4, label=f'SAOS ({n_real} realisations)')
    ax.loglog(rr, theory.vk_structure(rr, r0, L0), 'k', label=f'von Karman r0 {r0:g}, L0 {L0:g}')
    ax.loglog(rr, theory.vk_structure(rr, r0f, L0f), '--', color='C3', label=f'fit r0 {r0f:.3f}, L0 {L0f:.1f}')
    ax.loglog(rr, theory.kolmogorov_structure(rr, r0), ':', color='0.5', label='Kolmogorov (L0 = inf)')
    ax.set_xlabel('r [m]'); ax.set_ylabel('D(r) [rad^2]'); ax.legend(fontsize=7); ax.set_title(label, fontsize=8)
    report.figure(fig, 'outer_scale_fit')


def test_outer_scale_initial_screen(logger, report, tolerances, seed):
    """
    Outer-scale recovery from the initial FFT screen. The screen must span the turnover of the von Karman spectrum:
    128 px of 0.5 m = 64 m = 2.6 L0 (L0 = 25 m, r0 = 0.5 m, so that the 0.5 m pixel is resolved: pixel / r0 = 1). The
    joint r0/L0 fit of the mean D(r) of 40 screens (lags 2 px .. nx/2 = 32 m) must return the generation parameters,
    with the tolerance of the fit statistics (3.5 sigma over seeds).
    """
    tol = tolerances['atmosphere']['outer_scale_initial']
    r0, L0_, nx, pix, n = 0.5, 25.0, 128, 0.5, 40
    scr = PhaseScreenVonKarman(nx, pix, r0, L0_, random_seed=seed, logger=logger)
    rng = np.random.default_rng(seed)
    lags = _lags(nx // 2)
    dx, dy = np.mean([structure_function(scr.initial_screen(rng), lags) for _ in range(n)], axis=0)
    r = lags * pix
    _outer_scale_fit(report, tol, f'Initial screen {nx} px x {pix} m, {n} screens', r, 0.5 * (dx + dy), r0, L0_, n)
    report.metric('D/D_vK at 32 m (nx/2)', 0.5 * (dx + dy)[-1] / theory.vk_structure(r[-1], r0, L0_))
    report.assert_ok()


def test_outer_scale_extruded_screen(logger, report, tolerances, seed):
    """
    Outer-scale recovery from the extruded screen: 64 px of 1 m (2.6 L0, r0 = 1 m, L0 = 25 m), D(r) averaged over the
    two axes and over screens sampled every nx rows along 20000 extruded rows (312 samples); joint r0/L0 fit from
    2 px to nx/2 = 32 m, with the tolerance of the fit statistics (3.5 sigma over seeds).
    """
    tol = tolerances['atmosphere']['outer_scale_extruded']
    r0, L0_, nx, pix, n_rows = 1.0, 25.0, 64, 1.0, 20000
    scr = PhaseScreenVonKarman(nx, pix, r0, L0_, random_seed=seed, logger=logger)
    for _ in range(nx):
        scr.add_row(1)
    lags = _lags(nx // 2)
    samples = []
    for k in range(n_rows):
        scr.add_row(1)
        if (k + 1) % nx == 0:
            samples.append(structure_function(np.array(scr.scrn), lags))
    dx, dy = np.mean(samples, axis=0)
    r = lags * pix
    _outer_scale_fit(report, tol, f'Extruded screen {nx} px x {pix} m, {len(samples)} samples', r, 0.5 * (dx + dy), r0, L0_, len(samples))
    report.assert_ok()


def test_extruded_screen_low_frequencies(logger, report, tolerances, seed):
    """
    Large-scale structure function of the extruded screen: D(r) at r = nx/8, nx/4, nx/2 px (32 px screen of 2 m,
    r0 = 0.1 m, L0 = 25 m) along (y) and across (x) the extrusion, averaged over 16 independent screens extruded for
    40000 rows each (sampled every nx rows), must equal the von Karman D(r). The earlier 4000-row test hinted at a
    -4% deficit at nx/2 (about 3 sigma); this ensemble resolves it (standard error of the mean about 0.009).
    """
    tol = tolerances['atmosphere']['extruded_lowfreq']
    nx, pix, n_rows, n_seeds = 32, 2.0 / 32, 40000, 16
    lags = np.array([nx // 8, nx // 4, nx // 2])
    th = theory.vk_structure(lags * pix, R0, L0)
    rows = []
    for k in range(n_seeds):
        scr = PhaseScreenVonKarman(nx, pix, R0, L0, random_seed=seed + k, logger=logger)
        for _ in range(nx):
            scr.add_row(1)
        s = []
        for j in range(n_rows):
            scr.add_row(1)
            if (j + 1) % nx == 0:
                s.append(structure_function(np.array(scr.scrn), lags))
        dx, dy = np.mean(s, axis=0)
        rows.append(np.r_[dx / th, dy / th])
    rows = np.array(rows)
    mean, sem = rows.mean(0), rows.std(0, ddof=1) / np.sqrt(n_seeds)
    for name, i in (('x (across)', 0), ('y (along)', 3)):
        for j, lag in enumerate(lags):
            report.metric(f'D_{name[0]}/D_vK at {lag} px, standard error', sem[i + j])
        report.check(f'D_{name}/D_vK at nx/8, nx/4, nx/2', mean[i:i + 3], expected=1, atol=tol['atol'])
    report.assert_ok()


@pytest.mark.parametrize('axis', ['row', 'col'])
def test_extrusion_directions_are_mirror_images(logger, report, axis, seed):
    """
    Extruding towards -1 from the mirrored initial screen gives exactly the mirror image of extruding towards +1
    (same random stream): the stencils and A/B matrices are consistent for both directions of motion.
    """
    nx = 16
    a = PhaseScreenVonKarman(nx, 0.05, R0, L0, random_seed=seed, logger=logger)
    b = PhaseScreenVonKarman(nx, 0.05, R0, L0, random_seed=seed, logger=logger)
    flip = (lambda m: m[::-1, :]) if axis == 'row' else (lambda m: m[:, ::-1])
    b._scrn = flip(np.array(a._scrn)).copy()
    for _ in range(3 * nx):
        (a.add_row(1), b.add_row(-1)) if axis == 'row' else (a.add_col(1), b.add_col(-1))
    report.check('max |screen_+ - mirror(screen_-)| [rad]', np.max(np.abs(np.array(a.scrn) - flip(np.array(b.scrn)))), lo=0, hi=1e-10)
    report.assert_ok()


def test_screen_psd(logger, report, tolerances, seed):
    """
    Radially averaged PSD of 200 initial screens (64 px, Hann window) follows 0.0229 r0^-5/3 (f^2 + L0^-2)^-11/6:
    amplitude ratio within tolerance and log-log slope -11/3 in the band 6/L <= f <= 0.3/dx (lower bins: window leakage).
    """
    tol = tolerances['atmosphere']['psd']
    nx, pix, n = 64, 4.0 / 64, 200
    scr = PhaseScreenVonKarman(nx, pix, R0, L0, random_seed=seed, logger=logger)
    rng = np.random.default_rng(seed)
    w1 = np.hanning(nx)
    w = np.outer(w1, w1)
    psd2 = np.zeros((nx, nx))
    for _ in range(n):
        s = scr.initial_screen(rng)
        psd2 += np.abs(np.fft.fft2((s - s.mean()) * w)) ** 2
    psd2 *= pix ** 2 / (n * (w ** 2).sum())                   # periodogram normalised to rad^2 m^2
    f = np.fft.fftfreq(nx, pix)
    fr = np.hypot(*np.meshgrid(f, f))
    df = 1 / (nx * pix)
    bins = np.arange(0.5, nx // 2) * df
    idx = np.digitize(fr.ravel(), bins)
    fc = np.array([fr.ravel()[idx == k].mean() for k in range(1, len(bins))])
    pc = np.array([psd2.ravel()[idx == k].mean() for k in range(1, len(bins))])
    band = (fc >= 6 * df) & (fc <= 0.3 / pix)
    ratio = pc[band] / theory.vk_psd(fc[band], R0, L0)
    slope = np.polyfit(np.log(fc[band]), np.log(pc[band]), 1)[0]
    slope_th = np.polyfit(np.log(fc[band]), np.log(theory.vk_psd(fc[band], R0, L0)), 1)[0]
    report.check('PSD_SAOS / PSD_vK in band', ratio, expected=1, atol=tol['amplitude'])
    report.check('log-log slope (band)', slope, expected=slope_th, atol=tol['slope'])
    report.metric('theoretical slope in band (-11/3 asymptotically)', slope_th)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog(fc, pc, 'o', ms=4, label='SAOS (Hann, radial mean)')
    ff = np.logspace(np.log10(fc[0]), np.log10(fc[-1]), 100)
    ax.loglog(ff, theory.vk_psd(ff, R0, L0), 'k', label='von Karman')
    ax.axvspan(fc[band][0], fc[band][-1], color='0.93', label='test band')
    ax.set_xlabel('f [cycles/m]'); ax.set_ylabel('PSD [rad^2 m^2]'); ax.legend(fontsize=7)
    report.figure(fig, 'psd')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Profile, zenith and layer geometry
# ----------------------------------------------------------------------------------------------------------------------

def test_layer_r0_and_altitude_scaling(logger, report):
    """
    Profile scaling along the line of sight at zenith angle z: layer r0_i = r0 cos(z)^0.6 f_i^-3/5, so that
    sum r0_i^-5/3 = (r0 cos(z)^0.6)^-5/3; layer altitude h_i / cos z; layer size D + 2 tan(fov/2) h_i; npix * pixel = size.
    """
    z = 30.0
    frac, alt = [0.5, 0.3, 0.2], [0.0, 4000.0, 12000.0]
    tel = Telescope(resolution=32, diameter=2.0, samplingTime=1e-3, fov=40, logger=logger)
    atm = Atmosphere(r0=R0, L0=L0, fractionalR0=frac, altitude=alt, windDirection=[0, 0, 0], windSpeed=[1, 1, 1],
                     telescope=tel, zenith=z, logger=logger)
    atm.initializeAtmosphere(randomState=1)
    zr = np.deg2rad(z)
    layers = [getattr(atm, f'layer_{i + 1}') for i in range(3)]
    r0_layers = np.array([L.fractionalR0 for L in layers])        # SAOS stores the layer r0 in `fractionalR0`
    report.check('layer r0 / theory', r0_layers / theory.layer_r0(R0, np.array(frac), zr), expected=1, rtol=1e-12)
    report.check('sum r0_i^-5/3 / (r0 cos^0.6 z)^-5/3', np.sum(r0_layers ** (-5 / 3)) / (R0 * np.cos(zr) ** 0.6) ** (-5 / 3),
                 expected=1, rtol=1e-12)
    report.check('layer altitude / (h / cos z)', [L.altitude / (h / np.cos(zr)) for L, h in zip(layers[1:], alt[1:])], expected=1, rtol=1e-12)
    d_fov = [2.0 + 2 * np.tan(np.deg2rad(40 / 3600) / 2) * L.altitude for L in layers]
    report.check('layer size / (D + 2 tan(fov/2) h)', [L.D_fov / d for L, d in zip(layers, d_fov)], expected=1, rtol=1e-5)   # 206265 vs exact rad
    report.check('(npix * pixel - layer size) / pixel (whole-pixel symmetric margin: 0 <= . < 2)',
                 [(L.npix * L.spatial_res - L.D_fov) / L.spatial_res for L in layers], lo=-1e-9, hi=2.0)
    report.metric('layer pixel / telescope pixel D/N - 1 (max)', max(abs(L.spatial_res / (2.0 / 32) - 1) for L in layers))
    report.assert_ok()


def test_layer_pixel_equals_pupil_pixel(logger, report):
    """
    The pupil is cut as res layer pixels, so the layer pixel pitch must equal the pupil pixel D / res (otherwise the
    pupil seen through the layer is not D wide): D_fov / npix = D / res for layers at several altitudes.
    """
    tel = Telescope(resolution=32, diameter=2.0, samplingTime=1e-3, fov=40, logger=logger)
    atm = Atmosphere(r0=R0, L0=L0, fractionalR0=[0.4, 0.3, 0.2, 0.1], altitude=[0.0, 3000.0, 7500.0, 12000.0],
                     windDirection=[0] * 4, windSpeed=[1] * 4, telescope=tel, zenith=0, logger=logger)
    atm.initializeAtmosphere(randomState=1)
    ratio = [getattr(atm, f'layer_{i + 1}').spatial_res / (2.0 / 32) for i in range(4)]
    report.check('layer pixel / (D / res), altitudes 0, 3, 7.5, 12 km', ratio, expected=1, rtol=1e-6)
    report.assert_ok()


def test_pupil_area(logger, report):
    """Circular pupil with central obstruction: area = pi/4 N^2 (1 - co^2) within 1% (N = 64 and 128)."""
    for n in (64, 128):
        tel = Telescope(resolution=n, diameter=1.0, centralObstruction=0.3, logger=logger)
        report.check(f'pupil area / theory (N={n})', tel.pupil.sum() / (np.pi / 4 * n ** 2 * (1 - 0.3 ** 2)), expected=1, atol=0.01)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Footprints and motion. Reference frame (sources): azimuth 0 -> +x = +column, azimuth 90 -> +y = +row.
# ----------------------------------------------------------------------------------------------------------------------

def _single_layer(logger, altitude, speed=0.0, direction=0.0, res=32, D=3.2, Ts=0.01, fov=60, seed=3):
    tel = Telescope(resolution=res, diameter=D, samplingTime=Ts, fov=fov, logger=logger)
    atm = Atmosphere(r0=R0, L0=L0, fractionalR0=[1.0], altitude=[altitude], windDirection=[direction],
                     windSpeed=[speed], telescope=tel, zenith=0, logger=logger)
    atm.initializeAtmosphere(randomState=seed)
    return tel, atm


ATM_WAVELENGTH = 500e-9        # wavelength at which the layer phase is defined (input of the model, rad at 500 nm)
FOOTPRINT_SLOPE = 0.05         # rad per metre of layer of the prescribed ramp


def _footprint_displacement(logger, d_m, az, h=10000.0):
    """
    Physical displacement (m) (along columns = +x, along rows = +y) of the footprint of a source at azimuth `az`
    whose line of sight crosses a layer at altitude h at distance d_m from the on-axis one (elevation
    arctan(d_m / h)). The layer holds a prescribed linear phase ramp phi = g x (or g y) [rad], g in rad/m of layer;
    the OPD of a source then equals g (x_footprint) lambda_atm / 2 pi, so the mean OPD difference between the off-axis
    and the on-axis source gives the displacement with no reference to any SAOS pixel policy.
    """
    # Telescope field chosen so that the layer is exactly 2 D wide (npix = 2 res, layer pixel = D / res to 1e-9), which
    # isolates the footprint rounding from the layer pixel quantisation (tested separately)
    fov = 2 * np.arctan(3.2 * (1 - 1e-9) / (2 * h)) / theory.ARCSEC
    tel, atm = _single_layer(logger, h, fov=fov)
    layer = atm.layer_1
    n = np.array(layer.screen.scrn).shape[0]
    coord = np.arange(n) * layer.spatial_res                 # physical coordinate [m] of the layer samples
    th = np.arctan(d_m / h) / theory.ARCSEC
    on_src = Source('V', 0, coordinates=[0, 0], logger=logger)
    off_src = Source('V', 0, coordinates=[th, az], logger=logger)
    out = []
    for ramp in (np.tile(coord, (n, 1)), np.tile(coord[:, None], (1, n))):       # along columns, along rows
        layer.screen._scrn[:n, :n] = FOOTPRINT_SLOPE * ramp
        out.append(np.mean(atm.getOPD(off_src) - atm.getOPD(on_src)) * 2 * np.pi / ATM_WAVELENGTH / FOOTPRINT_SLOPE)
    return out[0], out[1]


FOOTPRINT_PIX = 0.1           # layer pixel D / res of _footprint_displacement [m]


def footprint_model(d_m, az):
    """Integer-pixel footprint model: each component of h tan(theta) rounded to the nearest layer pixel."""
    rad = np.deg2rad(az)
    return (np.round(d_m * np.cos(rad) / FOOTPRINT_PIX) * FOOTPRINT_PIX, np.round(d_m * np.sin(rad) / FOOTPRINT_PIX) * FOOTPRINT_PIX)


@pytest.mark.parametrize('az', (0, 90, 180, 270))
def test_footprint_displacement_direction(logger, report, tolerances, az):
    """
    A source at 0.2 m of footprint offset (h tan(theta) at a 10 km layer, theta = 4.1 arcsec) samples the layer 0.2 m
    away along its azimuth: displacement (col, row) = d (cos az, sin az), measured on a prescribed linear phase ramp.
    (0.2 m is two telescope pixels of 0.1 m, but arctan then the 206265 arcsec/rad conversion leave 1.99999.. pixel:
    the footprint must round it to two pixels, not truncate it to one.)
    """
    tol = tolerances['atmosphere']['footprint']
    d = 0.2
    dc, dr = _footprint_displacement(logger, d, az)
    report.check('footprint displacement along columns [m]', dc, expected=d * np.cos(np.deg2rad(az)), atol=tol['atol_m'])
    report.check('footprint displacement along rows [m]', dr, expected=d * np.sin(np.deg2rad(az)), atol=tol['atol_m'])
    report.assert_ok()


@pytest.mark.parametrize('d_m,az', [(0.16, 0), (0.09, 90), (0.04, 37.0), (0.15, 215.0), (0.16, 127.0), (0.09, 300.0)])
def test_footprint_displacement_fractional(logger, report, tolerances, d_m, az):
    """
    Fractional footprint displacement: d = 0.04-0.16 m at 10 km (0.4-1.6 telescope pixels of 0.1 m) along azimuths
    from 0 to 300 deg (no component within 0.05 px of a half pixel). Integer-pixel model: each component of h tan(theta)
    is rounded to the nearest layer pixel, so the displacement equals the rounded model exactly and lies within half a
    pixel of the physical one, without bias. Measured on a prescribed linear phase ramp along each axis.
    """
    tol = tolerances['atmosphere']['footprint']
    rad = np.deg2rad(az)
    dc, dr = _footprint_displacement(logger, d_m, az)
    mc, mr = footprint_model(d_m, az)
    report.check('footprint displacement along columns - rounded model [m]', dc, expected=mc, atol=tol['atol_m'])
    report.check('footprint displacement along rows - rounded model [m]', dr, expected=mr, atol=tol['atol_m'])
    report.check('|displacement - h tan(theta)| / pixel, columns and rows (<= 0.5, integer-pixel model)',
                 np.abs([dc - d_m * np.cos(rad), dr - d_m * np.sin(rad)]) / FOOTPRINT_PIX, lo=0, hi=0.5 + 1e-6)
    report.assert_ok()




@pytest.mark.parametrize('px_per_step,direction', [
    (0.5, 0), (0.5, 180),
    (0.5, 90), (0.5, 270), (0.5, 45),
    (2.5, 0)])
def test_wind_motion(logger, report, px_per_step, direction):
    """
    Frozen flow: after N steps the layer content has moved by v N Ts / pixel along the wind direction, in the same
    frame as the source azimuths (0 deg -> +column, 90 deg -> +row), within +-1 px (integer-pixel motion).
    Measured by least squares between snapshots of the layer screen.
    """
    D, res, Ts = 3.2, 32, 0.01
    speed = px_per_step * (D / res) / Ts
    tel, atm = _single_layer(logger, 0.0, speed, direction, res=res, D=D, Ts=Ts)
    layer = atm.layer_1
    every = max(1, int(2 / px_per_step))             # <= 2.5 px between snapshots
    n_snap = 10
    track = [np.zeros(2)]
    prev = np.array(layer.screen.scrn)
    for _ in range(n_snap):
        for _ in range(every):
            atm.update()
        cur = np.array(layer.screen.scrn)
        track.append(track[-1] + integer_shift(prev, cur, max_shift=4))
        prev = cur
    track = np.array(track)
    steps = np.arange(n_snap + 1) * every
    v = speed * steps * Ts / layer.spatial_res
    exp_row, exp_col = v * np.sin(np.deg2rad(direction)), v * np.cos(np.deg2rad(direction))
    report.metric('requested speed', px_per_step, unit='px/step')
    report.check('content shift rows', track[-1, 0], expected=exp_row[-1], atol=1.0)
    report.check('content shift cols', track[-1, 1], expected=exp_col[-1], atol=1.0)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(steps, exp_col, 'k-', label='expected x (columns)')
    ax.plot(steps, exp_row, 'k--', label='expected y (rows)')
    ax.plot(steps, track[:, 1], 'o', color='C0', label='measured x')
    ax.plot(steps, track[:, 0], 's', color='C3', mfc='none', label='measured y')
    ax.set_xlabel('iteration'); ax.set_ylabel('content shift [px]'); ax.legend(fontsize=7)
    ax.set_title(f'wind {px_per_step} px/step towards {direction} deg', fontsize=9)
    report.figure(fig, 'shift_vs_time')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Reproducibility and save/load
# ----------------------------------------------------------------------------------------------------------------------

def _opd_after(atm, src, n):
    for _ in range(n):
        atm.update()
    return np.array(atm.getOPD(src))


def test_same_seed_reproducible(logger, report):
    """Two atmospheres with the same randomState produce identical OPD sequences; a different seed does not."""
    src = Source('V', 0, coordinates=[0, 0], logger=logger)
    a = _single_layer(logger, 0.0, 3.0, 30.0, seed=5)[1]
    b = _single_layer(logger, 0.0, 3.0, 30.0, seed=5)[1]
    c = _single_layer(logger, 0.0, 3.0, 30.0, seed=6)[1]
    oa, ob, oc = (_opd_after(x, src, 25) for x in (a, b, c))
    report.check('max |OPD_a - OPD_b| same seed [m]', np.max(np.abs(oa - ob)), lo=0, hi=0)
    report.check('rms(OPD_a - OPD_c) / rms(OPD_a), different seed', np.std(oa - oc) / np.std(oa), lo=0.5)
    report.assert_ok()


def test_save_load_initial_state(logger, report, tmp_path):
    """Atmosphere.save + load reproduces the layer screens exactly at the time of saving."""
    tel, atm = _single_layer(logger, 5000.0, 3.0, 30.0)
    atm.save(str(tmp_path / 'atm.h5'))
    atm2 = Atmosphere(r0=R0, L0=L0, fractionalR0=[1.0], altitude=[5000.0], windDirection=[30.0], windSpeed=[3.0],
                      telescope=tel, zenith=0, logger=logger)
    atm2.load(str(tmp_path / 'atm.h5'))
    report.check('max |screen_saved - screen_loaded| [rad]', np.max(np.abs(np.array(atm.layer_1.screen.scrn) - np.array(atm2.layer_1.screen.scrn))), lo=0, hi=0)
    report.assert_ok()


def test_save_load_continues_identically(logger, report, tmp_path):
    """A loaded atmosphere evolves exactly as the original one would have (same OPD after k updates)."""
    src = Source('V', 0, coordinates=[0, 0], logger=logger)
    tel, atm = _single_layer(logger, 0.0, 3.0, 30.0)
    for _ in range(7):
        atm.update()
    atm.save(str(tmp_path / 'atm.h5'))
    atm2 = Atmosphere(r0=R0, L0=L0, fractionalR0=[1.0], altitude=[0.0], windDirection=[30.0], windSpeed=[3.0],
                      telescope=tel, zenith=0, logger=logger)
    atm2.load(str(tmp_path / 'atm.h5'))
    o1, o2 = _opd_after(atm, src, 20), _opd_after(atm2, src, 20)
    report.check('rms(OPD_original - OPD_loaded) / rms(OPD) after 20 updates', np.std(o1 - o2) / np.std(o1), lo=0, hi=1e-12)
    report.assert_ok()


def test_save_load_many_layers_order(logger, report, tmp_path):
    """With 11 layers, every loaded layer keeps its altitude (layer order preserved)."""
    n = 11
    alt = list(np.linspace(0, 10000, n))
    tel = Telescope(resolution=16, diameter=1.6, samplingTime=0.01, fov=10, logger=logger)
    kw = dict(r0=R0, L0=L0, fractionalR0=[1 / n] * n, altitude=alt, windDirection=[0] * n, windSpeed=[1] * n,
              telescope=tel, zenith=0, logger=logger)
    atm = Atmosphere(**kw)
    atm.initializeAtmosphere(randomState=2)
    atm.save(str(tmp_path / 'atm.h5'))
    atm2 = Atmosphere(**kw)
    atm2.load(str(tmp_path / 'atm.h5'))
    loaded = [getattr(atm2, f'layer_{i + 1}').altitude for i in range(n)]
    report.check('loaded altitude - original altitude [m]', np.array(loaded) - np.array(alt), expected=0, atol=1e-6)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Edge cases and chromatic shift
# ----------------------------------------------------------------------------------------------------------------------

def test_odd_resolution_and_zero_cn2_layer(logger, report):
    """
    An odd pupil sampling (31 px) gives a 31 x 31 OPD for an off-axis source, and a layer with zero Cn2 fraction is a layer
    without turbulence (phase identically 0, finite OPD) instead of a crash.
    """
    tel = Telescope(resolution=31, diameter=1.0, samplingTime=1e-3, fov=20, logger=logger)
    atm = Atmosphere(r0=R0, L0=L0, fractionalR0=[1.0, 0.0], altitude=[0.0, 5000.0], windDirection=[0, 90], windSpeed=[5, 10],
                     telescope=tel, zenith=0, logger=logger)
    atm.initializeAtmosphere(randomState=3)
    for _ in range(5):
        atm.update()
    opd = np.array(atm.getOPD(Source('V', 0, coordinates=[5, 30], logger=logger)))
    report.check('OPD shape == (31, 31)', float(opd.shape == (31, 31)), 1.0, atol=0)
    report.check('OPD finite', float(np.isfinite(opd).all()), 1.0, atol=0)
    report.check('max |phase| of the zero-Cn2 layer', float(np.abs(np.array(atm.layer_2.screen.scrn)).max()), 0.0, atol=0)
    report.assert_ok()


def test_random_seed_when_not_given(logger, report):
    """Without randomState two atmospheres differ, and the seed used is stored, so that it reproduces the same screens."""
    tel = Telescope(resolution=16, diameter=1.0, samplingTime=1e-3, logger=logger)
    kw = dict(r0=R0, L0=L0, fractionalR0=[1.0], altitude=[0.0], windDirection=[0], windSpeed=[1], telescope=tel, zenith=0, logger=logger)
    a, b = Atmosphere(**kw), Atmosphere(**kw)
    a.initializeAtmosphere()
    b.initializeAtmosphere()
    c = Atmosphere(**kw)
    c.initializeAtmosphere(randomState=a.randomState)
    sa, sb, sc = (np.array(x.layer_1.screen.scrn) for x in (a, b, c))
    report.check('two unseeded atmospheres differ (1 = yes)', float(not np.array_equal(sa, sb)), 1.0, atol=0)
    report.check('stored seed reproduces the screen: max |diff|', float(np.abs(sa - sc).max()), 0.0, atol=0)
    report.assert_ok()


def test_chromatic_shift_is_a_common_vector(logger, report):
    """
    The chromatic shift is an on-sky vector per layer added to the source direction: an on-axis source shifted by (4, 0)"
    on the 5 km layer samples the same footprint as a source at 4" azimuth 0, and the shift moves every source alike (a
    source at 4" azimuth 90 shifted by (4, 0)" samples the footprint of a source at (4, 4)").
    """
    tel = Telescope(resolution=32, diameter=1.6, samplingTime=1e-3, fov=40, logger=logger)
    atm = Atmosphere(r0=R0, L0=L0, fractionalR0=[0.5, 0.5], altitude=[0.0, 5000.0], windDirection=[0, 0], windSpeed=[1, 1],
                     telescope=tel, zenith=0, logger=logger)
    atm.initializeAtmosphere(randomState=1)
    shift = [[0.0, 0.0], [4.0, 0.0]]
    fp = lambda coords, cs=None: atm.get_pupil_footprint(Source('V', 0, coordinates=coords, chromatic_shift=cs, logger=logger))[1]
    report.check('shifted on-axis == source at (4", 0 deg)', float(np.array_equal(fp([0, 0], shift), fp([4, 0]))), 1.0, atol=0)
    report.check('shifted (4", 90 deg) == source at (4 sqrt2", 45 deg)', float(np.array_equal(fp([4, 90], shift), fp([4 * np.sqrt(2), 45]))), 1.0, atol=0)
    report.assert_ok()

