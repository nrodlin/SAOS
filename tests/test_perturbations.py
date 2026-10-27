"""
T2 validation of the perturbation modules: Vibration, LocalSeeing, spider arms (Telescope.apply_spiders), NCPA and
the exit of a process that created SAOS objects with logger=None.

Expectations are independent of SAOS:
  * Vibration: a tilt of theta [arcsec] on sky moves a diffraction-limited image by theta / plate_scale pixels
    (towards +col for x, +row for y); the camera peak series has the PSD of the file.
  * LocalSeeing: the injected OPD is the file (wrapping around its length), and the file is never modified.
  * Spider: the removed area is n w (R - R_co) (exact strip-annulus integral) / pixel^2 within the pixelisation
    scatter measured on an independent lattice; arms point to the requested angles (counter-clockwise from +x with
    y = +row, the SAOS convention of the DM tests: azimuth 90 deg = +y = rows); the PSF has diffraction spikes perpendicular
    to the arms (independent FFT of an independently built mask).
  * NCPA: load() returns the file; the NCPA is added to the science path and not to the WFS path (LightPath docs).
  * Logging: a script that raises with logger=None must exit (timeout 10 s).
"""
import os
import subprocess
import sys
import textwrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import ndimage

from citools import known_issue
from SAOS.LightPath import LightPath
from SAOS.LocalSeeing import LocalSeeing
from SAOS.NCPA import NCPA
from SAOS.ScienceCam import ScienceCam
from SAOS.ShackHartmann import ShackHartmann
from SAOS.Source import Source
from SAOS.Telescope import Telescope
from SAOS.Vibration import Vibration

pytestmark = pytest.mark.nightly

RAD2ARCSEC = 206265
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def make_camera(logger, res=64, fw=4.0, diameter=1.0, fs=1000.0, co=0.0, pupil=None, tel=None):
    """Telescope + R-band source + ScienceCam with fw pixels per lambda/D and a field of res pixels."""
    if tel is None:
        tel = Telescope(resolution=res, diameter=diameter, centralObstruction=co, samplingTime=1 / fs, pupil=pupil, logger=logger)
    src = Source(optBand='R', magnitude=5, logger=logger)
    ps = src.wavelength / tel.D * RAD2ARCSEC / fw
    cam = ScienceCam(fieldOfView=res * ps, plate_scale=ps, samplingTime=tel.samplingTime, telescope=tel, logger=logger, device='cpu')
    return tel, src, cam, ps


def fourier_upsample(img, up):
    n0, n1 = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    G = np.zeros((n0 * up, n1 * up), dtype=complex)
    o0, o1 = (n0 * up - n0) // 2, (n1 * up - n1) // 2
    G[o0:o0 + n0, o1:o1 + n1] = F
    return np.real(np.fft.ifft2(np.fft.ifftshift(G))) * up ** 2


def peak_position(img, up=16):
    """Sub-pixel peak (row, col) [px]: Fourier upsampling + parabolic refinement (band-limited PSFs)."""
    u = fourier_upsample(np.asarray(img, dtype=float), up)
    r, c = np.unravel_index(np.argmax(u), u.shape)

    def parab(m, z, p):
        den = m - 2 * z + p
        return 0.0 if den == 0 else 0.5 * (m - p) / den
    return (r + parab(u[r - 1, c], u[r, c], u[r + 1, c])) / up, (c + parab(u[r, c - 1], u[r, c], u[r, c + 1])) / up


def write_vibration(path, x, y, Ts):
    with h5py.File(path, 'w') as f:
        f.create_group('x_axis').create_dataset('data', data=np.asarray(x, dtype=float))
        f.create_group('y_axis').create_dataset('data', data=np.asarray(y, dtype=float))
        f['x_axis'].attrs['Ts'] = Ts
        f['y_axis'].attrs['Ts'] = Ts


def write_opd(path, opd, Ts=None, group='opd'):
    with h5py.File(path, 'w') as f:
        f.create_group(group).create_dataset('data', data=np.asarray(opd, dtype=float))
        if Ts is not None:
            f[group].attrs['Ts'] = Ts


def physical_coords(tel):
    """Pupil-plane coordinates [m] of the pixel centres, X along columns and Y along rows (y = +row)."""
    x = (np.arange(tel.resolution) - (tel.resolution - 1) / 2) * tel.D / tel.resolution
    return np.meshgrid(x, x)


# ----------------------------------------------------------------------------------------------------------------------
# Vibration
# ----------------------------------------------------------------------------------------------------------------------

def test_vibration_one_step_tilt_moves_image_by_file_angle(logger, tmp_path, report, tolerances):
    """
    One-step tilt: iteration k injects (x_k, y_k) arcsec. With no atmosphere and no AO the ScienceCam image moves by
    x_k / plate_scale pixels towards +col and y_k / plate_scale towards +row, i.e. the measured shift divided by the
    file angle is 1.00 (the zero-vibration frame is the origin). Estimator: Fourier-upsampled peak.
    """
    tol = tolerances['perturbations']['vibration']
    tel, src, cam, ps = make_camera(logger, res=64)
    xs = np.array([0.0, 0.05, -0.04, 0.0, 0.03, 0.06])
    ys = np.array([0.0, 0.0, 0.05, -0.06, 0.04, -0.03])
    f = str(tmp_path / 'vib.h5')
    write_vibration(f, xs, ys, tel.samplingTime)
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, sci=cam, vibration=Vibration(tel, f, logger=logger))
    peaks = []
    for _ in xs:
        lp.propagate(temporal_tick=True)
        peaks.append(peak_position(lp.sci_frame))
    peaks = np.array(peaks)
    dr, dc = (peaks - peaks[0]).T
    exp_c, exp_r = xs / ps, ys / ps
    report.check('x: shift [px] vs x / plate_scale', dc, exp_c, rtol=tol['shift_rtol'], atol=tol['shift_atol_px'], unit='px')
    report.check('y: shift [px] vs y / plate_scale', dr, exp_r, rtol=tol['shift_rtol'], atol=tol['shift_atol_px'], unit='px')
    m = np.abs(exp_c) > 0.5
    report.metric('x: measured / file (slope, expect 1.00)', float(np.sum(dc[m] * exp_c[m]) / np.sum(exp_c[m] ** 2)))
    m = np.abs(exp_r) > 0.5
    report.metric('y: measured / file (slope, expect 1.00)', float(np.sum(dr[m] * exp_r[m]) / np.sum(exp_r[m] ** 2)))
    report.assert_ok()


@pytest.mark.parametrize('lx, ly', [
    pytest.param(5, 8, id='x_shorter'),
    pytest.param(8, 5, id='y_shorter', marks=known_issue('VIB-LEN', 'length check tests x twice: when y is shorter than x, getCurrentVibrations '
                                                         'indexes y out of range instead of wrapping', 'SAOS/Vibration.py:108')),
    pytest.param(6, 6, id='equal'),
])
def test_vibration_wraps_with_different_length_series(lx, ly, logger, tmp_path, report, tolerances):
    """
    x and y series of different length: the sequence wraps around the shorter one, so that iteration k gives
    (x[k mod L], y[k mod L]), L = min(Lx, Ly), and never raises. The fitted tilts of the OPD (least squares on the
    physical coordinates) follow the file; the overall scale (tested above) is calibrated at iteration 0.
    """
    tol = tolerances['perturbations']['vibration']['index_rtol']
    tel = Telescope(resolution=32, diameter=1.0, samplingTime=1e-3, logger=logger)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0.5, 1.5, lx), rng.uniform(0.5, 1.5, ly) * rng.choice([-1, 1], ly)
    f = str(tmp_path / 'vib.h5')
    write_vibration(f, x, y, tel.samplingTime)
    vib = Vibration(tel, f, logger=logger)
    X, Y = physical_coords(tel)
    pup = tel.pupil.astype(bool)
    A = np.c_[X[pup], Y[pup]]
    L = min(lx, ly)

    def fit(it):
        return np.linalg.lstsq(A, vib.getCurrentVibrations(it)[pup], rcond=None)[0] * RAD2ARCSEC   # arcsec

    c0 = fit(0)
    gx, gy = c0[0] / x[0], c0[1] / y[0]
    err = []
    for it in range(0, 2 * max(lx, ly) + 1):
        c = fit(it)
        err.append([c[0] / gx - x[it % L], c[1] / gy - y[it % L]])
    report.check('max |fitted - file[k mod L]| / max |file|', np.max(np.abs(err)) / max(np.abs(x).max(), np.abs(y).max()), hi=tol)
    report.metric('scale fitted / file, x and y (iteration 0)', [gx, gy])
    report.assert_ok()


def test_vibration_psd_equals_file_psd(logger, tmp_path, report, tolerances):
    """
    The image-motion series measured on the ScienceCam equals the file series (up to the exact scale 1 / plate_scale), so
    its PSD equals the file PSD: band-integrated power ratio 1.00 in 8 bands, and rms ratio 1.00. Series: 3 tones +
    AR(1) noise, rms 0.3 px, 1024 steps at 1 kHz.
    """
    tol = tolerances['perturbations']['vibration']
    n, fs = 1024, 1000.0
    rng = np.random.default_rng(11)
    t = np.arange(n) / fs
    noise = np.zeros(n)
    for k in range(1, n):
        noise[k] = 0.9 * noise[k - 1] + rng.standard_normal()
    tel, src, cam, ps = make_camera(logger, res=32, fs=fs)

    def series(rms_px):
        s = (np.sin(2 * np.pi * 37 * t + 1) + 0.6 * np.sin(2 * np.pi * 112 * t) + 0.4 * np.sin(2 * np.pi * 251 * t + 2) + 0.5 * noise / noise.std())
        return s / s.std() * rms_px * ps                         # arcsec

    x, y = series(0.3), np.roll(series(0.2), 100)
    f = str(tmp_path / 'vib.h5')
    write_vibration(f, x, y, 1 / fs)
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, sci=cam, vibration=Vibration(tel, f, logger=logger))
    pk = []
    for _ in range(n):
        lp.propagate(temporal_tick=True)
        pk.append(peak_position(lp.sci_frame, up=8))
    pk = np.array(pk)
    meas = {'x': pk[:, 1] - pk[:, 1].mean(), 'y': pk[:, 0] - pk[:, 0].mean()}
    ref = {'x': (x - x.mean()) / ps, 'y': (y - y.mean()) / ps}
    freq = np.fft.rfftfreq(n, 1 / fs)
    edges = np.linspace(0, fs / 2, 9)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, ax_name in enumerate(('x', 'y')):
        Pm = np.abs(np.fft.rfft(meas[ax_name])) ** 2
        Pr = np.abs(np.fft.rfft(ref[ax_name])) ** 2
        bm = np.array([Pm[(freq >= a) & (freq < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        br = np.array([Pr[(freq >= a) & (freq < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        report.check(f'{ax_name}: band power measured / file', bm / br, 1.0, rtol=tol['psd_rtol'])
        report.check(f'{ax_name}: rms measured / file', meas[ax_name].std() / ref[ax_name].std(), 1.0, rtol=tol['psd_rtol'])
        ax[i].semilogy(freq, Pr, label='file')
        ax[i].semilogy(freq, Pm, '--', label='camera')
        ax[i].set_title(f'{ax_name} periodogram [px^2]', fontsize=8)
        ax[i].legend(frameon=False, fontsize=7)
    report.figure(fig, 'vibration_psd')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Local seeing
# ----------------------------------------------------------------------------------------------------------------------

def make_localseeing(logger, tmp_path, res=32, T=5, seed=5, file_res=None):
    tel = Telescope(resolution=res, diameter=1.0, samplingTime=1e-3, logger=logger)
    fr = file_res or res
    opd = np.random.default_rng(seed).standard_normal((fr, fr, T)) * 1e-7
    f = str(tmp_path / 'ls.h5')
    write_opd(f, opd, tel.samplingTime)
    return tel, opd, f


def test_localseeing_opd_equals_file_with_wraparound(logger, tmp_path, report, tolerances):
    """
    The OPD injected at iteration k is the file frame k mod T (also beyond T: wrap-around), both as the LightPath
    local-seeing term and in the science OPD (file x pupil; no atmosphere, DM or NCPA): bit equality.
    """
    tel, opd, f = make_localseeing(logger, tmp_path)
    T = opd.shape[2]
    ls = LocalSeeing(tel, f, logger=logger)
    lp = LightPath(logger)
    lp.initialize_path(src=Source('R', 5, logger=logger), tel=tel, localSeeing=ls)
    e_ls, e_sci = [], []
    for k in range(2 * T + 3):
        lp.propagate(temporal_tick=True)
        e_ls.append(np.max(np.abs(lp.localSeeing_opd - opd[:, :, k % T])))
        e_sci.append(np.max(np.abs(lp.sci_opd - opd[:, :, k % T] * tel.pupil)))
    report.check('max |localSeeing_opd - file[k mod T]| [m]', max(e_ls), atol=tolerances['perturbations']['localseeing']['exact'], expected=0)
    report.check('max |sci_opd - file[k mod T] pupil| [m]', max(e_sci), atol=tolerances['perturbations']['localseeing']['exact'], expected=0)
    report.assert_ok()


def test_localseeing_resolution_mismatch_is_bilinear_resampling(logger, tmp_path, report, tolerances):
    """
    A file at twice the pupil resolution is resampled bilinearly: halving a grid by linear interpolation at the pixel
    centres is the 2x2 block mean, then multiplied by the pupil.
    """
    tel, opd, f = make_localseeing(logger, tmp_path, res=32, file_res=64, T=3)
    ls = LocalSeeing(tel, f, logger=logger)
    ref = opd.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3)) * tel.pupil[:, :, None]
    err = np.max(np.abs(ls.localSeeing_input - ref)) / np.max(np.abs(ref))
    report.check('max |resampled - 2x2 mean x pupil| / max', err, hi=tolerances['perturbations']['localseeing']['resize_rtol'])
    report.assert_ok()


def test_localseeing_input_not_modified_by_interaction_matrix_propagate(logger, tmp_path, report):
    """
    The user's sequence is data: propagating a normal step and then an interaction-matrix step (which suppresses the
    disturbances of the path) must leave the LocalSeeing input unchanged, and the next normal step must inject the file.
    """
    tel, opd, f = make_localseeing(logger, tmp_path)
    ls = LocalSeeing(tel, f, logger=logger)
    before = ls.localSeeing_input.copy()
    lp = LightPath(logger)
    lp.initialize_path(src=Source('R', 5, logger=logger), tel=tel, localSeeing=ls)
    lp.propagate(temporal_tick=False)
    lp.propagate(temporal_tick=False, interaction_matrix=True)
    report.check('max |input after IM propagate - input before| [m]', np.max(np.abs(ls.localSeeing_input - before)), expected=0, atol=0)
    report.check('max |input - file| [m]', np.max(np.abs(ls.localSeeing_input - opd)), expected=0, atol=0)
    lp.propagate(temporal_tick=False)
    report.check('next normal step injects file frame 0: max error [m]', np.max(np.abs(lp.localSeeing_opd - opd[:, :, 0])), expected=0, atol=0)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Spider
# ----------------------------------------------------------------------------------------------------------------------

def ref_masks(N, D, co, w, angles, shift_px=(0.0, 0.0)):
    """
    Independent geometry: annulus R_co <= r <= R and half-line strips of width w from the centre at the requested
    angles (counter-clockwise from +x, y = +row), sampled at pixel centres (i - (N-1)/2) D/N displaced by shift_px.
    Returns (annulus, removed).
    """
    px = D / N
    x = (np.arange(N) - (N - 1) / 2) * px
    X, Y = np.meshgrid(x + shift_px[1] * px, x + shift_px[0] * px)
    r = np.hypot(X, Y)
    ann = (r <= D / 2) & (r >= co * D / 2)
    rem = np.zeros_like(ann)
    for th in np.deg2rad(angles):
        along = X * np.cos(th) + Y * np.sin(th)
        perp = -X * np.sin(th) + Y * np.cos(th)
        rem |= (along > 0) & (np.abs(perp) < w / 2)
    return ann, ann & rem


def strip_annulus_area(R, Rco, w):
    """Exact area of the half-strip |t| < w/2, s > 0 inside the annulus Rco <= r <= R."""
    h = w / 2
    f = lambda r: h * np.sqrt(r ** 2 - h ** 2) + r ** 2 * np.arcsin(h / r)
    return f(R) - f(Rco)


def arm_angles(removed):
    """Direction [deg, counter-clockwise from +x with y = +row] of every connected removed region (PCA axis, pointing away from the centre)."""
    N = removed.shape[0]
    lab, n = ndimage.label(removed)
    out = []
    for i in range(1, n + 1):
        rr, cc = np.nonzero(lab == i)
        if len(rr) < 20:
            continue
        pts = np.c_[cc - (N - 1) / 2, rr - (N - 1) / 2]
        mu = pts.mean(axis=0)
        v = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
        if v @ mu < 0:
            v = -v
        out.append(np.rad2deg(np.arctan2(v[1], v[0])) % 360)
    return np.array(sorted(out))


def ang_diff(a, b):
    return (a - b + 180) % 360 - 180


SPIDER_N, SPIDER_D, SPIDER_CO, SPIDER_W = 256, 8.0, 0.3, 0.8      # arm width 25.6 px: lattice quantisation ~1 %
SPIDER_CASES = [
    pytest.param([0, 90, 180, 270], id='cardinal'),
    pytest.param([45, 135, 225, 315], id='diagonal'),
    pytest.param([20, 140, 260], id='non_cardinal'),
    pytest.param([17, 107, 197, 287], id='rotated'),
    pytest.param([357, 87, 177, 267], id='near_cardinal'),
]


def spider_telescope(logger, angles):
    tel = Telescope(resolution=SPIDER_N, diameter=SPIDER_D, samplingTime=1e-3, centralObstruction=SPIDER_CO, logger=logger)
    n0 = int(tel.pupil.sum())
    tel.apply_spiders(angles, SPIDER_W)
    return tel, n0


def spider_params(marks):
    return [pytest.param(c.values[0], id=c.id, marks=marks.get(c.id, [])) for c in SPIDER_CASES]


@pytest.mark.parametrize('angles', spider_params({}))
def test_spider_removed_area(angles, logger, report, tolerances):
    """
    Removed area = n w (R - R_co) / pixel^2 (exact strip-annulus integral n [A(R) - A(R_co)], pixel = D/N) within the
    pixelisation of the lattice: the tolerance is the larger of 3.5 sigma of the count of the independent mask over 64
    random sub-pixel offsets and the deviation of its count at zero offset (arms aligned with the lattice are
    quantised deterministically).
    """
    tol = tolerances['perturbations']['spider']
    tel, n0 = spider_telescope(logger, angles)
    removed = n0 - int(tel.pupil.sum())
    px = SPIDER_D / SPIDER_N
    n = len(angles)
    expected = n * strip_annulus_area(SPIDER_D / 2, SPIDER_CO * SPIDER_D / 2, SPIDER_W) / px ** 2
    rng = np.random.default_rng(1)
    counts = np.array([ref_masks(SPIDER_N, SPIDER_D, SPIDER_CO, SPIDER_W, angles, rng.uniform(-0.5, 0.5, 2))[1].sum() for _ in range(tol['n_lattice'])])
    sigma = counts.std()
    report.metric('lattice count of the independent mask: mean', counts.mean())
    report.metric('lattice sigma [px]', sigma)
    zero = ref_masks(SPIDER_N, SPIDER_D, SPIDER_CO, SPIDER_W, angles)[1].sum()
    report.metric('independent lattice count at zero offset (SAOS centring)', zero)
    report.metric('simple formula n w (R - R_co) / px^2 [px]', n * SPIDER_W * (SPIDER_D / 2 * (1 - SPIDER_CO)) / px ** 2)
    # Arms aligned with the lattice (cardinal, diagonal) are quantised deterministically for ANY implementation centred on the
    # array centre: the independent count at zero offset is then part of the pixelisation, so it enters the tolerance.
    report.check('removed pixels (SAOS) vs exact area', removed, expected, atol=max(tol['sigma_mult'] * sigma, abs(zero - expected)), unit='px')
    report.metric('removed / expected', removed / expected)
    report.assert_ok()


@pytest.mark.parametrize('angles', spider_params({}))
def test_spider_arm_angles(angles, logger, report, tolerances):
    """
    Each arm points to the requested angle (counter-clockwise from +x, y = +row): the PCA axis of every removed region
    (pointing away from the centre) is compared with the requested angle; the estimator is first checked on the
    independent mask.
    """
    tol = tolerances['perturbations']['spider']['angle_deg']
    tel, n0 = spider_telescope(logger, angles)
    ann, ref_removed = ref_masks(SPIDER_N, SPIDER_D, SPIDER_CO, SPIDER_W, angles)
    got = arm_angles(ann & ~tel.pupil.astype(bool))
    ref = arm_angles(ref_removed)
    want = np.sort(np.asarray(angles) % 360)
    report.check('number of arms found', len(got), len(angles), atol=0)
    report.check('estimator on the independent mask: max |angle - requested| [deg]', np.max(np.abs(ang_diff(ref, want))) if len(ref) == len(want) else 999, expected=0, atol=tol, unit='deg')
    if len(got) == len(want):
        report.check('SAOS arms: max |angle - requested| [deg]', np.max(np.abs(ang_diff(got, want))), expected=0, atol=tol, unit='deg')
    report.metric('SAOS arm angles [deg]', got)
    report.assert_ok()


@pytest.mark.parametrize('angle', [
    pytest.param(30, id='30'),
    pytest.param(0, id='0'),
    pytest.param(357, id='357'),
])
def test_spider_single_arm_is_one_half_strip(angle, logger, report):
    """
    One arm removes one connected half-strip beyond the central obstruction (the obstruction separates the two sides of
    the line): the removed region has one connected component, of area n w (R - R_co) within the arm-area tolerance of
    test_spider_removed_area. An arm whose SAOS direction lies just below a multiple of 90 deg (357 deg) is split by the quadrant clipping
    into two pieces on opposite sides of the centre (total area unchanged, geometry wrong).
    """
    tel, n0 = spider_telescope(logger, [angle])
    removed = (ref_masks(SPIDER_N, SPIDER_D, SPIDER_CO, 0.0, [0])[0]) & ~tel.pupil.astype(bool)
    lab, n = ndimage.label(removed)
    report.check('connected removed regions', n, 1, atol=0)
    report.metric('areas of the regions [px]', np.bincount(lab.ravel())[1:])
    report.assert_ok()


def azimuth_profile(psf, r_lo, r_hi, nbins=360):
    """Mean intensity in azimuth bins (deg, counter-clockwise from +col with y = +row) of the annulus r_lo..r_hi [px] around the peak, normalised to unit mean."""
    c = np.unravel_index(np.argmax(psf), psf.shape)
    rr, cc = np.indices(psf.shape)
    y, x = rr - c[0], cc - c[1]
    r = np.hypot(x, y)
    m = (r >= r_lo) & (r <= r_hi)
    phi = np.rad2deg(np.arctan2(y[m], x[m])) % 360
    b = (phi / 360 * nbins).astype(int) % nbins
    prof = np.bincount(b, weights=psf[m], minlength=nbins) / np.maximum(np.bincount(b, minlength=nbins), 1)
    prof = ndimage.gaussian_filter1d(prof, 3, mode='wrap')            # 3 deg
    return (np.arange(nbins) + 0.5) * 360 / nbins, prof / prof.mean()


def profile_distance(pa, pb):
    """rms difference of two normalised azimuthal profiles."""
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def fft_psf(pupil, fw):
    """Independent PSF: |FFT|^2 of the pupil zero-padded to fw pixels per lambda/D, cropped to the pupil size."""
    N = pupil.shape[0]
    pad = np.zeros((N * fw, N * fw))
    pad[:N, :N] = pupil
    c = N * fw // 2
    return np.fft.fftshift(np.abs(np.fft.fft2(pad)) ** 2)[c - N // 2:c + N // 2, c - N // 2:c + N // 2]


def test_spider_psf_spikes_perpendicular_to_arms(logger, report, tolerances):
    """
    Diffraction spikes are perpendicular to the arms (a line is perpendicular on both sides). Two perpendicular arms at 20
    and 110 deg (counter-clockwise from +x, y = +row) give four spikes at 20, 110, 200 and 290 deg. The azimuthal profile
    (annulus 2.5-7 lambda/D, normalised) of the SAOS ScienceCam PSF is compared with the one of the independent FFT of
    an independently built mask: the rms difference must be within the pixelisation scatter of the same comparison
    between lattices of the independent mask displaced by random sub-pixel offsets (mean + 3.5 sigma over 32 offsets). The
    spike azimuths are recorded (estimator biased by up to 3 deg by the broad lobes: not a criterion).
    """
    tol = tolerances['perturbations']['spider']
    N, D, co, w, fw = 128, 4.0, 0.1, 0.1, 4
    angles = [20, 110]
    tel = Telescope(resolution=N, diameter=D, samplingTime=1e-3, centralObstruction=co, logger=logger)
    tel.apply_spiders(angles, w)
    src = Source('R', 5, logger=logger)
    ps = src.wavelength / D * RAD2ARCSEC / fw
    cam = ScienceCam(fieldOfView=N * ps, plate_scale=ps, samplingTime=1e-3, telescope=tel, logger=logger, device='cpu')
    psf = cam.compute_psf(np.zeros((N, N)), fw).cpu().numpy()
    ann, rem = ref_masks(N, D, co, w, angles)
    ref_psf = fft_psf((ann & ~rem).astype(float), fw)
    r_lo, r_hi = 2.5 * fw, 7 * fw
    az, p_saos = azimuth_profile(psf, r_lo, r_hi)
    _, p_ref = azimuth_profile(ref_psf, r_lo, r_hi)
    rng = np.random.default_rng(4)
    floor = []
    for _ in range(32):
        a, r = ref_masks(N, D, co, w, angles, rng.uniform(-0.5, 0.5, 2))
        floor.append(profile_distance(azimuth_profile(fft_psf((a & ~r).astype(float), fw), r_lo, r_hi)[1], p_ref))
    floor = np.array(floor)
    _, p_mirror = azimuth_profile(fft_psf((lambda a, r: (a & ~r).astype(float))(*ref_masks(N, D, co, w, [-t for t in angles])), fw), r_lo, r_hi)
    report.metric('pixelisation floor: mean rms difference between lattices', floor.mean())
    report.metric('mirrored arms (independent): rms difference, for scale', profile_distance(p_mirror, p_ref))
    report.check('SAOS vs independent: rms difference of azimuthal profiles', profile_distance(p_saos, p_ref), hi=floor.mean() + tol['sigma_mult'] * floor.std())
    report.metric('independent spike azimuths (4 strongest maxima) [deg]', np.sort(az[np.argsort(p_ref)[::-1][:1]]))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(np.log10(psf + 1e-9), origin='upper'); ax[0].set_title('SAOS PSF (log)', fontsize=8)
    ax[1].imshow(np.log10(ref_psf + 1e-9), origin='upper'); ax[1].set_title('independent FFT (log)', fontsize=8)
    ax[2].plot(az, p_saos, label='SAOS'); ax[2].plot(az, p_ref, '--', label='independent'); ax[2].plot(az, p_mirror, ':', label='mirrored arms')
    ax[2].legend(frameon=False, fontsize=7); ax[2].set_title('azimuthal profile (normalised)', fontsize=8)
    report.figure(fig, 'spider_spikes')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# NCPA
# ----------------------------------------------------------------------------------------------------------------------

def test_ncpa_load_same_shape_round_trip(logger, tmp_path, report, tolerances):
    """A file with the pupil shape is returned by getPhase after load (float32 buffer: relative error <= 1e-6 of the maximum)."""
    tel = Telescope(resolution=32, diameter=1.0, samplingTime=1e-3, logger=logger)
    opd = np.random.default_rng(2).standard_normal((32, 32)) * 1e-7
    f = str(tmp_path / 'ncpa.h5')
    write_opd(f, opd)
    ncpa = NCPA(tel, logger=logger)
    assert ncpa.load(f) is True
    report.check('max |getPhase - file| / max |file|', np.max(np.abs(ncpa.getPhase() - opd)) / np.max(np.abs(opd)), hi=tolerances['perturbations']['ncpa']['float32_rtol'])
    report.assert_ok()


def test_ncpa_load_resized_file_is_bilinear_resampling(logger, tmp_path, report, tolerances):
    """A file at twice the pupil resolution is resampled bilinearly (2x2 block mean for a factor 2)."""
    tel = Telescope(resolution=32, diameter=1.0, samplingTime=1e-3, logger=logger)
    opd = np.random.default_rng(2).standard_normal((64, 64)) * 1e-7
    f = str(tmp_path / 'ncpa.h5')
    write_opd(f, opd)
    ncpa = NCPA(tel, logger=logger)
    ncpa.load(f)
    ref = opd.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    report.check('max |getPhase - 2x2 mean| / max', np.max(np.abs(ncpa.getPhase() - ref)) / np.max(np.abs(ref)), hi=tolerances['perturbations']['ncpa']['resize_rtol'])
    report.assert_ok()


def test_ncpa_enters_science_path_not_wfs_path(logger, report, tolerances):
    """
    LightPath docs: wfs_opd = atm + vibration + local seeing + DM, sci_opd = wfs_opd + NCPA. A tilt NCPA (set through
    setOPD, same shape) therefore (i) moves the science image by theta / plate_scale pixels (+col for a ramp along x),
    (ii) leaves the WFS measurement unchanged (bit equal) and (iii) satisfies sci_opd = (wfs_opd + ncpa) pupil exactly.
    """
    tol = tolerances['perturbations']['ncpa']
    N = 32
    tel, src, cam, ps = make_camera(logger, res=N)
    wfs = ShackHartmann(nSubap=4, telescope=tel, src=src, lightRatio=1.0, plate_scale=src.wavelength / (tel.D / 4) * RAD2ARCSEC / 2.0,
                        fieldOfView=9 * src.wavelength / (tel.D / 4) * RAD2ARCSEC / 2.0, guardPx=1, use_brightest=81, logger=logger, device='cpu')
    theta = 0.05 / RAD2ARCSEC                                    # 0.05 arcsec
    X, _ = physical_coords(tel)
    ncpa = NCPA(tel, logger=logger)
    ncpa.setOPD(theta * X)
    base = LightPath(logger); base.initialize_path(src=src, tel=tel, wfs=wfs, sci=cam)
    withn = LightPath(logger); withn.initialize_path(src=src, tel=tel, wfs=wfs, sci=cam, ncpa=ncpa)
    base.propagate(temporal_tick=True); withn.propagate(temporal_tick=True)
    dr, dc = np.array(peak_position(withn.sci_frame)) - np.array(peak_position(base.sci_frame))
    report.check('science image shift along x / (theta / plate_scale)', dc / (0.05 / ps), 1.0, rtol=tol['shift_rtol'])
    report.check('science image shift along y [px]', dr, 0.0, atol=0.01, unit='px')
    report.check('WFS slopes with NCPA == without (max |diff| [px])', np.max(np.abs(withn.slopes_1D - base.slopes_1D)), expected=0, atol=tol['exact'])
    report.check('sci_opd - (wfs_opd + ncpa) pupil [m]', np.max(np.abs(withn.sci_opd - (withn.wfs_opd + ncpa.getPhase()) * tel.pupil)), expected=0, atol=tol['exact'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Logging at exit
# ----------------------------------------------------------------------------------------------------------------------

LOG_SCRIPTS = {
    'telescope': "from SAOS.Telescope import Telescope\n"
                 "t = Telescope(resolution=16, diameter=1.0, logger=None)\n",
    'lightpath_source': "from SAOS.Telescope import Telescope\nfrom SAOS.Source import Source\nfrom SAOS.LightPath import LightPath\n"
                        "t = Telescope(resolution=16, diameter=1.0, logger=None)\ns = Source('R', 5, logger=None)\nlp = LightPath(logger=None)\n",
    'logginghelper': "from SAOS.LoggingHelper import LoggingHelper\nlh = LoggingHelper()\n",
}


@pytest.mark.parametrize('name', ['telescope', 'lightpath_source', 'logginghelper'])
def test_exit_after_exception_with_logger_none(name, report, tolerances, tmp_path):
    """
    A script that creates SAOS objects with logger=None and then raises an exception exits (non-zero status, traceback on
    stderr) within 10 s; a hang is killed by the timeout and reported as failure.
    """
    timeout = tolerances['perturbations']['logging']['exit_timeout_s']
    script = tmp_path / f'{name}.py'
    script.write_text(LOG_SCRIPTS[name] + "raise RuntimeError('intentional failure')\n")
    env = dict(os.environ, PYTHONPATH=REPO_DIR + os.pathsep + os.environ.get('PYTHONPATH', ''))
    t0 = os.times()[4]
    try:
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(tmp_path))
        hung, rc, err = False, p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        hung, rc, err = True, None, (e.stderr or b'').decode() if isinstance(e.stderr, bytes) else (e.stderr or '')
    elapsed = os.times()[4] - t0
    report.metric('wall time until exit or kill [s]', elapsed)
    report.note(f'{name}: hung={hung}, returncode={rc}')
    report.check('process exited by itself (1 = yes)', 0.0 if hung else 1.0, expected=1.0, atol=0)
    if not hung:
        report.check('exit status (the exception propagates)', rc, expected=1, atol=0)
        report.check('traceback printed', 1.0 if 'RuntimeError: intentional failure' in err else 0.0, expected=1.0, atol=0)
    report.assert_ok()


def test_logger_none_prints_each_message_once(report, tolerances, tmp_path):
    """
    Several SAOS objects created with logger=None share one logging queue: a warning logged after creating Telescope, Source and
    LightPath is printed exactly once, the process exits by itself with status 0 and no exception is reported from __del__ at exit.
    """
    timeout = tolerances['perturbations']['logging']['exit_timeout_s']
    script = tmp_path / 'once.py'
    script.write_text(LOG_SCRIPTS['lightpath_source'] + "import logging\nlogging.getLogger().warning('marker-message')\n")
    env = dict(os.environ, PYTHONPATH=REPO_DIR + os.pathsep + os.environ.get('PYTHONPATH', ''))
    try:
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(tmp_path))
        hung, rc, out = False, p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        hung, rc, out = True, None, ''
    report.check('process exited by itself (1 = yes)', 0.0 if hung else 1.0, expected=1.0, atol=0)
    report.check('exit status', rc if rc is not None else -1, expected=0, atol=0)
    report.check('times the warning is printed', out.count('marker-message'), expected=1, atol=0)
    report.check('exceptions ignored in __del__ at exit', out.count('Exception ignored in'), expected=0, atol=0)
    report.assert_ok()


@pytest.mark.parametrize('kw', [dict(altitude=90e3), dict(Na_profile=np.array([[90e3], [1.0]])), dict(FWHM_spot_up=1.0),
                                dict(laser_coordinates=[1.0, 0.0])], ids=['altitude', 'Na_profile', 'FWHM_spot_up', 'laser_coordinates'])
def test_lgs_source_not_implemented(kw, logger):
    """Laser guide stars are not implemented: any LGS parameter of Source raises NotImplementedError (never a silent NGS)."""
    with pytest.raises(NotImplementedError, match='Laser guide stars'):
        Source('Na', 8, logger=logger, **kw)

