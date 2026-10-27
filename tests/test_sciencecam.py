"""
T1 validation of the science camera (ScienceCam) and of the long-exposure accumulation done by LightPath.

ScienceCam.compute_psf / get_frame are called directly with synthetic phases (no atmosphere, no DM). The long-exposure
tests use a LightPath whose "atmosphere" is a stub returning a prescribed OPD sequence, so that the accumulation is
tested in isolation.

Pixel conventions: frames are [row, col]; a phase ramp increasing along columns moves the image towards +col.
Angular units: lambda/D of the telescope; `fw` = pixels per lambda/D at the camera plate scale.
"""
import types

import numpy as np
import pytest
from scipy.special import j1

import theory
from citools import known_issue, fwhm_1d, fwhm_2d
from SAOS.LightPath import LightPath
from SAOS.ScienceCam import ScienceCam
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

RAD2ARCSEC = 206265


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def make_camera(logger, res=64, n_pix=64, fw=4.0, co=0.0, diameter=1.0, fs=1000.0, band='R', **cam_kwargs):
    """Telescope + Source + ScienceCam with `fw` pixels per lambda/D and a field of n_pix pixels."""
    tel = Telescope(resolution=res, diameter=diameter, centralObstruction=co, samplingTime=1 / fs, logger=logger)
    src = Source(optBand=band, magnitude=5, logger=logger)
    ps = src.wavelength / diameter * RAD2ARCSEC / fw
    cam = ScienceCam(fieldOfView=n_pix * ps, plate_scale=ps, samplingTime=1 / fs, telescope=tel, logger=logger,
                     device='cpu', **cam_kwargs)
    assert cam.nPix == n_pix
    return tel, src, cam


def pupil_coords(tel):
    """Pupil-plane coordinates [m] of the telescope grid (pixel centres), X along columns, Y along rows."""
    x = (np.arange(tel.resolution) - (tel.resolution - 1) / 2) * tel.D / tel.resolution
    return np.meshgrid(x, x)


def fourier_upsample(img, up):
    n0, n1 = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    G = np.zeros((n0 * up, n1 * up), dtype=complex)
    o0, o1 = (n0 * up - n0) // 2, (n1 * up - n1) // 2
    G[o0:o0 + n0, o1:o1 + n1] = F
    return np.real(np.fft.ifft2(np.fft.ifftshift(G))) * up ** 2


def peak_position(img, up=16):
    """Sub-pixel peak position (row, col) [px]: Fourier upsampling + parabolic refinement (band-limited PSFs)."""
    u = fourier_upsample(np.asarray(img, dtype=float), up)
    r, c = np.unravel_index(np.argmax(u), u.shape)
    def parab(m, z, p):
        den = m - 2 * z + p
        return 0.0 if den == 0 else 0.5 * (m - p) / den
    dr = parab(u[r - 1, c], u[r, c], u[r + 1, c])
    dc = parab(u[r, c - 1], u[r, c], u[r, c + 1])
    return (r + dr) / up, (c + dc) / up         # upsampled sample k sits at original pixel k / up


def radial_profile(img, centre, n_bins, step=0.25):
    """Azimuthal mean of `img` around `centre` (row, col) in rings of width `step` px; returns (r, profile)."""
    rr, cc = np.indices(img.shape)
    d = np.hypot(rr - centre[0], cc - centre[1])
    edges = np.arange(n_bins + 1) * step
    prof = np.array([img[(d >= a) & (d < b)].mean() if np.any((d >= a) & (d < b)) else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])
    return 0.5 * (edges[:-1] + edges[1:]), prof


def high_order_phase(rng, res, f_cut_cyc_per_d, rms_rad, pupil):
    """Kolmogorov-like phase (PSD f^-11/3) high-passed above f_cut [cycles/D], scaled to `rms_rad` over the pupil."""
    f = np.fft.fftfreq(res) * res
    F = np.hypot(*np.meshgrid(f, f))
    amp = np.zeros_like(F)
    sel = F > f_cut_cyc_per_d
    amp[sel] = F[sel] ** (-11 / 6)
    ph = np.real(np.fft.ifft2(amp * (rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res)))))
    m = pupil > 0
    ph -= ph[m].mean()
    return ph * rms_rad / ph[m].std()


def exact_frame(pupil, phase, fw, n_pix):
    """
    Independent noise-free frame: |FFT|^2 of the pupil zero-padded to fw px per lambda/D (theory.psf_reference), cropped
    to n_pix px around the optical axis (pixel n_pix/2) and normalised to unit energy inside the field, which is the
    energy convention of a camera frame. The pupil is only geometry (mask); no SAOS output enters the result.
    """
    ref = theory.psf_reference(np.asarray(pupil, dtype=float), phase, oversampling=fw)
    c = ref.shape[0] // 2
    ref = ref[c - n_pix // 2:c - n_pix // 2 + n_pix, c - n_pix // 2:c - n_pix // 2 + n_pix]
    return ref / ref.sum()


class StubAtmosphere:
    """Stands for Atmosphere in a LightPath: getOPD returns the next OPD of a prescribed sequence."""

    def __init__(self, opds):
        self.opds = list(opds)
        self.k = 0

    def getOPD(self, src):
        opd = self.opds[self.k % len(self.opds)]
        self.k += 1
        return opd.copy()


# ----------------------------------------------------------------------------------------------------------------------
# Diffraction-limited PSF
# ----------------------------------------------------------------------------------------------------------------------

def test_airy_psf_unobstructed(logger, tolerances, report):
    """
    Flat phase, unobstructed pupil, camera field sampled exactly like the pupil (n_pix = res, no resampling):
    the frame is the Airy pattern [2 J1(x)/x]^2. Criteria: energy normalised to 1, peak on the optical axis,
    FWHM = 1.029 lambda/D and first dark ring at 1.220 lambda/D (pixelated pupil of 64 px), radial profile vs Airy.
    """
    tol = tolerances['sciencecam']['psf']
    fw = 4.0
    tel, src, cam = make_camera(logger, res=64, n_pix=64, fw=fw)
    frame = cam.get_frame(src, np.zeros((64, 64)))
    report.check('energy', frame.sum(), 1.0, rtol=1e-12)
    pk = peak_position(frame)
    report.check('peak row offset from n_pix/2 [px]', pk[0] - 32, 0.0, atol=0.02)
    report.check('peak col offset from n_pix/2 [px]', pk[1] - 32, 0.0, atol=0.02)
    fwhm = fwhm_2d(frame) / fw
    report.check('FWHM [lambda/D]', fwhm, theory.airy_fwhm(1, 1), rtol=tol['fwhm_rtol'], trend=True)
    up = 8
    u = fourier_upsample(frame, up)
    r, prof = radial_profile(u, np.unravel_index(np.argmax(u), u.shape), n_bins=int(2.2 * fw * up), step=1.0)
    r = r / up / fw
    k0 = np.argmin(np.where((r > 0.9) & (r < 1.6), prof, np.inf))
    report.check('first dark ring [lambda/D]', r[k0], theory.airy_first_zero(1, 1), rtol=tol['first_zero_rtol'])
    airy =(2 * j1(np.pi * np.maximum(r, 1e-9)) / (np.pi * np.maximum(r, 1e-9))) ** 2
    prof_n = prof / prof[0]
    sel = r < 2.0
    report.check('max |profile - Airy| (r < 2 lambda/D, peak = 1)', np.max(np.abs(prof_n[sel] - airy[sel])), 0.0,
                 atol=tol['airy_profile_atol'])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogy(r, prof_n, '-', label='SAOS ScienceCam (azimuthal mean)')
    ax[0].semilogy(r, airy, 'k--', lw=1, label='Airy [2 J1(x)/x]^2')
    ax[0].axvline(theory.airy_first_zero(1, 1), color='0.6', lw=0.8)
    ax[0].set_xlabel('r [lambda/D]'); ax[0].set_ylabel('I / I(0)'); ax[0].set_ylim(1e-4, 1.2); ax[0].legend(fontsize=8)
    ax[1].imshow(np.log10(np.maximum(frame / frame.max(), 1e-5)), cmap='magma'); ax[1].set_title('log10 frame')
    fig.suptitle(f'Diffraction-limited PSF, unobstructed: FWHM {fwhm:.4f} lambda/D (Airy 1.029)')
    report.figure(fig, 'airy_profile')
    report.assert_ok()


def test_psf_obstructed_vs_reference(logger, tolerances, report):
    """
    Flat phase, central obstruction 0.3: the frame must equal the |FFT|^2 of the same pupil computed independently
    (theory.psf_reference, zero padding to `fw` px per lambda/D). Criteria: FWHM ratio and max |difference| / peak.
    """
    tol = tolerances['sciencecam']['psf']
    fw, res = 4.0, 64
    tel, src, cam = make_camera(logger, res=res, n_pix=res, fw=fw, co=0.3)
    frame = cam.get_frame(src, np.zeros((res, res)))
    ref = theory.psf_reference(tel.pupil.astype(float), oversampling=fw)
    c = ref.shape[0] // 2
    ref = ref[c - res // 2:c + res // 2, c - res // 2:c + res // 2]
    ref /= ref.sum()
    report.check('FWHM SAOS / reference', fwhm_2d(frame) / fwhm_2d(ref), 1.0, rtol=tol['fwhm_rtol'])
    report.check('max |frame - reference| / peak', np.max(np.abs(frame - ref)) / ref.max(), 0.0,
                 atol=tol['reference_profile_atol'])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = (np.arange(res) - res // 2) / fw
    ax.semilogy(xs, frame[res // 2] / ref.max(), label='SAOS (row cut)')
    ax.semilogy(xs, ref[res // 2] / ref.max(), 'k--', lw=1, label='reference FFT')
    ax.set_xlabel('x [lambda/D]'); ax.set_ylim(1e-5, 1.2); ax.legend(fontsize=8)
    ax.set_title('Obstructed pupil (0.3): SAOS vs independent FFT')
    report.figure(fig, 'obstructed_cut')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Image motion and plate scale
# ----------------------------------------------------------------------------------------------------------------------

def test_image_motion_tilt(logger, tolerances, report):
    """
    A tilt of angle theta moves the image by theta / plate_scale. Phase ramps along x (columns) and y (rows) of
    0.5-6 lambda/D, camera sampled like the pupil (n_pix = res). Criteria: shift / expected = 1 on the tilted axis,
    no motion on the other axis, +ramp -> +pixel.
    """
    tol = tolerances['sciencecam']['motion']
    fw, res = 4.0, 64
    tel, src, cam = make_camera(logger, res=res, n_pix=res, fw=fw)
    X, Y = pupil_coords(tel)
    lam = src.wavelength
    p0 = np.array(peak_position(cam.get_frame(src, np.zeros((res, res)))))
    thetas = np.array([0.5, 1.0, 2.0, 4.0, 6.0])          # lambda/D
    meas = {'x': [], 'y': []}
    for axis, coord in (('x', X), ('y', Y)):
        for t in thetas:
            p = np.array(peak_position(cam.get_frame(src, 2 * np.pi * t * coord / tel.D))) - p0
            main, cross = (p[1], p[0]) if axis == 'x' else (p[0], p[1])
            meas[axis].append((main, cross))
    for axis in ('x', 'y'):
        main = np.array([m[0] for m in meas[axis]])
        cross = np.array([m[1] for m in meas[axis]])
        report.check(f'{axis} tilt: shift / (theta / plate scale)', main / (thetas * fw), 1.0, rtol=tol['shift_rtol'],
                     trend=False)
        report.check(f'{axis} tilt: motion on the other axis [px]', np.max(np.abs(cross)), 0.0, atol=tol['cross_axis_atol_px'])
    report.metric('x tilt 6 lambda/D: measured shift [px]', meas['x'][-1][0])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for axis, mk in (('x', 'o'), ('y', 's')):
        ax.plot(thetas, [m[0] for m in meas[axis]], mk, label=f'{axis} tilt (measured)')
    ax.plot(thetas, thetas * fw, 'k--', lw=1, label='theta / plate scale')
    ax.set_xlabel('tilt [lambda/D]'); ax.set_ylabel('image shift [px]'); ax.legend(fontsize=8)
    ax.set_title(f'Science image motion ({fw:g} px per lambda/D)')
    report.figure(fig, 'image_motion')
    report.assert_ok()


def test_plate_scale_with_resampled_pupil(logger, tolerances, report):
    """
    Camera field with n_pix != res (the usual case: EST red 555 px vs 216). Image motion and FWHM must still follow
    the requested plate scale: shift / (theta / ps) = 1 and FWHM = 1.029 lambda/D within 0.2%.
    """
    tol = tolerances['sciencecam']['plate_scale']
    fw, res, n_pix = 4.0, 64, 160
    tel, src, cam = make_camera(logger, res=res, n_pix=n_pix, fw=fw)
    X, _ = pupil_coords(tel)
    f0 = cam.get_frame(src, np.zeros((res, res)))
    t = 4.0
    p = np.array(peak_position(cam.get_frame(src, 2 * np.pi * t * X / tel.D))) - np.array(peak_position(f0))
    model = n_pix * (res - 1) / ((n_pix - 1) * res) * np.round(fw * n_pix) / (fw * n_pix)
    report.metric('align_corners scale model n_pix (res-1) / ((n_pix-1) res)', model)
    report.check('shift / (theta / plate scale)', p[1] / (t * fw), 1.0, rtol=tol['rtol'])
    report.check('FWHM / (1.029 lambda/D)', fwhm_2d(f0) / fw / theory.airy_fwhm(1, 1), 1.0, rtol=tol['rtol'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Strehl ratio of the PSF
# ----------------------------------------------------------------------------------------------------------------------

def test_strehl_vs_native_grid(logger, tolerances, rng, report):
    """
    On-axis intensity of the PSF relative to the aberration-free one must equal |<exp(i phi)>|^2 over the pupil grid
    the phase is defined on (exact Strehl for that sampled phase). High-order phase (PSD f^-11/3 above 8 cycles/D,
    like an AO residual), rms 0.5 and 0.9 rad, 4 realisations each; camera field n_pix = 2.5 res as in EST.
    """
    tol = tolerances['sciencecam']['strehl']
    fw, res, n_pix = 4.0, 64, 160
    tel, src, cam = make_camera(logger, res=res, n_pix=n_pix, fw=fw)
    pupil = cam.pupil
    m = pupil > 0
    c = n_pix // 2
    ideal = cam.compute_psf(np.zeros((res, res)), fw).numpy()
    rows = []
    for rms in (0.5, 0.9):
        for _ in range(4):
            ph = high_order_phase(rng, res, 8, rms, pupil)
            exact = np.abs(np.mean(np.exp(1j * ph[m]))) ** 2
            psf = cam.compute_psf(ph, fw).numpy()
            rows.append((rms, exact, psf[c, c] / ideal[c, c]))
    rows = np.array(rows)
    for rms in (0.5, 0.9):
        sel = rows[:, 0] == rms
        ratio = rows[sel, 2].mean() / rows[sel, 1].mean()
        report.check(f'SR SAOS / SR exact, rms {rms} rad (exact SR {rows[sel, 1].mean():.3f})', ratio, 1.0,
                     rtol=tol['sr_rtol'], trend=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rows[:, 1], rows[:, 2], 'o')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('exact SR |<exp(i phi)>|^2 (pupil grid)'); ax.set_ylabel('SR from ScienceCam.compute_psf')
    ax.set_title(f'Strehl: phase resampled {res} -> {n_pix} px'); ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)
    report.figure(fig, 'strehl_saos_vs_exact')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Long-exposure accumulation (LightPath)
# ----------------------------------------------------------------------------------------------------------------------

def frames_until_long_exposure(logger, fs, k, decimation=1, max_iter=200):
    """Number of LightPath iterations until the first long-exposure frame, for T_int = k * decimation / fs."""
    tel, src, cam = make_camera(logger, res=16, n_pix=16, fw=2.0, fs=fs, integrationTime=k * decimation / fs,
                                decimation=decimation)
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, sci=cam)
    n_frames = 0
    for i in range(max_iter):
        lp.propagate(True)
        if lp.sci_frame is not None:
            n_frames += 1
        if lp.long_exposure_frame is not None:
            return n_frames
    return np.nan


@pytest.mark.parametrize('fs', [
    1000.0, 2000.0,
    1500.0,
    1800.0,
])
def test_long_exposure_frame_count(fs, logger, report):
    """
    A long exposure of T_int = k * decimation * Ts must accumulate exactly k short frames (k = 1..12, decimation 1;
    and k = 1..6 with decimation 2), with T_int written as the user would (k / fs).
    """
    got = np.array([frames_until_long_exposure(logger, fs, k) for k in range(1, 13)])
    report.check(f'frames per long exposure at {fs:g} Hz, k = 1..12', got, np.arange(1, 13), atol=0)
    got2 = np.array([frames_until_long_exposure(logger, fs, k, decimation=2) for k in range(1, 7)])
    report.check(f'frames per long exposure at {fs:g} Hz, decimation 2, k = 1..6', got2, np.arange(1, 7), atol=0)
    bad = [int(k) for k, g in zip(range(1, 13), got) if g != k]
    if bad:
        report.note(f'k with a wrong frame count: {bad} (got {got[np.array(bad) - 1].tolist()})')
    report.assert_ok()


def test_long_exposure_mean_and_energy(logger, tolerances, rng, report):
    """
    Noise-free long exposure = mean of the unit-energy short-exposure frames of the exposure, scaled to the collected
    photons flux * T_int * area * throughput. Random tilts per frame (stub atmosphere), k = 8 frames. The expected
    frames come from an independent FFT of the pupil mask with the tilt phase ramp (not from SAOS frames); the
    collecting area is the pixelated pupil area, and the throughput is the lightRatio given to the camera.
    """
    tol = tolerances['sciencecam']['long_exposure']
    fs, k, res, fw, throughput = 1000.0, 8, 32, 3.0, 0.9
    tel, src, cam = make_camera(logger, res=res, n_pix=res, fw=fw, fs=fs, integrationTime=k / fs, lightRatio=throughput)
    X, Y = pupil_coords(tel)
    lam_d = src.wavelength / tel.D
    tilts = rng.normal(0, 1.0, size=(k, 2)) * lam_d
    atm = StubAtmosphere([tx * X + ty * Y for tx, ty in tilts])
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, atm=atm, sci=cam)
    for i in range(k):
        lp.propagate(True)
    le = lp.long_exposure_frame
    assert le is not None, 'no long exposure after k frames'
    pupil = np.asarray(tel.pupil, dtype=float)
    area = pupil.sum() * (tel.D / res) ** 2
    photons = src.flux * (k / fs) * area * throughput
    phase = lambda tx, ty: 2 * np.pi * (tx * X + ty * Y) / src.wavelength
    expected = np.mean([exact_frame(pupil, phase(tx, ty), fw, res) for tx, ty in tilts], axis=0)
    report.check('long exposure energy / (flux T_int area throughput)', le.sum() / photons, 1.0, rtol=tol['energy_rtol'])
    report.check('max |LE / photons - mean(independent frames)| / peak', np.max(np.abs(le / photons - expected)) /
                 expected.max(), 0.0, atol=tol['mean_rtol'])
    report.assert_ok()


def test_long_exposure_gaussian_tilt(logger, tolerances, report):
    """
    Random image motion with distribution p(tilt) blurs the long exposure into PSF_DL * p. Tilts on a 30 x 30
    stratified grid of Gaussian quantiles (sigma = 0.8 lambda/D per axis, 900 frames, fed through LightPath).
    Criteria: (1) LE = mean of the independent (FFT) frames of the tilts actually used;
    (2) LE = independent DL frame * Gaussian(sigma) (continuous theory; the quantile grid approximates it to < 1%).
    The quadrature law FWHM_LE^2 = FWHM_DL^2 + (2.355 sigma)^2 is reported as context (approximate for an Airy core).
    """
    from scipy.special import ndtri
    tol = tolerances['sciencecam']['long_exposure']
    fs, res, fw, sigma = 1000.0, 48, 3.0, 0.8
    n = 30
    tel, src, cam = make_camera(logger, res=res, n_pix=res, fw=fw, fs=fs, integrationTime=n * n / fs)
    X, Y = pupil_coords(tel)
    lam_d = src.wavelength / tel.D
    q = ndtri((np.arange(n) + 0.5) / n) * sigma
    tilts = [(a, b) for a in q for b in q]
    atm = StubAtmosphere([(tx * X + ty * Y) * lam_d for tx, ty in tilts])
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, atm=atm, sci=cam)
    for _ in range(n * n):
        lp.propagate(True)
    le = lp.long_exposure_frame
    assert le is not None
    le = le / le.sum()
    pupil = np.asarray(tel.pupil, dtype=float)
    dl = exact_frame(pupil, np.zeros((res, res)), fw, res)                 # independent diffraction-limited frame
    f = np.fft.fftfreq(res)
    FX, FY = np.meshgrid(f, f)
    # Expected LE = mean over the tilts actually used of the independent tilted frames (tilt a along X -> +a fw px)
    emp = np.mean([exact_frame(pupil, 2 * np.pi * (a * X + b * Y) / tel.D, fw, res) for a, b in tilts], axis=0)
    report.check('max |LE - mean(independent tilted frames)| / peak', np.max(np.abs(le - emp)) / emp.max(), 0.0,
                 atol=tol['discrete_profile_atol'])
    ref = np.real(np.fft.ifft2(np.fft.fft2(dl) * np.exp(-2 * (np.pi * sigma * fw) ** 2 * (FX ** 2 + FY ** 2))))
    ref /= ref.sum()
    report.check('FWHM LE / FWHM (DL * Gaussian)', fwhm_2d(le) / fwhm_2d(ref), 1.0, rtol=tol['gaussian_fwhm_rtol'])
    report.check('max |LE - DL * Gaussian| / peak', np.max(np.abs(le - ref)) / ref.max(),  0.0,
                 atol=tol['gaussian_profile_atol'])
    s_px = sigma * fw
    quad = np.sqrt(fwhm_2d(dl) ** 2 + (2.3548 * s_px) ** 2)
    report.metric('FWHM LE / quadrature law sqrt(FWHM_DL^2 + (2.355 sigma)^2)', fwhm_2d(le) / quad)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = (np.arange(res) - res // 2) / fw
    ax.plot(xs, le[res // 2] / ref.max(), label='SAOS long exposure (row cut)')
    ax.plot(xs, ref[res // 2] / ref.max(), 'k--', lw=1, label='DL frame * Gaussian(sigma)')
    ax.plot(xs, dl[res // 2] / dl.max(), color='0.6', lw=0.8, label='diffraction limited (peak 1)')
    ax.set_xlabel('x [lambda/D]'); ax.legend(fontsize=8)
    ax.set_title(f'Long exposure with Gaussian image motion, sigma = {sigma} lambda/D per axis')
    report.figure(fig, 'long_exposure_gaussian_tilt')
    report.assert_ok()
