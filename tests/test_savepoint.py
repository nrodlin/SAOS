"""
T1 validation of the metrics saved by Savepoint and of the HDF5 output.

Savepoint.compute_stats is fed synthetic data with a known answer (OPD maps, slope vectors, PSFs computed here by an
independent FFT, solar-like images); the HDF5 round trip and the saving cadence use a small LightPath whose atmosphere
is a stub returning prescribed OPDs. Files go to pytest's tmp_path.
"""
import types

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import theory
from citools import known_issue
from SAOS.LightPath import LightPath
from SAOS.Savepoint import Savepoint
from SAOS.ScienceCam import ScienceCam
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

RAD2ARCSEC = 206265


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def circular_pupil(res, co=0.0):
    x = np.arange(res) - (res - 1) / 2
    r = np.hypot(*np.meshgrid(x, x))
    return (r <= res / 2) & (r >= co * res / 2)


def exact_frame(pupil, phase, fw, n_pix, tilt_px=(0.0, 0.0)):
    """
    Independent PSF: |FFT|^2 of the pupil zero-padded to fw px per lambda/D, cropped to n_pix around the optical
    axis. tilt_px = image shift (row, col) in pixels, applied as a phase ramp.
    """
    res = pupil.shape[0]
    x = np.arange(res) - (res - 1) / 2
    X, Y = np.meshgrid(x, x)
    ph = phase + 2 * np.pi * (tilt_px[1] * X + tilt_px[0] * Y) / (fw * res)
    psf = theory.psf_reference(pupil.astype(float), ph, oversampling=fw)
    c = psf.shape[0] // 2
    return psf[c - n_pix // 2:c - n_pix // 2 + n_pix, c - n_pix // 2:c - n_pix // 2 + n_pix]


def high_order_phase(rng, res, f_cut, rms, pupil, f_max=np.inf):
    f = np.fft.fftfreq(res) * res
    F = np.hypot(*np.meshgrid(f, f))
    amp = np.where((F > f_cut) & (F <= f_max), np.maximum(F, 1e-9) ** (-11 / 6), 0.0)
    ph = np.real(np.fft.ifft2(amp * (rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res)))))
    ph -= ph[pupil].mean()
    return ph * rms / ph[pupil].std()


def point_source_lp(ideal, wavelength):
    """Minimal LightPath stand-in for compute_stats('sci_frame') of a point source."""
    return types.SimpleNamespace(src=types.SimpleNamespace(tag='source', wavelength=wavelength),
                                 sci=types.SimpleNamespace(fake_src_dict={str(int(wavelength * 1e9)): ideal}))


def saos_strehl(sp, frame, ideal, wavelength=500e-9):
    return float(sp.compute_stats(frame[None], None, 'sci_frame', point_source_lp(ideal, wavelength))['strehl'][0])


class StubAtmosphere:
    def __init__(self, opds):
        self.opds, self.k = list(opds), 0

    def getOPD(self, src):
        opd = self.opds[self.k % len(self.opds)]
        self.k += 1
        return opd.copy()


def small_lightpath(logger, rng, n_opd, decimation=1, res=32, rms_rad=0.6):
    """
    LightPath whose atmosphere is a stub returning n_opd prescribed OPDs (band-limited, 4-7 cycles/D, rms_rad at the
    source wavelength, so that the halo stays inside the field). Science camera at 2 px per lambda/D (Nyquist) with
    n_pix = res, i.e. no resampling of the pupil; field +-8 lambda/D for res 32.
    """
    tel = Telescope(resolution=res, diameter=1.0, samplingTime=1e-3, logger=logger)
    src = Source(optBand='R', magnitude=5, logger=logger)
    ps = src.wavelength / tel.D * RAD2ARCSEC / 2.0
    cam = ScienceCam(fieldOfView=res * ps, plate_scale=ps, samplingTime=1e-3, telescope=tel, logger=logger,
                     device='cpu', decimation=decimation)
    assert cam.nPix == res
    pupil = tel.pupil.astype(bool)
    opds = [high_order_phase(rng, res, 4, rms_rad, pupil, f_max=7) * src.wavelength / (2 * np.pi) for _ in range(n_opd)]
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, atm=StubAtmosphere(opds), sci=cam)
    return lp, opds


def expected_frame(lp, opd):
    """Independent noise-free science frame (unit energy in the field) and strehl of an OPD map on the pupil mask."""
    pupil = lp.tel.pupil.astype(float)
    ph = 2 * np.pi * opd * pupil / lp.src.wavelength
    fr = exact_frame(pupil, ph, 2, pupil.shape[0])
    return fr / fr.sum()


@pytest.fixture
def savepoint(logger, tmp_path):
    def make(**kw):
        return Savepoint(file_path=str(tmp_path / 'savepoint.h5'), logger=logger, **kw)
    return make


# ----------------------------------------------------------------------------------------------------------------------
# Statistics of maps and slopes
# ----------------------------------------------------------------------------------------------------------------------

def test_opd_statistics(savepoint, rng, report):
    """
    OPD/phase statistics are computed over the pupil pixels only: min, max, mean, std (piston-removed rms, i.e. the
    WFE) and rms. Asymmetric map (ramp + offset + bump in one quadrant, large values outside the pupil).
    """
    res = 32
    pupil = circular_pupil(res, 0.3)
    x = np.arange(res) - res / 2
    X, Y = np.meshgrid(x, x)
    opd = 1e-7 * (0.3 + 0.02 * X - 0.01 * Y) + 2e-7 * np.exp(-((X - 6) ** 2 + (Y + 7) ** 2) / 8)
    opd[~pupil] = 1e-3
    st = savepoint().compute_stats(opd[None], pupil[None], 'opd')
    v = opd[pupil]
    for name, ref in (('min', v.min()), ('max', v.max()), ('mean', v.mean()), ('std', v.std()),
                      ('rms', np.sqrt(np.mean(v ** 2)))):
        report.check(f'{name} / direct computation over the pupil', float(np.squeeze(st[name])) / ref, 1.0, rtol=1e-12)
    report.assert_ok()


def test_slopes_1d_statistics_split_xy(savepoint, rng, report):
    """slopes_1D = [x (n), y (n)]: statistics are reported per axis as [[x, y]]."""
    n = 40
    sx, sy = rng.normal(-1, 0.1, n), rng.normal(3, 0.5, n)
    st = savepoint().compute_stats(np.r_[sx, sy], None, 'slopes_1D')
    for name, f in (('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std),
                    ('rms', lambda a: np.sqrt(np.mean(a ** 2)))):
        report.check(f'{name} x', st[name][0, 0], f(sx), rtol=1e-12)
        report.check(f'{name} y', st[name][0, 1], f(sy), rtol=1e-12)
    report.assert_ok()


def test_slopes_2d_statistics_split_xy(savepoint, rng, report):
    """slopes_2D = [x block (n rows); y block (n rows)] as saved by LightPath: min/max per axis."""
    n = 8
    sx, sy = rng.uniform(-1, 0, (n, n)), rng.uniform(5, 6, (n, n))
    st = savepoint().compute_stats(np.r_[sx, sy][None], None, 'slopes_2D')     # Savepoint expands 2D data to 3D
    report.check('min x', st['min'][0, 0], sx.min(), rtol=1e-12)
    report.check('max x', st['max'][0, 0], sx.max(), rtol=1e-12)
    report.check('min y', st['min'][0, 1], sy.min(), rtol=1e-12)
    report.check('max y', st['max'][0, 1], sy.max(), rtol=1e-12)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Strehl ratio
# ----------------------------------------------------------------------------------------------------------------------

def test_strehl_known_phase(savepoint, rng, tolerances, report):
    """
    Strehl of a PSF with a known high-order phase (PSD f^-11/3 above 4 cycles/D, no tilt): the saved metric must
    equal |<exp(i phi)>|^2 over the pupil. Frames from an independent FFT at 2 px per lambda/D (Nyquist), wide field
    (32 lambda/D) so that the field edge plays no role.
    """
    tol = tolerances['savepoint']['strehl']
    res, fw, n_pix = 32, 2, 64
    pupil = circular_pupil(res)
    sp = savepoint()
    ideal = exact_frame(pupil, np.zeros((res, res)), fw, n_pix)
    rows = []
    for rms in (0.3, 0.6, 0.9):
        for _ in range(3):
            ph = high_order_phase(rng, res, 4, rms, pupil)
            exact = np.abs(np.mean(np.exp(1j * ph[pupil]))) ** 2
            rows.append((rms, exact, saos_strehl(sp, exact_frame(pupil, ph, fw, n_pix), ideal)))
    rows = np.array(rows)
    report.check('SR saved / SR exact (9 frames, SR 0.45-0.92)', rows[:, 2] / rows[:, 1], 1.0, rtol=tol['known_phase_rtol'])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rows[:, 1], rows[:, 2], 'o')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('exact SR |<exp(i phi)>|^2'); ax.set_ylabel('Savepoint strehl')
    ax.set_title('Saved Strehl vs exact (independent PSFs, Nyquist)')
    report.figure(fig, 'strehl_known_phase')
    report.assert_ok()


def test_strehl_subpixel_tilt(savepoint, tolerances, report):
    """
    A pure tilt does not change the Strehl ratio: a diffraction-limited PSF shifted by 0, 0.25 and 0.5 px (both axes)
    at Nyquist sampling must score SR = 1.
    """
    tol = tolerances['savepoint']['strehl']
    res, fw, n_pix = 32, 2, 64
    pupil = circular_pupil(res)
    sp = savepoint()
    ideal = exact_frame(pupil, np.zeros((res, res)), fw, n_pix)
    for s in (0.25, 0.5):
        sr = saos_strehl(sp, exact_frame(pupil, np.zeros((res, res)), fw, n_pix, tilt_px=(s, s)), ideal)
        report.check(f'SR of a perfect PSF shifted by ({s}, {s}) px', sr, 1.0, rtol=tol['tilt_rtol'])
    report.assert_ok()


def test_strehl_in_field_convention(savepoint, rng, tolerances, report):
    """
    The saved Strehl is the measured SR in the camera field (documented convention, the owner's decision 2026-10-27):
    SR = (peak / energy in the field) / (ideal peak / ideal energy in the field) = SR_peak * f0 / f, with f and f0 the
    fractions of the total energy inside the field for the aberrated and the ideal PSF. Independent terms: SR_peak = maximum
    of the aberrated PSF sampled 16x finer (unit ideal peak), f / f0 from sums of independent frames normalised to the
    same total energy. Checked for a 32 lambda/D and a 6 lambda/D field (high-order phase, 1 rad rms); the exact
    |<exp(i phi)>|^2 is recorded.
    """
    tol = tolerances['savepoint']['strehl']
    res, fw = 32, 2
    pupil = circular_pupil(res)
    sp = savepoint()
    ph = high_order_phase(rng, res, 4, 1.0, pupil)
    report.metric('exact SR |<exp(i phi)>|^2', np.abs(np.mean(np.exp(1j * ph[pupil]))) ** 2)
    sr_peak = theory.psf_reference(pupil.astype(float), ph, oversampling=fw * 16).max()
    for n_pix in (64, 12):
        ideal = exact_frame(pupil, np.zeros((res, res)), fw, n_pix)
        frame = exact_frame(pupil, ph, fw, n_pix)
        expected = sr_peak * ideal.sum() / frame.sum()
        sr = saos_strehl(sp, frame, ideal)
        report.metric(f'f0 / f, field {n_pix // fw} lambda/D', ideal.sum() / frame.sum())
        report.check(f'SR / in-field convention, field {n_pix // fw} lambda/D', sr / expected, 1.0, rtol=tol['convention_rtol'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Extended-object metrics
# ----------------------------------------------------------------------------------------------------------------------

def mfgs_reference(img):
    """
    Independent MFGS (mean fraction of gradient similarity) written from its definition with explicit loops: image
    divided by its mean; M = 3x3 median filter (symmetric border extension, as the gradients); gradient magnitude of the image and of M
    with the 3x3 kernel [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]] and its transpose (symmetric border extension);
    similarity (2 g_I g_M + C) / (g_I^2 + g_M^2 + C), C = 1e-10; MFGS = mean over pixels.
    """
    a = np.asarray(img, dtype=float)
    a = a / a.mean()
    n0, n1 = a.shape
    pz = np.pad(a, 1, mode='symmetric')
    med = np.array([[np.sort(pz[i:i + 3, j:j + 3].ravel())[4] for j in range(n1)] for i in range(n0)])
    kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=float)
    ky = kx.T

    def grad(m):
        p = np.pad(m, 1, mode='symmetric')
        g = np.zeros_like(m)
        for i in range(n0):
            for j in range(n1):
                w = p[i:i + 3, j:j + 3]
                g[i, j] = np.hypot((w * kx).sum(), (w * ky).sum())
        return g

    gi, gm = grad(a), grad(med)
    c = 1e-10
    return float(np.mean((2 * gi * gm + c) / (gi ** 2 + gm ** 2 + c)))


def test_contrast_and_mfgs(savepoint, report):
    """
    Solar-image metrics on small deterministic images. Contrast = std/mean: 1 + a sin(2 pi m x / N) over whole periods
    gives a/sqrt(2). MFGS: a deterministic 12 x 12 texture (smooth waves + a hot pixel + a pseudo-random pattern) must equal the
    independent loop implementation `mfgs_reference` for several amplitudes of the high-frequency pattern, with
    contrast and MFGS invariant to the image scale (frames are saved in photons or DN). The border of the median filter is
    extended symmetrically (zero padding biased the border medians low, SAV-MFGS-BORDER).
    """
    sp = savepoint()
    lp = types.SimpleNamespace(src=types.SimpleNamespace(tag='sun'))
    n, a = 64, 0.12
    x = np.arange(n)
    sine = np.tile(1 + a * np.sin(2 * np.pi * 4 * x / n), (n, 1))
    report.check('contrast of 1 + a sin / (a / sqrt 2)', float(sp.compute_stats(sine[None], None, 'sci_frame', lp)['contrast'][0])
                 / (a / np.sqrt(2)), 1.0, rtol=1e-12)
    flat = np.full((12, 12), 3.7)
    report.check('contrast of a constant image', float(sp.compute_stats(flat[None], None, 'sci_frame', lp)['contrast'][0]),
                 0.0, atol=1e-12)
    m = 12
    ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
    pattern = ((ii * 7 + jj * 13 + ii * jj * 5) % 11 - 5) / 5.0           # deterministic, zero-mean-ish, values in [-1, 1]
    smooth = 1 + 0.3 * np.sin(2 * np.pi * jj / m) * np.cos(2 * np.pi * ii / m)
    values = {}
    for amp in (0.0, 0.05, 0.3):
        img = smooth + amp * pattern
        img[3, 8] += 0.8                                                    # hot pixel
        ref = mfgs_reference(img)
        values[amp] = ref
        for k in (1.0, 1e-6, 1e6):
            st = sp.compute_stats((k * img)[None], None, 'sci_frame', lp)
            report.check(f'MFGS / reference, pattern {amp}, scale {k:g}', float(st['MFGS'][0]) / ref, 1.0, rtol=1e-9)
            report.check(f'contrast / (std/mean), pattern {amp}, scale {k:g}', float(st['contrast'][0]) / (img.std() / img.mean()), 1.0, rtol=1e-9)
    report.metric('reference MFGS, pattern 0 / 0.05 / 0.3', [values[k] for k in (0.0, 0.05, 0.3)])
    report.assert_ok()


def test_mfgs_constant_image(savepoint, report):
    """An image without any structure has no gradients in either image, so every pixel is similar: MFGS = 1."""
    lp = types.SimpleNamespace(src=types.SimpleNamespace(tag='sun'))
    for k in (3.7, 1e5):
        st = savepoint().compute_stats(np.full((12, 12), k)[None], None, 'sci_frame', lp)
        report.check(f'MFGS of a constant image ({k:g})', float(st['MFGS'][0]), 1.0, atol=1e-12)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# HDF5 output and saving cadence
# ----------------------------------------------------------------------------------------------------------------------

def test_hdf5_roundtrip(logger, savepoint, rng, tolerances, report):
    """
    Data and metrics saved for a LightPath over 6 iterations equal what was physically injected: the atmosphere OPD
    per direction (the prescribed stub OPD, iteration index, std over the pupil), the science OPD (the prescribed OPD
    on the pupil mask), the short-exposure science frames (one per iteration; independent FFT of the pupil with the
    prescribed phase, unit energy in the field) and their saved Strehl ratio (the exact |<exp(i phi)>|^2 of the
    prescribed phase, band-limited 4-7 cycles/D so that the halo is inside the +-8 lambda/D field).
    """
    tol = tolerances['savepoint']
    n_it = 6
    lp, opds = small_lightpath(logger, rng, n_it)
    sp = savepoint(atm_per_dir=1, sci=1, sci_frame=1, only_metrics=False)
    for i in range(n_it):
        lp.propagate(True)
        sp.save([lp], i)
    pupil = lp.tel.pupil.astype(bool)
    exp_frames = np.array([expected_frame(lp, o) for o in opds])
    exp_sr = np.array([np.abs(np.mean(np.exp(2j * np.pi * o[pupil] / lp.src.wavelength))) ** 2 for o in opds])
    with h5py.File(sp.file_path, 'r') as f:
        g = f['LightPath_0']
        atm = g['atm_opd/data'][:].reshape(n_it, *pupil.shape)
        sci = g['sci_opd/data'][:].reshape(n_it, *pupil.shape)
        frames = g['sci_frame_shortExp/data'][:].reshape(n_it, *exp_frames.shape[1:])
        report.check('iteration index (atm_opd)', g['atm_opd/iteration'][:], np.arange(n_it), atol=0)
        report.check('iteration index (sci_opd)', g['sci_opd/iteration'][:], np.arange(n_it), atol=0)
        report.check('iteration index (sci_frame_shortExp)', g['sci_frame_shortExp/iteration'][:], np.arange(n_it), atol=0)
        report.check('atm_opd data / prescribed OPD on the pupil (max abs diff) [m]',
                     np.max(np.abs((atm - np.array(opds))[:, pupil])), 0.0, atol=1e-15)
        report.check('sci_opd data / prescribed OPD on the pupil (max abs diff) [m]',
                     np.max(np.abs((sci - np.array(opds))[:, pupil])), 0.0, atol=1e-15)
        report.check('sci_opd outside the pupil is zero [m]', np.max(np.abs(sci[:, ~pupil])), 0.0, atol=0)
        report.check('sci_opd std == std of the prescribed OPD over the pupil', np.ravel(g['sci_opd/std'][:]),
                     [o[pupil].std() for o in opds], rtol=1e-9)
        report.check('short-exposure frames saved', frames.shape[0], n_it, atol=0)
        report.check('saved frame vs independent FFT frame (max abs diff / peak)',
                     np.max(np.abs(frames - exp_frames), axis=(1, 2)) / exp_frames.max(axis=(1, 2)), 0.0, atol=tol['frame_atol'])
        sr = np.ravel(g['sci_frame_shortExp/strehl'][:])
        report.check('strehl saved for every frame', sr.shape[0], n_it, atol=0)
        report.check('saved strehl / exact |<exp(i phi)>|^2', sr / exp_sr, 1.0, rtol=tol['strehl']['known_phase_rtol'])
    report.assert_ok()


def test_saved_frames_match_decimation(logger, savepoint, rng, tolerances, report):
    """
    Science camera decimated by 3 and frames saved every 3 iterations: the 4 frames taken in 12 iterations (at
    iterations 2, 5, 8, 11: the camera takes a frame at the last iteration of every decimation window, (i + 1) % 3 == 0, the
    Savepoint convention) must be saved, with the iteration index of the OPD they come from and the content of the
    independent FFT frame of that OPD.
    """
    tol = tolerances['savepoint']
    n_it, dec = 12, 3
    lp, opds = small_lightpath(logger, rng, n_it, decimation=dec)
    sp = savepoint(sci_frame=dec, only_metrics=False)
    for i in range(n_it):
        lp.propagate(True)
        sp.save([lp], i)
    want = np.arange(dec - 1, n_it, dec)
    with h5py.File(sp.file_path, 'r') as f:
        has = 'sci_frame_shortExp' in f['LightPath_0']
        n_saved = f['LightPath_0/sci_frame_shortExp/data'].shape[0] if has else 0
        report.check('frames saved', n_saved, len(want), atol=0)
        if has:
            its = np.ravel(f['LightPath_0/sci_frame_shortExp/iteration'][:])
            report.check('iteration indices of the saved frames', its, want[:len(its)], atol=0)
            frames = f['LightPath_0/sci_frame_shortExp/data'][:]
            frames = frames.reshape(frames.shape[0], *lp.tel.pupil.shape)
            for k, it in enumerate(its):
                if it in want:
                    report.check(f'saved frame {k} (iteration {it}) vs independent frame of OPD {it} (max abs diff / peak)',
                                 np.max(np.abs(frames[k] - expected_frame(lp, opds[it]))) / expected_frame(lp, opds[it]).max(),
                                 0.0, atol=tol['frame_atol'])
    report.assert_ok()
