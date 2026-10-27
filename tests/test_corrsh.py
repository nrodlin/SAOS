"""
T1 validation of SAOS.CorrelatingShackHartmann (solar correlating SH, used with ExtendedSource): spot optics (get_psf), image
formation (compute_images) and the end-to-end correlation measurement (wfs_measure).

Configuration: 6x6 subapertures of d = 0.1 m over a 48 px pupil (8 phase px per subaperture), V band (500 nm), plate scale
0.4"/px (lambda/d = 2.58 px), FoV 10", ExtendedSource 3x3 sub-directions (FoV 10", padding 20.1", margin 8": sun patches of
58 px per subaperture), use_brightest = 5 (the setting of the owner's validation scripts). A global tilt of theta arcsec moves
the image by theta / plate_scale px.

Independent references (no SAOS code, no SAOS-measured quantity):
 - optics: the far field of a subaperture is a direct Fourier sum (ref_spot) of the continuous complex amplitude over the
   physical subaperture (side d, M cells), evaluated at the angle (k - npix_sun//2) plate_scale of pixel k, the contract of
   get_psf (PSF centred at npix_sun//2, window of npix_sun px, unit energy in the window);
 - convolution: a direct circular convolution (sum of rolled copies of the object) of an object that this module builds;
 - measurement: the true image shift theta / plate_scale of a global tilt.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from citools import known_issue
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from test_wfs import D, NSUB, NP, P, RES, SUBAP, X, Y, annulus_mask, make_wfs as make_sh, SUB_C

pytestmark = pytest.mark.fast

ARCSEC = np.pi / 180 / 3600
PS = 0.4                      # WFS plate scale ["/px]
LAM = 500e-9                  # V band
SAMP = LAM / SUBAP / ARCSEC / PS            # px per lambda/d
K_G = 2 * np.pi / LAM * P * PS * ARCSEC     # phase gradient [rad/px] per px of spot motion (2 pi / lambda * pixel * angle)
DET_EXACT = dict(noiseFlag=True, quantumEfficiency=1.0, shotNoise=0, darkCurrent=0, readoutNoise=0, quantization_conversion=1,
                 fullWellCapacity=1e12, nBits=32, darkCalibration=0, randomState=1)

_SRC = {}


def make_source(logger, margin=8.0):
    if margin not in _SRC:
        _SRC[margin] = ExtendedSource(optBand='V', nSubDirs=3, fov=10, subDir_margin=margin, patch_padding=20.1, img_PS=0.0128,
                                      logger=logger)
    return _SRC[margin]


def make_csh(logger, pupil=None, fov=10.0, margin=8.0, use_brightest=5, **kw):
    """Correlating SH with 6x6 subapertures; pupil: None (square, all valid) or a binary mask of RES x RES px."""
    pup = np.ones((RES, RES), dtype=bool) if pupil is None else pupil
    tel = Telescope(resolution=RES, diameter=D, samplingTime=1e-3, pupil=pup, logger=logger)
    src = make_source(logger, margin)
    wfs = CorrelatingShackHartmann(nSubap=NSUB, telescope=tel, src=src, lightRatio=1.0, plate_scale=PS, fieldOfView=fov, guardPx=2,
                                   fft_fieldOfView_oversampling=0.5, use_brightest=use_brightest, logger=logger, device='cpu', **kw)
    return wfs, tel, src


def tilt_phase(shift_px, axis='x'):
    """Phase [rad] of a global tilt that moves the image by shift_px px along the columns (x) or rows (y)."""
    return shift_px * K_G * (X if axis == 'x' else Y)


# ----------------------------------------------------------------------------------------------------------------------
# Independent optics reference
# ----------------------------------------------------------------------------------------------------------------------

def ref_spot(phase_fn, n, M=256, amp=None, samp=SAMP):
    """
    Far field of one subaperture on the n x n pixel window of get_psf, normalised to unit sum, [row = y, column = x].
    phase_fn(xi, eta): phase [rad] at the normalised subaperture coordinates (columns, rows) in [-1/2, 1/2] (cell centres of an
    M x M grid); amp: optional M x M amplitude. Pixel k sits at u = (k - n//2)/samp [lambda/d].
    """
    s = (np.arange(M) + 0.5) / M - 0.5
    XI, ET = np.meshgrid(s, s)
    field = np.exp(1j * phase_fn(XI, ET)) if amp is None else amp * np.exp(1j * phase_fn(XI, ET))
    u = (np.arange(n) - n // 2) / samp
    Ex = np.exp(-2j * np.pi * np.outer(u, s))
    img = np.abs(Ex @ field @ Ex.T / M ** 2) ** 2
    return img / img.sum(), img.sum()


def cog(img):
    """Centroid (x = column, y = row) [px] relative to the pixel n//2 of a window; works on [..., n, n]."""
    n = img.shape[-1]
    k = np.arange(n) - n // 2
    tot = img.sum(axis=(-2, -1))
    return (img.sum(-2) * k).sum(-1) / tot, (img.sum(-1) * k).sum(-1) / tot


def fit_sinc2_position(profile):
    """
    Position [px] (index units) of the maximum of a cut through the spot of a square aperture, fitted with the exact shape
    A sinc^2((k - k0) / SAMP) (the cut through the peak of the separable sinc^2 spot) by least squares over the whole window.
    """
    from scipy import optimize
    k = np.arange(len(profile), dtype=float)
    f = lambda p: p[0] * np.sinc((k - p[1]) / SAMP) ** 2 - profile
    r = optimize.least_squares(f, [profile.max(), k[np.argmax(profile)]], x_scale=[profile.max(), 1.0], xtol=1e-12, ftol=1e-14)
    return r.x[1]


def second_moment(img):
    """Second central moment along x and y [px^2]."""
    n = img.shape[-1]
    k = np.arange(n) - n // 2
    px, py = img.sum(-2), img.sum(-1)
    px, py = px / px.sum(-1, keepdims=True), py / py.sum(-1, keepdims=True)
    cx, cy = (px * k).sum(-1), (py * k).sum(-1)
    return (px * (k - cx[..., None]) ** 2).sum(-1), (py * (k - cy[..., None]) ** 2).sum(-1)


def psf_of(wfs, src, phases):
    """get_psf of the stack phases [C, RES, RES]: numpy [nValid, C, n, n] (valid subapertures only)."""
    fwhm = src.wavelength * 206265 / (wfs.subaperture_size * wfs.plate_scale)
    return wfs.get_psf(np.asarray(phases, dtype=float), fwhm, wfs.npix_sun).numpy()[wfs.valid_subapertures_1D]


# ----------------------------------------------------------------------------------------------------------------------
# A: optics (get_psf)
# ----------------------------------------------------------------------------------------------------------------------

def test_spot_tilt_displacement(logger, report, tolerances):
    """
    A tilt theta (arcsec) over the pupil moves every spot by theta / plate_scale px along its axis and leaves the other axis
    untouched: the far field of a subaperture with a linear phase of 2 pi (theta/(lambda/d)) xi (ref_spot). The centroid is
    taken on the npix_sun window, relative to the pixel npix_sun//2 (truncation of the window is part of the reference).
    Shifts 0.05 ... 2 px, both axes and signs; compared: centroid per subaperture, the whole spot image and, independently of the
    window, the position of the spot (sinc^2 fit) against theta / plate_scale itself.
    """
    tol = tolerances['corrsh']['tilt']
    wfs, tel, src = make_csh(logger)
    n = wfs.npix_sun
    shifts = np.array([0.05, 0.3, 1.0, 2.0])
    cases = [(s * sg, ax) for s in shifts for sg in (1, -1) for ax in 'xy']
    psf = psf_of(wfs, src, [tilt_phase(s, ax) for s, ax in cases])
    worst_c, worst_img, worst_cross, spread, worst_peak = 0.0, 0.0, 0.0, 0.0, 0.0
    ref_x = {}
    for c, (s, ax) in enumerate(cases):
        ref, _ = ref_spot(lambda xi, et: 2 * np.pi * s / SAMP * (xi if ax == 'x' else et), n)
        ref_x[(s, ax)] = ref
        cx, cy = cog(psf[:, c])
        rx, ry = cog(ref)
        d, cross = (cx, cy), (cy, cx)
        direct, other = ((cx, rx), (cy, ry)) if ax == 'x' else ((cy, ry), (cx, rx))
        worst_c = max(worst_c, np.abs(direct[0] - direct[1]).max())
        worst_cross = max(worst_cross, np.abs(other[0] - other[1]).max())
        spread = max(spread, np.ptp(direct[0]))
        worst_img = max(worst_img, np.abs(psf[:, c] / psf[:, c].sum((-2, -1), keepdims=True) - ref).max() / ref.max())
        cut = psf[0, c][n // 2, :] if ax == 'x' else psf[0, c][:, n // 2]
        worst_peak = max(worst_peak, abs(fit_sinc2_position(cut) - n // 2 - s))
    report.check('max |CoG - reference CoG| (direct axis)', worst_c, 0.0, atol=tol['cog_atol'], unit='px')
    report.check('max |CoG - reference CoG| (other axis, cross term)', worst_cross, 0.0, atol=tol['cog_atol'], unit='px')
    report.check('spread of the CoG over the subapertures (pure tilt)', spread, 0.0, atol=tol['cog_atol'], unit='px')
    report.check('max |spot - reference| / reference peak', worst_img, 0.0, atol=tol['image_atol'])
    report.check('position of the spot (sinc^2 fit of the cut through the peak) - shift', worst_peak, 0.0,
                 atol=tol['peak_atol'], unit='px')
    c1 = [i for i, (s, ax) in enumerate(cases) if ax == 'x' and abs(s) == 1.0 and s > 0][0]
    report.metric('CoG of the +1 px spot (x, subaperture 0)', cog(psf[0, c1])[0], unit='px')
    s, ax = cases[c1]
    fig, a = plt.subplots(figsize=(5.5, 3.4))
    k = np.arange(n) - n // 2
    a.semilogy(k, psf[0, c1].sum(0), 'o-', ms=3, label='SAOS')
    a.semilogy(k, ref_x[(s, ax)].sum(0), '--', label='Fourier sum')
    a.set_xlabel('pixel - npix_sun//2'); a.set_ylabel('x profile'); a.set_title('spot, tilt 1 px along x', fontsize=8)
    a.legend(fontsize=7, frameon=False); a.grid(True, which='both', color='0.92')
    report.figure(fig, 'spot_profile_tilt')
    report.assert_ok()


def test_flat_spot_centred(logger, report, tolerances):
    """
    Flat phase: the spot of every subaperture is the real, even far field of a square aperture, centred at the pixel
    npix_sun//2: argmax at npix_sun//2, mirror symmetry I[k] = I[npix_sun - k] (k = 1 ... npix_sun - 1) and centroid equal to
    that of the reference window (whose asymmetric extent, pixels -n//2 ... n - n//2 - 1, gives a small non-zero value).
    """
    tol = tolerances['corrsh']['centring']
    wfs, tel, src = make_csh(logger)
    n = wfs.npix_sun
    psf = psf_of(wfs, src, np.zeros((1, RES, RES)))[:, 0]
    ref, _ = ref_spot(lambda xi, et: 0 * xi, n)
    pk = [np.unravel_index(np.argmax(p), p.shape) for p in psf]
    report.check('peak pixel row/col - npix_sun//2', np.array(pk) - n // 2, 0.0, atol=0)
    mirror = psf[:, 1:, 1:] - psf[:, :0:-1, :0:-1]
    report.check('max |I[k] - I[n-k]| / peak (mirror symmetry about npix_sun//2)', np.abs(mirror).max() / psf.max(), 0.0, atol=tol['mirror_atol'])
    cx, cy = cog(psf)
    rx, ry = cog(ref)
    report.metric('reference CoG of the window (x, y)', [rx, ry], unit='px')
    report.check('CoG x - reference', cx - rx, 0.0, atol=tol['cog_atol'], unit='px')
    report.check('CoG y - reference', cy - ry, 0.0, atol=tol['cog_atol'], unit='px')
    report.check('energy of the window (fully illuminated)', psf.sum((-2, -1)), 1.0, atol=tol['energy_atol'])
    report.assert_ok()


@pytest.mark.parametrize('phase', ['flat', 'defocus'])
def test_spot_second_moment(logger, report, tolerances, phase):
    """
    Second central moment M2 [px^2] of the spot on the npix_sun window (the quantity that sets the correlation width and the
    centroid noise) against the continuous-aperture Fourier sum (M = 512 cells) on the same window. The pupil sampling of
    get_psf has a design bias of ~1.22 (u/n)^2 <= 1 % (u: half-width of the patch in lambda/d, n: pupil samples per subaperture).
    'defocus': phi = c (x^2 + y^2) over the pupil, c = 0.01 rad/px^2, evaluated per subaperture at its own centre.
    """
    tol = tolerances['corrsh']['m2']
    wfs, tel, src = make_csh(logger)
    n = wfs.npix_sun
    c = 0.01
    ph = np.zeros((RES, RES)) if phase == 'flat' else c * (X ** 2 + Y ** 2)
    psf = psf_of(wfs, src, [ph])[:, 0]
    mx, my = second_moment(psf)
    worst = 0.0
    valid = np.argwhere(wfs.valid_subapertures)
    rel = []
    for (i, j), ax_, ay_ in zip(valid, mx, my):
        xc, yc = SUB_C[j], SUB_C[i]
        f = (lambda xi, et: 0 * xi) if phase == 'flat' else (lambda xi, et: c * ((xc + NP * xi) ** 2 + (yc + NP * et) ** 2))
        ref, _ = ref_spot(f, n, M=512 if phase == 'flat' else 256)
        rx, ry = second_moment(ref)
        rel += [ax_ / rx - 1, ay_ / ry - 1]
        if phase == 'flat':
            break                      # identical subapertures
    report.metric('npix_pupil (samples per subaperture)', wfs.npix_pupil)
    report.check('M2 / reference - 1 (max |.| over subapertures and axes)', np.abs(rel).max(), 0.0, atol=tol['rel_atol'])
    report.metric('M2 x, y of subaperture 0 [px^2]', [mx[0], my[0]], unit='px^2')
    report.assert_ok()


def test_partial_subaperture_energy_and_shape(logger, report, tolerances):
    """
    Circular pupil with a 20 % central obstruction (binary mask of 48 px; 8 px per subaperture), flat phase. (1) a subaperture is
    valid iff >= 50 % of its pixels are inside the pupil; (2) the unit-energy PSF of a fully lit subaperture becomes the fraction
    of the pupil pixels of that subaperture (transmitted energy, mean of the mask block); (3) the shape of the spot is the far
    field of the binary mask block (amplitude 1 on its pixels, normalised to unit window sum).
    """
    tol = tolerances['corrsh']['partial']
    m = annulus_mask(RES)
    wfs, tel, src = make_csh(logger, pupil=m)
    n = wfs.npix_sun
    frac = m.reshape(NSUB, NP, NSUB, NP).mean((1, 3))
    report.check('valid map == (fraction >= 0.5)', np.mean(wfs.valid_subapertures == (frac >= 0.5)), 1.0, atol=0)
    fwhm = src.wavelength * 206265 / (wfs.subaperture_size * wfs.plate_scale)
    psf_all = wfs.get_psf(np.zeros((1, RES, RES)), fwhm, n).numpy()[:, 0]
    energy = psf_all.sum((-2, -1)).reshape(NSUB, NSUB)
    v = frac >= 0.5
    report.check('energy / transmitted fraction (valid subapertures, max |.-1|)', np.abs(energy[v] / frac[v] - 1).max(), 0.0, atol=tol['energy_atol'])
    report.check('energy of the invalid subapertures', np.abs(energy[~v]).max(), 0.0, atol=0)
    worst = 0.0
    k = 16
    for i, j in np.argwhere(v & (frac < 1)):
        a = np.kron(m[i * NP:(i + 1) * NP, j * NP:(j + 1) * NP].astype(float), np.ones((k, k)))
        ref, _ = ref_spot(lambda xi, et: 0 * xi, n, M=NP * k, amp=a)
        p = psf_all[i * NSUB + j]
        worst = max(worst, np.abs(p / p.sum() - ref).max() / ref.max())
    report.metric('number of partially lit valid subapertures', int(np.sum(v & (frac < 1))))
    report.check('max |spot - reference| / reference peak (partial subapertures)', worst, 0.0, atol=tol['shape_atol'])
    fig, a = plt.subplots(figsize=(3.6, 3.4))
    im = a.imshow(energy / np.where(v, frac, 1), origin='upper', cmap='viridis', vmin=0.99, vmax=1.01)
    a.set_title('energy / transmitted fraction (valid)', fontsize=8)
    fig.colorbar(im, ax=a, fraction=0.046)
    report.figure(fig, 'partial_energy_ratio')
    report.assert_ok()


def sampled_quadratic(c, x):
    """
    c x^2 as a pixel map of the pupil (x in pupil px from the pupil centre): values at the pixel centres (-(RES-1)/2 ... (RES-1)/2),
    continued linearly between them and beyond the outermost pixel centres (the continuation of a pixel map).
    """
    xp = np.arange(RES) - (RES - 1) / 2
    vp = c * xp ** 2
    out = np.interp(x, xp, vp)
    out = np.where(x < xp[0], vp[0] + (vp[1] - vp[0]) * (x - xp[0]), out)
    return np.where(x > xp[-1], vp[-1] + (vp[-1] - vp[-2]) * (x - xp[-1]), out)


def test_defocus_mirror_antisymmetry(logger, report, tolerances):
    """
    Defocus phi = c (x^2 + y^2) about the pupil centre (square pupil, c = 0.013 rad/px^2: spot motion up to ~2 px): the x
    displacement of the spot of subaperture (i, j) is minus that of the mirror subaperture (i, N-1-j), independent of the row,
    and the y displacement likewise across the rows. The displacements also equal those of the reference (each subaperture
    propagated with its own phase, same window); the phase map is a pixel map of the quadratic, so the reference uses its piecewise
    linear continuation between the pixel centres and beyond the outermost ones (the continuous quadratic differs by up to 0.005 px
    in the edge subapertures, where the outer half pixel is extrapolated and the inner one interpolated: the pixelisation of the
    input, not of the sensor); the asymmetric window extent makes the reference itself
    antisymmetric only to its own small residual, which the SAOS residual must reproduce.
    """
    tol = tolerances['corrsh']['defocus']
    wfs, tel, src = make_csh(logger)
    n = wfs.npix_sun
    c = 0.013
    psf = psf_of(wfs, src, [c * (X ** 2 + Y ** 2)])[:, 0]
    cx, cy = (a.reshape(NSUB, NSUB) for a in cog(psf))
    rx, ry = np.zeros((NSUB, NSUB)), np.zeros((NSUB, NSUB))
    for i in range(NSUB):
        for j in range(NSUB):
            xc, yc = SUB_C[j], SUB_C[i]
            ref, _ = ref_spot(lambda xi, et: sampled_quadratic(c, xc + NP * xi) + sampled_quadratic(c, yc + NP * et), n)
            rx[i, j], ry[i, j] = (a for a in cog(ref))
    asym_x = cx + cx[:, ::-1]
    asym_y = cy + cy[::-1, :]
    ref_asym = max(np.abs(rx + rx[:, ::-1]).max(), np.abs(ry + ry[::-1, :]).max())
    report.metric('largest displacement', np.abs(cx).max(), unit='px')
    report.metric('reference antisymmetry residual (window asymmetry)', ref_asym, unit='px')
    report.check('max |cx[i,j] + cx[i,N-1-j]|', np.abs(asym_x).max(), 0.0, atol=ref_asym + tol['asym_atol'], unit='px')
    report.check('max |cy[i,j] + cy[N-1-i,j]|', np.abs(asym_y).max(), 0.0, atol=ref_asym + tol['asym_atol'], unit='px')
    report.check('cx independent of the row (max |cx - row mean|)', np.abs(cx - cx.mean(0)).max(), 0.0, atol=tol['asym_atol'], unit='px')
    report.check('max |cx - reference|', np.abs(cx - rx).max(), 0.0, atol=tol['cog_atol'], unit='px')
    report.check('max |cy - reference|', np.abs(cy - ry).max(), 0.0, atol=tol['cog_atol'], unit='px')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# B: convolution (compute_images)
# ----------------------------------------------------------------------------------------------------------------------

def direct_convolution(obj, psf):
    """Circular convolution out[r, c] = sum_ab psf[a, b] obj[r - (a - n//2), c - (b - n//2)] by summing rolled copies."""
    n = obj.shape[-1]
    out = np.zeros_like(obj, dtype=float)
    for a in range(n):
        for b in range(n):
            if psf[a, b] != 0:
                out += psf[a, b] * np.roll(obj, (a - n // 2, b - n // 2), axis=(0, 1))
    return out


def hires_object(small, k):
    """Object of (n k) x (n k) px whose k x k block mean is `small` [n, n] (compute_images downsamples by block means)."""
    return np.kron(small, np.ones((k, k)))


@pytest.mark.parametrize('margin', [8.0, 8.4], ids=['npix_sun_even', 'npix_sun_odd'])
def test_convolution_delta_not_shifted(logger, report, tolerances, margin):
    """
    The image of a point-like object (one bright pixel of the object grid) is the PSF of the subaperture placed at that pixel:
    a delta at the PSF origin npix_sun//2 gives the PSF itself (no shift, for even and odd npix_sun), and a delta at an
    offset (dr, dc) gives the PSF displaced by (dr, dc) (np.roll, circular). Spot centroids and the whole image are compared.
    """
    tol = tolerances['corrsh']['conv']
    wfs, tel, src = make_csh(logger, margin=margin)
    n, k = wfs.npix_sun, 4
    report.metric('npix_sun', n)
    ph = np.zeros((1, RES, RES))
    ph[0] = tilt_phase(0.7, 'x') + tilt_phase(-0.4, 'y')
    psf = np.repeat(psf_of(wfs, src, ph), 1, axis=1)
    fwhm = src.wavelength * 206265 / (wfs.subaperture_size * wfs.plate_scale)
    psf_t = wfs.get_psf(ph, fwhm, n)
    worst, worst_cog = 0.0, 0.0
    for dr, dc in ((0, 0), (-9, 6), (13, -4)):
        small = np.zeros((n, n))
        small[n // 2 + dr, n // 2 + dc] = 1.0
        sun = np.repeat(np.repeat(hires_object(small, k)[:, :, None, None], 3, 2), 3, 3)
        img = wfs.compute_images(psf_t, sun, n).numpy()[:, 0]             # [nValid, n, n]
        ref = np.roll(psf[:, 0], (dr, dc), axis=(1, 2))
        worst = max(worst, np.abs(img / img.sum((-2, -1), keepdims=True) - ref / ref.sum((-2, -1), keepdims=True)).max() / ref.max())
        cx, cy = cog(img)
        rx, ry = cog(ref)
        worst_cog = max(worst_cog, np.abs(cx - (rx)).max(), np.abs(cy - ry).max())
    report.check('max |image - displaced PSF| / peak', worst, 0.0, atol=tol['image_atol'])
    report.check('max |CoG(image) - CoG(displaced PSF)|', worst_cog, 0.0, atol=tol['cog_atol'], unit='px')
    report.assert_ok()


def test_convolution_matches_direct_and_conserves_flux(logger, report, tolerances):
    """
    A random positive object per sub-direction (smooth, 58 px) convolved with the spots of an obstructed pupil (flat and tilted
    phase): (1) every image equals the direct circular convolution of the object with the PSF (shape compared at unit sum);
    (2) the flux of the image of a subaperture is the flux of the object times the transmitted energy of that subaperture, with the
    same constant for all subapertures and sub-directions (a convolution with a unit-sum PSF conserves flux).
    """
    tol = tolerances['corrsh']['conv']
    m = annulus_mask(RES)
    wfs, tel, src = make_csh(logger, pupil=m)
    n, k = wfs.npix_sun, 4
    rng = np.random.default_rng(3)
    small = np.empty((n, n, 3, 3))
    for a in range(3):
        for b in range(3):
            g = rng.random((n, n)) ** 3
            f = np.fft.fft2(g) * np.exp(-(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2) / (2 * 0.08 ** 2))
            small[:, :, a, b] = 1 + np.real(np.fft.ifft2(f)) / np.abs(np.real(np.fft.ifft2(f))).max()
    sun = np.kron(small, np.ones((k, k, 1, 1)))
    ph = np.stack([tilt_phase(0.8 * (c - 4) / 4, 'x') + tilt_phase(0.5, 'y') * (c % 2) for c in range(9)])
    fwhm = src.wavelength * 206265 / (wfs.subaperture_size * wfs.plate_scale)
    psf_t = wfs.get_psf(ph, fwhm, n)
    img = wfs.compute_images(psf_t, sun, n).numpy()                       # [nValid, 9, n, n]
    psf = psf_t.numpy()[wfs.valid_subapertures_1D]
    frac = m.reshape(NSUB, NP, NSUB, NP).mean((1, 3)).ravel()[wfs.valid_subapertures_1D]
    worst = 0.0
    for s_, c in ((0, 0), (5, 4), (len(frac) - 1, 8), (12, 7)):
        ref = direct_convolution(small[:, :, c // 3, c % 3], psf[s_, c])
        im = img[s_, c]
        worst = max(worst, np.abs(im / im.sum() - ref / ref.sum()).max() / (ref / ref.sum()).max())
    report.check('max |image - direct convolution| / peak', worst, 0.0, atol=tol['image_atol'])
    flux = img.sum((-2, -1)) / (small.sum((0, 1)).reshape(1, 9) * frac[:, None])        # per subaperture and sub-direction
    report.metric('flux / (object flux x energy): mean', flux.mean())
    report.check('flux / (object flux x energy): std/mean over subapertures and sub-directions', flux.std() / flux.mean(), 0.0, atol=tol['flux_atol'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C: correlation measurement (wfs_measure)
# ----------------------------------------------------------------------------------------------------------------------

def register_shift(a, b, margin=4):
    """
    Displacement (dx, dy) [px] such that b(x, y) ~ a(x - dx, y - dy): least squares over the interior (margin px excluded) of the
    cubic-spline shifted `a` (independent registration of the two images, no correlation peak or centroid).
    """
    from scipy import ndimage, optimize

    def cost(p):
        sh = ndimage.shift(a, (p[1], p[0]), order=3, mode='nearest')
        return np.sum((sh[margin:-margin, margin:-margin] - b[margin:-margin, margin:-margin]) ** 2)

    r = optimize.minimize(cost, [0.0, 0.0], method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-18 * np.sum(a ** 2)))
    return r.x


def measure(wfs, tel, src, ref, phase):
    pr, rs = ref
    s, _, _, _ = wfs.wfs_measure(phase, src, pr, rs)
    n = wfs.nValidSubaperture
    return s[:n], s[n:]


def residual_phase(rms=0.3, seed=7):
    """Fixed residual phase [rad] (high-order, f^-11/6 amplitude above 2 cycles per subaperture), so that every subaperture image
    differs slightly from the others, as in closed loop: the pseudo-reference is then really another subaperture."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(RES) * RES / NSUB                   # cycles per subaperture
    F = np.hypot(*np.meshgrid(f, f))
    amp = np.where(F > 2.0, np.maximum(F, 1e-9) ** (-11 / 6), 0.0)
    ph = np.real(np.fft.ifft2(amp * (rng.standard_normal((RES, RES)) + 1j * rng.standard_normal((RES, RES)))))
    return ph * rms / ph.std()


def test_global_tilt_linearity_and_cross_coupling(logger, report, tolerances):
    """
    Linearity over the full measuring range and cross-coupling, with the owner's estimator (use_brightest from the yaml, 9).
    A fixed residual phase (0.3 rad rms, high order) makes the subaperture images differ, so the pseudo-reference (subaperture
    pseudoref_index, measured with that phase, as the reference slopes) is a different subaperture from the ones evaluated; the
    pseudo-reference subaperture itself is excluded. A global tilt moves the object by s px along x or y, for s in
    [-R, R], R = range_arcsec / plate_scale (4.0" = 10 px, 80 % of the half window, for the 10" = 25 px window). Criteria on the mean over the evaluated
    subapertures (full scale R); the worst subaperture is recorded for information:
      linearity       a line g s + b is fitted to the mean slopes (on-axis component) over [-R, R]; the RMS of the residuals
                      about that line <= lin_frac * R (2 % of full scale). The gain g (absorbed by the interaction matrix) is
                      recorded separately, as the deviation from the ideal unit-gain line.
      cross-coupling  RMS over the evaluated subapertures of the slopes on the other axis (their mean is ~0, so the RMS
                      measures the per-subaperture leak) <= cross_frac * R at every s   (3 % of full scale)
    The images themselves are checked with an independent least-squares registration (so a failure is the estimator's).
    """
    tol = tolerances['corrsh']['linearity']
    wfs, tel, src = make_csh(logger, use_brightest=tol['use_brightest'])
    base = residual_phase(tol['residual_rms_rad'])
    _, ref_maps, _, pseudoref = wfs.wfs_measure(base, src)
    ref = (pseudoref, ref_maps)
    R = tol['range_arcsec'] / wfs.plate_scale
    shifts = np.linspace(-R, R, tol['n_points'])
    keep = np.ones(wfs.nValidSubaperture, dtype=bool)
    keep[wfs.pseudoref_index] = False
    report.metric('full scale R [px]', R)
    report.metric('evaluated subapertures (pseudo-reference excluded)', int(keep.sum()))
    # independent check of the images (x tilt, first evaluated subaperture)
    k0 = int(np.flatnonzero(keep)[0])
    img0 = wfs.get_subaps(wfs.wfs_measure(base, src, *ref)[2])[k0].numpy()
    s_reg = shifts[np.abs(shifts) <= 3.0][::2]                # within the search range of register_shift
    reg = np.array([register_shift(img0, wfs.get_subaps(wfs.wfs_measure(base + tilt_phase(s), src, *ref)[2])[k0].numpy())
                    for s in s_reg])
    report.metric('independent registration of the images (|shift| <= 3 px): max |x shift - true| [px]', np.abs(reg[:, 0] - s_reg).max())
    results = {}
    for axis, (i_on, i_off) in (('x', (0, 1)), ('y', (1, 0))):
        meas = np.array([measure(wfs, tel, src, ref, base + tilt_phase(s, axis)) for s in shifts])   # [shift, 2, nValid]
        on, off = meas[:, i_on][:, keep], meas[:, i_off][:, keep]
        err = on - shifts[:, None]
        results[axis] = (on, off)
        # linearity: line fitted to the mean slopes over [-R, R]; RMS of the residuals about that line, in units of R
        mean_on = on.mean(1)
        g, b0 = np.polyfit(shifts, mean_on, 1)
        resid = mean_on - (g * shifts + b0)
        report.check(f'{axis} tilt: RMS of the mean slopes about the fitted line / R (linearity)',
                     np.sqrt(np.mean(resid ** 2)) / R, lo=0, hi=tol['lin_frac'])
        report.metric(f'{axis} tilt: fitted gain g (mean slopes)', g)
        report.metric(f'{axis} tilt: fitted offset b [px]', b0)
        report.metric(f'{axis} tilt: max |mean slope - shift| / R (deviation from the ideal line, information)', np.abs(err.mean(1)).max() / R)
        rms_off = np.sqrt(np.mean(off ** 2, axis=1))
        report.check(f'{axis} tilt: max over shifts of the RMS over subapertures of the cross slopes / R (cross-coupling)',
                     rms_off.max() / R, lo=0, hi=tol['cross_frac'])
        report.metric(f'{axis} tilt: shift of the largest cross-slope RMS [px]', shifts[np.argmax(rms_off)])
        report.metric(f'{axis} tilt: max over shifts of |mean cross slope| / R (information)', np.abs(off.mean(1)).max() / R)
        report.metric(f'{axis} tilt: worst subaperture |slope - shift| / R (information)', np.abs(err).max() / R)
        report.metric(f'{axis} tilt: worst subaperture |cross slope| / R (information)', np.abs(off).max() / R)
        report.metric(f'{axis} tilt: max |slope - shift| [px]', np.abs(err).max())
        report.metric(f'{axis} tilt: max |cross slope| [px]', np.abs(off).max())
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for c, axis in enumerate(('x', 'y')):
        on, off = results[axis]
        g_c, b_c = np.polyfit(shifts, on.mean(1), 1)
        ax[0, c].plot(shifts, on.mean(1) - (g_c * shifts + b_c), 'o-', ms=2.5, color='#1baf7a', label=f'mean - fitted line (g = {g_c:.3f})')
        ax[0, c].plot(shifts, on.mean(1) - shifts, '-', color='#2a78d6', label='mean - ideal line')
        ax[0, c].fill_between(shifts, (on - shifts[:, None]).min(1), (on - shifts[:, None]).max(1), color='#2a78d6', alpha=0.2, lw=0, label='min-max')
        ax[0, c].axhspan(-tol['lin_frac'] * R, tol['lin_frac'] * R, color='0.9', zorder=0, label='2 % of full scale (RMS limit)')
        ax[1, c].plot(shifts, np.sqrt(np.mean(off ** 2, axis=1)), 'o-', ms=2.5, color='#d55181', label='RMS over subapertures')
        ax[1, c].plot(shifts, off.mean(1), ':', color='#d55181', lw=1, label='mean (information)')
        ax[1, c].axhspan(-tol['cross_frac'] * R, tol['cross_frac'] * R, color='0.9', zorder=0, label='3 % of full scale (limit)')
        ax[0, c].set_title(f'tilt along {axis}, use_brightest {tol["use_brightest"]}', fontsize=9)
        ax[0, c].set_ylabel('slope - true shift [px]'); ax[1, c].set_ylabel('cross slope [px]')
        if c == 0:
            ax[1, c].legend(fontsize=7, frameon=False)
        ax[1, c].set_xlabel('true shift [px]')
        for r in (0, 1):
            ax[r, c].axhline(0, color='k', lw=0.6); ax[r, c].grid(True, color='0.92')
    ax[0, 0].legend(fontsize=7, frameon=False)
    report.figure(fig, 'linearity_cross_coupling')
    report.assert_ok()


def test_reference_averaged(logger, report, tolerances):
    """
    With noiseFlag=True the pseudo-reference is the average of n_reference_frames images of subaperture pseudoref_index and the
    reference slopes the average of n further frames correlated with it, as in operation. sigma_1 is the single-frame noise rms
    of the slopes with that fixed pseudo-reference (30 further frames). The reference keeps the noise of the n averaged frames,
    sigma_1^2 / n, plus the effect of the residual noise of the averaged pseudo-reference (independent, single-image noise / n):
    between 0 (pseudo-reference noise ignored by the estimator) and sigma_1^2 / n (linear propagation, symmetric in the two
    images). Hence, for the subapertures other than the pseudo-reference, sqrt(n) * rms(noisy - noise-free reference) / sigma_1
    lies in [1, sqrt(2)], widened by the estimator statistics (n_sigma / sqrt(2 M), M slopes).
    """
    tol = tolerances['corrsh']['reference']
    det = dict(noiseFlag=True, quantumEfficiency=1.0, shotNoise=1, quantization_conversion=1, fullWellCapacity=1e12,
               nBits=32, darkCalibration=0, readoutNoise=0, darkCurrent=0, randomState=5)
    noisy, tel, src = make_csh(logger, use_brightest=9, n_reference_frames=tol['n_frames'], **det)
    clean, _, _ = make_csh(logger, use_brightest=9)
    pr_n, ref_n = noisy.initialize_wfs(tel, src)
    pr_c, ref_c = clean.initialize_wfs(tel, src)
    null = np.zeros((src.nSubDirs ** 2, RES, RES))
    frames = np.array([noisy.wfs_measure(null, src, pr_n, ref_n)[0] for _ in range(30)])
    keep = np.ones(noisy.nValidSubaperture, dtype=bool)
    keep[noisy.pseudoref_index] = False
    keep = np.r_[keep, keep]
    noise_rms = np.sqrt(frames[:, keep].var(0, ddof=1).mean())
    diff = (ref_n - ref_c)[noisy.valid_slopes_maps][keep]
    band = tol['n_sigma'] / np.sqrt(2 * diff.size)
    report.metric('single-frame noise rms', noise_rms, unit='px')
    report.metric('averaged frames n', tol['n_frames'])
    pn, pc = pr_n / pr_n.mean(), pr_c / pr_c.mean()
    report.metric('pseudo-reference (each normalised to unit mean): rms(noisy - noise-free)', float(((pn - pc) ** 2).mean().sqrt()))
    report.check('sqrt(n) * rms(reference noisy - noise-free) / single-frame noise rms',
                 np.sqrt(tol['n_frames'] * np.mean(diff ** 2)) / noise_rms, lo=1 - band, hi=np.sqrt(2) + band)
    report.assert_ok()


def test_tilt_axes_isotropy_and_sign(logger, report, tolerances):
    """
    (1) Sign: a tilt that moves the image towards +x (+y) gives positive x (y) slopes, as ShackHartmann does for the same phase
    (point source, same telescope and plate scale). (2) Isotropy: the sun patches are transposed (rows <-> columns, and the
    sub-directions likewise); the response to a y tilt of the transposed object must equal the x response of the original one
    (the algorithm must treat rows and columns alike) and the cross responses likewise. Shift 0.5 px.
    """
    tol = tolerances['corrsh']['isotropy']
    wfs, tel, src = make_csh(logger)
    ref = wfs.initialize_wfs(tel, src)
    sx_x, sy_x = measure(wfs, tel, src, ref, tilt_phase(0.5, 'x'))
    sx_y, sy_y = measure(wfs, tel, src, ref, tilt_phase(0.5, 'y'))
    sh, tel_sh, src_sh = make_sh(logger, samp=SAMP, fov_px=25, use_brightest=None)
    n = sh.nValidSubaperture
    s_sh_x = sh.wfs_measure(tilt_phase(0.5, 'x'), src_sh)[0]
    s_sh_y = sh.wfs_measure(tilt_phase(0.5, 'y'), src_sh)[0]
    report.metric('SH reference: slopes for the x tilt (x, y mean)', [s_sh_x[:n].mean(), s_sh_x[n:].mean()], unit='px')
    report.check('x tilt: sign of the x slope (CSH vs SH)', np.sign(sx_x.mean()), np.sign(s_sh_x[:n].mean()), atol=0)
    report.check('y tilt: sign of the y slope (CSH vs SH)', np.sign(sy_y.mean()), np.sign(s_sh_y[n:].mean()), atol=0)
    report.check('sign of the SH x slope', np.sign(s_sh_x[:n].mean()), 1.0, atol=0)
    srcT = copy.copy(src)
    srcT.subDirs_sun = np.ascontiguousarray(src.subDirs_sun.transpose(1, 0, 3, 2))
    srcT.filter_2D = np.ascontiguousarray(src.filter_2D.transpose(1, 0, 3, 2))
    refT = wfs.initialize_wfs(tel, srcT)
    tx_y, ty_y = measure(wfs, tel, srcT, refT, tilt_phase(0.5, 'y'))
    report.check('transposed object, y tilt: y slope - original x slope (max)', np.abs(ty_y - sx_x).max(), 0.0, atol=tol['atol'], unit='px')
    report.check('transposed object, y tilt: x slope - original y slope (max)', np.abs(tx_y - sy_x).max(), 0.0, atol=tol['atol'], unit='px')
    report.assert_ok()


def test_unit_in_rad_constructs(logger, report):
    """unit_in_rad=True must construct and give a finite, positive px/rad conversion (slopes_units)."""
    wfs, tel, src = make_csh(logger, unit_in_rad=True)
    report.check('slopes_units finite and positive', float(np.isfinite(wfs.slopes_units) and wfs.slopes_units > 0), 1.0, atol=0)
    report.assert_ok()


def test_unit_in_rad_reference_slopes(logger, report):
    """
    With unit_in_rad=True the reference slopes of initialize_wfs are in rad, as the measurements: a null phase measured against
    them gives zero slopes, and a 0.5 px tilt gives the px/rad ratio of the same tilt measured in px (unit_in_rad=False).
    """
    wfs, tel, src = make_csh(logger, unit_in_rad=True)
    wfs_px, _, _ = make_csh(logger)
    ref, ref_px = wfs.initialize_wfs(tel, src), wfs_px.initialize_wfs(tel, src)
    sx0, sy0 = measure(wfs, tel, src, ref, np.zeros((RES, RES)))
    report.check('null phase: max |slope| [rad]', max(np.abs(sx0).max(), np.abs(sy0).max()), 0.0, atol=1e-9)
    sx, _ = measure(wfs, tel, src, ref, tilt_phase(0.5, 'x'))
    sx_px, _ = measure(wfs_px, tel, src, ref_px, tilt_phase(0.5, 'x'))
    report.check('0.5 px tilt: slope [rad] x slopes_units / slope [px]', sx.mean() * wfs.slopes_units / sx_px.mean(), 1.0, rtol=1e-6)
    report.assert_ok()


def test_photons_scale_with_fov_squared(logger, report, tolerances):
    """
    The photons collected scale with the sky area observed: (fieldOfView / src.fov)^2 times the flux of the source (src.fov
    is the field the flux was computed for). Noise-free-count detector (no shot noise, QE 1, no readout/dark noise, 1 e-/DN): the
    electrons of the frame are the photons. Absolute: flux x area x time x lightRatio x (fov/src.fov)^2 with the area of the
    pupil (D^2) and the sampling time; relative: ratio of the frames of FoV 6" and 10".
    """
    tol = tolerances['corrsh']['photons']
    tot = {}
    for fov in (6.0, 10.0):
        wfs, tel, src = make_csh(logger, fov=fov, **DET_EXACT)
        frame = wfs.wfs_measure(np.zeros((RES, RES)), src)[2]
        tot[fov] = frame.sum()
    expected = {f: make_source(logger).flux * D * D * 1e-3 * (f / 10.0) ** 2 for f in tot}
    for f in tot:
        report.check(f'electrons / expected, FoV {f:g}"', tot[f] / expected[f], 1.0, atol=tol['rtol'])
    report.check('electrons(6") / electrons(10") (expected 0.36)', tot[6.0] / tot[10.0], 0.36, atol=tol['rtol'])
    report.assert_ok()
