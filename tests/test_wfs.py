"""
T1 validation of SAOS.ShackHartmann (point source): slope scale, axes and cross terms, linearity, symmetry, spot size,
centroid noise, reference slopes, partial subapertures and parameter validation.

Configuration: 6x6 subapertures of d = 0.1 m over a 48 px pupil (8 phase px per subaperture), source band R (640 nm),
2 px per lambda/d (Nyquist) unless stated. wfs_measure is called with synthetic phases; slopes are in px. A phase
gradient g [rad/px] along the columns tilts the beam by lambda g / (2 pi p) and moves the spot by
k g pixels, k = lambda / (2 pi p) * 206265 / plate_scale.

Independent reference (no SAOS code, no SAOS-measured quantity): the far field of a subaperture is propagated
here with a direct Fourier sum of the continuous phase over the physical subaperture (side d, N_ref samples of equal
cells, converged: see test_reference_converged), evaluated at the detector pixel centres u = (m - (n-1)/2)/samp
[lambda/d]; the centroid is the centre of gravity of the n x n window, and the slope is its displacement from the window
centre, where an ideal sensor puts the zero-tilt spot. The centroid algorithm (window truncation) is part of the
reference, the scale of the spot motion (k_px, from lambda, p and the plate scale) and the position of the window are not.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from citools import known_issue
from SAOS.ShackHartmann import ShackHartmann
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

RES, NSUB, D = 48, 6, 0.6
NP = RES // NSUB                       # phase px per subaperture
P = D / RES                            # phase pixel [m]
SUBAP = D / NSUB
COORD = np.arange(RES) - (RES - 1) / 2
X, Y = np.meshgrid(COORD, COORD)       # X: column coordinate [px], Y: row coordinate [px]


def make_wfs(logger, pupil='square', samp=2.0, fov_px=15, use_brightest=None, **kw):
    """WFS with `samp` px per lambda/d and a window of fov_px px; use_brightest=None -> all pixels of the window."""
    pup = np.ones((RES, RES), dtype=bool) if pupil == 'square' else None
    tel = Telescope(resolution=RES, diameter=D, samplingTime=1e-3, pupil=pup,
                    centralObstruction=0.2 if pupil == 'obstructed' else 0.0, logger=logger)
    src = Source('R', kw.pop('magnitude', 0), logger=logger)
    ps = src.wavelength / SUBAP * 206265 / samp
    wfs = ShackHartmann(nSubap=NSUB, telescope=tel, src=src, lightRatio=kw.pop('lightRatio', 1.0), plate_scale=ps,
                        fieldOfView=fov_px * ps, guardPx=1, use_brightest=fov_px ** 2 if use_brightest is None else use_brightest,
                        logger=logger, device='cpu', **kw)
    wfs.k_px = src.wavelength / (2 * np.pi * P) * 206265 / ps      # spot shift [px] per phase gradient [rad/px]
    wfs.samp = samp
    return wfs, tel, src


def split_xy(wfs, s):
    n = wfs.nValidSubaperture
    return s[:n], s[n:]


def propagate_cog(phase_fn, n, samp, M=192, amp=None):
    """
    Ideal centroid [px] (x, y; relative to the window centre) of the n x n window of the far field of one subaperture.
    phase_fn(xi, eta): phase [rad] at the normalised subaperture coordinates xi (columns), eta (rows) in [-1/2, 1/2]
    (units of d, cell centres of an M x M grid); amp: optional M x M amplitude. Window pixel m sits at
    u = (m - (n-1)/2)/samp [lambda/d]. Returns the centroid and the window energy.
    """
    s = (np.arange(M) + 0.5) / M - 0.5
    XI, ET = np.meshgrid(s, s)
    field = np.exp(1j * phase_fn(XI, ET)) if amp is None else amp * np.exp(1j * phase_fn(XI, ET))
    u = (np.arange(n) - (n - 1) / 2) / samp
    Ex = np.exp(-2j * np.pi * np.outer(u, s))
    img = np.abs(Ex @ field @ Ex.T / M ** 2) ** 2           # [row = y, column = x]
    idx = np.arange(n) - (n - 1) / 2
    tot = img.sum()
    return ((img.sum(0) * idx).sum() / tot, (img.sum(1) * idx).sum() / tot), tot


def tilt_cog(wfs, shift_px, M=192):
    """Ideal x centroid [px] of a square subaperture for a spot motion `shift_px` (phase gradient shift_px / k_px per px)."""
    g = shift_px / wfs.k_px
    return propagate_cog(lambda xi, et: g * NP * xi, wfs.npix_subap, wfs.samp, M)[0][0]


def tilt_response(wfs, src, shift_px, axis='x'):
    g = shift_px / wfs.k_px
    s, _, _ = wfs.wfs_measure(g * (X if axis == 'x' else Y), src)
    return split_xy(wfs, s)


# ----------------------------------------------------------------------------------------------------------------------
# C1: tip/tilt, axes, cross terms, scale
# ----------------------------------------------------------------------------------------------------------------------

def test_tilt_axes_isotropy_cross_term(logger, report, tolerances):
    """
    Tilt along x (columns) moves every spot along x only, by the same amount in all subapertures; tilt along y gives
    the same response on the y slopes (isotropy); a tilt at 30 deg gives slopes at 30 deg. Sign: a positive phase
    gradient along +x gives a positive x slope. Square pupil, shift 0.3 px.
    """
    tol = tolerances['wfs']['tilt_axes']
    wfs, tel, src = make_wfs(logger)
    sxx, sxy = tilt_response(wfs, src, 0.3, 'x')
    syx, syy = tilt_response(wfs, src, 0.3, 'y')
    report.check('x tilt: mean x slope sign', np.sign(sxx.mean()), 1.0, atol=0)
    report.check('x tilt: |y slope| / x slope (cross term)', np.abs(sxy).max() / sxx.mean(), hi=tol['cross_max'])
    report.check('y tilt: |x slope| / y slope (cross term)', np.abs(syx).max() / syy.mean(), hi=tol['cross_max'])
    report.check('isotropy: x slope(x tilt) / y slope(y tilt)', sxx.mean() / syy.mean(), 1.0, atol=tol['isotropy'])
    report.check('uniformity: std/mean of x slopes', sxx.std() / sxx.mean(), hi=tol['uniformity'])
    ang = np.deg2rad(30)
    g = 0.3 / wfs.k_px
    s, s2d, _ = wfs.wfs_measure(g * (np.cos(ang) * X + np.sin(ang) * Y), src)
    sx, sy = split_xy(wfs, s)
    report.check('30 deg tilt: slope angle', np.rad2deg(np.arctan2(sy.mean(), sx.mean())), 30.0, atol=tol['angle_deg'], unit='deg')
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    for a, m, t in zip(ax, (s2d[:NSUB], s2d[NSUB:]), ('x slopes', 'y slopes')):
        im = a.imshow(m, origin='lower', cmap='RdBu_r', vmin=-0.4, vmax=0.4)
        a.set_title(f'tilt 0.3 px at 30 deg: {t}', fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.046)
    report.figure(fig, 'slopes_tilt30')
    report.assert_ok()


@pytest.mark.parametrize('fov_px', [15, 30])
@known_issue('WFS-SCALE', 'subaperture phase interpolated with align_corners=True: N samples span d, the FFT pixel is '
             '(N-1)/N of the plate scale and slopes are N/(N-1) too large (+7 % at N=15, +3.4 % at N=30)',
             'ShackHartmann.py:423-433, 458')
def test_tilt_scale_vs_exact_geometry(logger, report, tolerances, fov_px):
    """
    Small tilt (0.05 px) on a square pupil: the SAOS slope must equal the ideal centroid of the window (propagate_cog)
    for the nominal spot motion k_px g: the angular scale of the sensor in px per rad. Ratio 1 within scale_atol.
    """
    wfs, tel, src = make_wfs(logger, fov_px=fov_px)
    sxx, _ = tilt_response(wfs, src, 0.05, 'x')
    N = wfs.npix_lenslet
    report.metric('N (samples across the subaperture)', N)
    report.metric('N/(N-1)', N / (N - 1))
    report.check('slope / ideal centroid', sxx.mean() / tilt_cog(wfs, 0.05), 1.0, atol=tolerances['wfs']['scale_atol'], trend=True)
    report.assert_ok()


def test_reference_converged(report, tolerances):
    """
    Convergence of the reference itself (no SAOS): the centroid of the ideal sensor computed with N equal cells across the
    subaperture approaches the continuum (M = 1024 cells) as 1/N^2; at the number of samples used by the tests (N = 31, 
    window 31 px) the relative difference must stay below the reference tolerance of the absolute-slope tests.
    """
    tol = tolerances['wfs']['absolute']
    n, samp = 31, 2.0
    wfs = type('W', (), dict(k_px=NP * samp / (2 * np.pi), npix_subap=n, samp=samp))
    worst = 0.0
    for sh in (0.05, 0.5, 2.0, 5.0):
        cont = tilt_cog(wfs, sh, M=1024)
        conv = [tilt_cog(wfs, sh, M=m) / cont - 1 for m in (n, 2 * n, 4 * n)]
        report.metric(f'shift {sh} px: relative error of N, 2N, 4N cells', conv)
        worst = max(worst, abs(conv[0]))
    report.check('max relative error with N cells', worst, hi=tol['rtol'])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for sh in (0.5, 5.0):
        ms = np.array([8, 16, 31, 62, 124, 248])
        cont = tilt_cog(wfs, sh, M=1024)
        ax.loglog(ms, [abs(tilt_cog(wfs, sh, M=m) / cont - 1) for m in ms], 'o-', label=f'{sh} px')
    ax.set_xlabel('cells across the subaperture'); ax.set_ylabel('|ref/continuum - 1|')
    ax.legend(fontsize=7, frameon=False); ax.grid(True, which='both', color='0.92')
    report.figure(fig, 'reference_convergence')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C3/C4: linearity, odd symmetry, continuity at 0
# ----------------------------------------------------------------------------------------------------------------------

@known_issue('WFS-SCALE', 'slopes are N/(N-1) too large (align_corners=True interpolation of the subaperture phase) and the '
             'zero-tilt spot is off the window centre (WFS-SPOT-OFFCENTRE), which makes the response asymmetric',
             'ShackHartmann.py:423-433, 452-458')
def test_linearity_sweep(logger, report, tolerances):
    """
    Absolute response s(theta) for spot motions of 0.01 ... 6 px (both signs) on a square pupil, 31 px window, compared
    with the ideal centroid of the window (propagate_cog) for the nominal motion k_px g: |s - ref| <= atol + rtol |ref|
    at every shift (no normalisation by the SAOS small-signal gain). Diagnostic only (not asserted): the gain curve
    normalised by its small-signal value, SAOS vs ideal.
    """
    tol = tolerances['wfs']['absolute']
    wfs, tel, src = make_wfs(logger, fov_px=31)
    shifts = np.geomspace(0.01, 6.0, 18)
    curves = {}
    for sign in (+1, -1):
        s_saos = np.array([tilt_response(wfs, src, sign * s, 'x')[0].mean() for s in shifts])
        ref = np.array([tilt_cog(wfs, sign * s) for s in shifts])
        curves[sign] = (s_saos, ref)
        lab = '+' if sign > 0 else '-'
        report.check(f'{lab} shifts: |s - ref| / (atol + rtol |ref|) (max, must be <= 1)',
                     np.max(np.abs(s_saos - ref) / (tol['atol'] + tol['rtol'] * np.abs(ref))), hi=1.0)
        report.metric(f'{lab} shifts: max |s/ref - 1|', np.max(np.abs(s_saos / ref - 1)))
        report.metric(f'{lab} shifts: diagnostic max |normalised SAOS gain / normalised ideal gain - 1|',
                      np.max(np.abs((s_saos / s_saos[0]) / (ref / ref[0]) - 1)))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for sign, c in ((+1, '#2a78d6'), (-1, '#d55181')):
        s_saos, ref = curves[sign]
        ax.semilogx(shifts, s_saos / (sign * shifts), 'o-', color=c, ms=3, label=f'SAOS, {"+" if sign > 0 else "-"} shift')
        ax.semilogx(shifts, ref / (sign * shifts), '--', color=c, label=f'ideal, {"+" if sign > 0 else "-"} shift')
    ax.axhline(1, color='k', lw=0.5)
    ax.set_xlabel('nominal spot shift [px]'); ax.set_ylabel('slope / nominal shift')
    ax.set_title(f'SH response, window {wfs.npix_subap} px, {wfs.samp:g} px per lambda/d', fontsize=8)
    ax.legend(fontsize=7, frameon=False); ax.grid(True, which='both', color='0.92')
    report.figure(fig, 'linearity')
    report.assert_ok()


@pytest.mark.parametrize('fov_px', [15, 16])
@known_issue('WFS-SPOT-OFFCENTRE', 'the zero-tilt spot is not at the window centre (+1 px with an odd window, +0.5 px '
             'with an even one), so the dynamic range is asymmetric', 'ShackHartmann.py:452-456, 487')
def test_spot_centred_in_window(logger, report, tolerances, fov_px):
    """The unaberrated spot must sit at the centre of its subaperture window: |CoG - (n-1)/2| <= 0.05 px."""
    wfs, tel, src = make_wfs(logger, fov_px=fov_px)
    img = wfs.get_subaps(wfs.wfs_measure(np.zeros((RES, RES)), src)[2])[0]
    r, c = np.indices(img.shape)
    cx = (c * img).sum() / img.sum()
    report.check('spot CoG - window centre (x)', cx - (wfs.npix_subap - 1) / 2, 0.0, atol=tolerances['wfs']['centring_atol'], unit='px')
    report.assert_ok()


def test_odd_symmetry_all_pixels(logger, report, tolerances):
    """
    CoG on all pixels (odd window, 15 px): s(-theta) = -s(theta) within 0.5 % of |s| up to 3 px, and the small-signal
    gain is the same at 0.002 px and 0.05 px (continuous at 0).
    """
    tol = tolerances['wfs']['symmetry']
    wfs, tel, src = make_wfs(logger, fov_px=15)
    asym = []
    for s in (0.05, 0.5, 1.5, 3.0):
        sp = tilt_response(wfs, src, s, 'x')[0].mean()
        sm = tilt_response(wfs, src, -s, 'x')[0].mean()
        asym.append((sp + sm) / (sp - sm))
    report.check('max |s(+t) + s(-t)| / |s(+t) - s(-t)|', np.abs(asym).max(), hi=tol['asym_max'])
    g_small = tilt_response(wfs, src, 0.002, 'x')[0].mean() / 0.002
    g_ref = tilt_response(wfs, src, 0.05, 'x')[0].mean() / 0.05
    report.check('gain(0.002 px) / gain(0.05 px)', g_small / g_ref, 1.0, atol=tol['continuity_atol'])
    report.assert_ok()


def test_continuity_default_brightest(logger, report, tolerances):
    """
    Default use_brightest = 50: the small-signal response must still be linear and odd, gain(+-0.002 px) equal to
    gain(+-0.05 px) within 2 %.
    """
    tol = tolerances['wfs']['symmetry']
    wfs, tel, src = make_wfs(logger, fov_px=15, use_brightest=50)
    g = {s: tilt_response(wfs, src, s, 'x')[0].mean() / s for s in (0.002, -0.002, 0.05, -0.05)}
    report.check('gain(+0.002) / gain(+0.05)', g[0.002] / g[0.05], 1.0, atol=tol['brightest_atol'])
    report.check('gain(-0.002) / gain(+0.05)', g[-0.002] / g[0.05], 1.0, atol=tol['brightest_atol'])
    report.check('gain(-0.05) / gain(+0.05)', g[-0.05] / g[0.05], 1.0, atol=tol['brightest_atol'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C2: defocus and astigmatism
# ----------------------------------------------------------------------------------------------------------------------

def test_defocus_astigmatism_relations(logger, report, tolerances):
    """
    Scale-free algebra of the linear response to quadratic phases (square pupil, c = 5e-4 rad/px^2: spot motions below
    0.1 px, far inside the linear range of the centroid): with phi = c (x^2 + y^2) the x slope depends only on the
    column and equals the y slope of the transposed subaperture; astigmatism phi = c x y gives the cross term
    s_x(astig) = s_y(defocus)/2 and s_y(astig) = s_x(defocus)/2; phi = c (x^2 - y^2) gives s_x = s_x(defocus),
    s_y = -s_y(defocus). The identities hold for any common scale factor, so they say nothing about the absolute
    response: that is tested for 0.002 <= c <= 0.03 in test_quadratic_phase_absolute_slopes. Beyond the linear
    range the identities are not exact (even part of the response), which is why the amplitude is small here.
    """
    tol = tolerances['wfs']['quadratic']
    wfs, tel, src = make_wfs(logger)
    c = 0.0005
    _, dfc, _ = wfs.wfs_measure(c * (X ** 2 + Y ** 2), src)
    _, ast, _ = wfs.wfs_measure(c * X * Y, src)
    _, z6, _ = wfs.wfs_measure(c * (X ** 2 - Y ** 2), src)
    dx, dy, ax_, ay, z6x, z6y = dfc[:NSUB], dfc[NSUB:], ast[:NSUB], ast[NSUB:], z6[:NSUB], z6[NSUB:]
    scale = np.abs(dx).max()
    report.check('defocus: x slopes independent of row (max |s - row mean| / max|s|)', np.abs(dx - dx.mean(0)).max() / scale, hi=tol['rel_max'])
    report.check('defocus: s_x[i,j] - s_y[j,i] (transpose symmetry)', np.abs(dx - dy.T).max() / scale, hi=tol['rel_max'])
    report.check('astig xy: s_x - s_y(defocus)/2', np.abs(ax_ - dy / 2).max() / scale, hi=tol['rel_max'])
    report.check('astig xy: s_y - s_x(defocus)/2', np.abs(ay - dx / 2).max() / scale, hi=tol['rel_max'])
    report.check('x^2-y^2: s_x - s_x(defocus)', np.abs(z6x - dx).max() / scale, hi=tol['rel_max'])
    report.check('x^2-y^2: s_y + s_y(defocus)', np.abs(z6y + dy).max() / scale, hi=tol['rel_max'])
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    jj, ii = np.meshgrid(np.arange(NSUB), np.arange(NSUB))
    for a, (sx, sy, t) in zip(axs, ((dx, dy, 'defocus'), (ax_, ay, 'astigmatism xy'), (z6x, z6y, 'astigmatism x^2-y^2'))):
        a.quiver(jj, ii, sx, sy, angles='xy', scale_units='xy', scale=2 * scale)
        a.set_title(t, fontsize=8); a.set_aspect('equal'); a.set_xlabel('subaperture column'); a.set_ylabel('row')
    report.figure(fig, 'quadratic_slopes')
    report.assert_ok()


SUB_C = NP * np.arange(NSUB) + (NP - 1) / 2 - (RES - 1) / 2     # subaperture centres on the pupil grid [px]


def quadratic_expected(wfs, axx, ayy, axy, M=128):
    """
    Ideal slopes [px] (x map, y map: [row, column]) for the phase axx x^2 + ayy y^2 + axy x y (x, y in pupil px from the
    pupil centre): every subaperture is propagated with propagate_cog; its centre is on the pupil grid, SUB_C.
    """
    sx, sy = np.zeros((NSUB, NSUB)), np.zeros((NSUB, NSUB))
    for i in range(NSUB):
        for j in range(NSUB):
            xc, yc = SUB_C[j], SUB_C[i]
            f = lambda xi, et: (axx * (xc + NP * xi) ** 2 + ayy * (yc + NP * et) ** 2 + axy * (xc + NP * xi) * (yc + NP * et))
            (sx[i, j], sy[i, j]), _ = propagate_cog(f, wfs.npix_subap, wfs.samp, M)
    return sx, sy


@pytest.mark.parametrize('c', [0.002, 0.01, 0.03], ids=['c0.002', 'c0.01', 'c0.03'])
@pytest.mark.parametrize('mode', ['defocus', 'astig_xy', 'astig_x2y2'])
@known_issue('WFS-SCALE', 'slopes are N/(N-1) too large (align_corners=True), the subaperture grid is shifted (WFS-SUBAP-GRID) '
             'and the zero-tilt spot is off the window centre (WFS-SPOT-OFFCENTRE)', 'ShackHartmann.py:365-395, 423-433, 452-458')
def test_quadratic_phase_absolute_slopes(logger, report, tolerances, mode, c):
    """
    Absolute slopes for quadratic phases of amplitude c [rad/px^2] (defocus c(x^2+y^2), astigmatism c x y and
    c(x^2-y^2); the largest mean gradient is 2 c 20 px = 0.08 ... 1.2 rad/px, i.e. up to ~3.7 px of spot motion): each
    subaperture is propagated independently (quadratic_expected) over the 31 px window, so the expected response includes
    the window truncation and the curvature of the phase inside the subaperture. |s - ref| <= atol + rtol |ref|.
    """
    tol = tolerances['wfs']['absolute']
    wfs, tel, src = make_wfs(logger, fov_px=31)
    axx, ayy, axy = dict(defocus=(c, c, 0), astig_xy=(0, 0, c), astig_x2y2=(c, -c, 0))[mode]
    _, s2d, _ = wfs.wfs_measure(axx * X ** 2 + ayy * Y ** 2 + axy * X * Y, src)
    rx, ry = quadratic_expected(wfs, axx, ayy, axy)
    for name, m, r in (('x', s2d[:NSUB], rx), ('y', s2d[NSUB:], ry)):
        report.metric(f'{name} slopes: max |ideal|', np.abs(r).max(), unit='px')
        report.check(f'{name} slopes: max |s - ref| / (atol + rtol |ref|) (must be <= 1)',
                     np.max(np.abs(m - r) / (tol['atol'] + tol['rtol'] * np.abs(r))), hi=1.0)
        report.metric(f'{name} slopes: max |s - ref|', np.abs(m - r).max(), unit='px')
    report.assert_ok()


@known_issue('WFS-SUBAP-GRID', 'split_subapertures samples subaperture k on phase pixels [k np, k np + np] (centre '
             'shifted +0.5 px from the pupil grid) and the last one on [res-np-1, res-1] (pitch np-1)',
             'ShackHartmann.py:365-395')
def test_defocus_subaperture_geometry(logger, report, tolerances):
    """
    Geometry of the subaperture grid, independent of the angular scale: for defocus phi = c (x^2 + y^2) the mean gradient
    over subaperture j is 2 c x_j, so the x slopes are an affine function of the subaperture centre x_j of the pupil grid
    (np j + (np - 1)/2 - (res - 1)/2 px). The line s_j = a x_j + b is fitted to the slopes (a absorbs any common scale,
    which is tested in test_quadratic_phase_absolute_slopes): the centre offset b/a must vanish and the residual of the
    fit, in px of subaperture position, must be below the geometry tolerance. Small amplitude (linear regime), both axes.
    """
    tol = tolerances['wfs']['geometry']
    wfs, tel, src = make_wfs(logger, fov_px=31)
    c = 0.002
    _, s2d, _ = wfs.wfs_measure(c * (X ** 2 + Y ** 2), src)
    for name, prof in (('x', s2d[:NSUB].mean(0)), ('y', s2d[NSUB:].mean(1))):
        a, b = np.polyfit(SUB_C, prof, 1)
        resid = np.abs(prof - (a * SUB_C + b)).max() / a
        report.check(f'{name}: centre offset b/a', b / a, 0.0, atol=tol['offset_atol'], unit='px')
        report.check(f'{name}: max fit residual / a (pitch uniformity)', resid, hi=tol['pitch_atol'], unit='px')
        report.metric(f'{name}: centres inferred s/a', prof / a, unit='px')
    report.metric('grid centres [px]', SUB_C, unit='px')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C5: spot size
# ----------------------------------------------------------------------------------------------------------------------

def test_spot_fwhm(logger, report, tolerances):
    """
    Diffraction-limited spot of a square subaperture of side d: sinc^2, FWHM = 0.8859 lambda/d along x and y (the
    1.03 lambda/D value is for a circular aperture). 4 px per lambda/d, 31 px window, flat phase.
    """
    wfs, tel, src = make_wfs(logger, samp=4.0, fov_px=31)
    img = wfs.get_subaps(wfs.wfs_measure(np.zeros((RES, RES)), src)[2])[0]
    r, c = np.unravel_index(np.argmax(img), img.shape)
    fx = _fwhm_cut(img[r, :])
    fy = _fwhm_cut(img[:, c])
    expected = 0.8859 * wfs.samp
    report.check('FWHM x / (0.886 lambda/d)', fx / expected, 1.0, atol=tolerances['wfs']['fwhm_atol'])
    report.check('FWHM y / (0.886 lambda/d)', fy / expected, 1.0, atol=tolerances['wfs']['fwhm_atol'])
    report.assert_ok()


def _fwhm_cut(profile, up=16):
    """FWHM [px] of a sampled sinc^2 cut: band-limited interpolation (zero-padded FFT) then half-maximum crossings."""
    from citools import fwhm_1d
    n = len(profile)
    F = np.fft.fft(profile)
    G = np.zeros(n * up, dtype=complex)           # zero padding of the spectrum in FFT order (positive | negative freq.)
    h = n // 2
    G[:h + 1] = F[:h + 1]
    G[-h:] = F[-h:]
    fine = np.real(np.fft.ifft(G)) * up
    return fwhm_1d(fine) / up


# ----------------------------------------------------------------------------------------------------------------------
# C6: centroid noise
# ----------------------------------------------------------------------------------------------------------------------

@known_issue('WFS-SPOT-OFFCENTRE', 'the zero-tilt spot is +1 px (odd window) off the window centre and the spot is stretched by '
             'N/(N-1) (WFS-SCALE), so the window second moment M2 is 7 % below the ideal one (1.55 vs 1.66 px^2) and the '
             'centroid noise is 7 % below M2/N', 'ShackHartmann.py:423-433, 452-458')
def test_cog_photon_noise(logger, report, tolerances):
    """
    Photon-noise-limited CoG (QE 1, 1 e-/DN, no readout/dark noise, centroid over all pixels of the window): for N
    detected electrons var(s) = M2 / N, M2 = sum (x - x_c)^2 I / sum I the second central moment of the expected spot.
    M2 is that of the ideal spot (propagate_cog's window image, nothing taken from SAOS) and N is the electron count of
    the subaperture window in the noisy frames themselves (mean over frames and subapertures), so the photometric
    calibration of the source is not involved. Three flux levels (x and y, all subapertures, 120 frames each).
    """
    tol = tolerances['wfs']['photon_noise']
    det = dict(noiseFlag=True, quantumEfficiency=1.0, shotNoise=1, darkCurrent=0, readoutNoise=0, quantization_conversion=1,
               fullWellCapacity=1e12, nBits=32, darkCalibration=0, randomState=11)
    n, samp = 15, 2.0
    # ideal spot of the full square subaperture on the n x n window
    M = 192
    sm = (np.arange(M) + 0.5) / M - 0.5
    u = (np.arange(n) - (n - 1) / 2) / samp
    Ex = np.exp(-2j * np.pi * np.outer(u, sm))
    ideal = np.abs(Ex @ np.ones((M, M)) @ Ex.T / M ** 2) ** 2
    idx = np.arange(n) - (n - 1) / 2
    px = ideal.sum(0)
    m2 = ((idx - (idx * px).sum() / px.sum()) ** 2 * px).sum() / px.sum()
    rows = []
    for mag in (2.0, 1.0, 0.0):          # per-pixel photon rounding negligible at these fluxes
        wfs, tel, src = make_wfs(logger, magnitude=mag, **det)
        s, tot = [], []
        for _ in range(120):
            sl, _, frame = wfs.wfs_measure(np.zeros((RES, RES)), src)
            s.append(sl)
            tot.append(wfs.get_subaps(frame).sum(axis=(1, 2)))
        s, n_e = np.array(s), np.mean(tot)
        var = (s - s.mean(0)).var(axis=0).mean() * 120 / 119
        rows.append((n_e, var, m2 / n_e))
        report.check(f'N_e = {n_e:.0f}: var / (M2/N_e)', var / (m2 / n_e), 1.0, atol=tol['ratio_atol'], trend=mag == 1.0)
    n_e, var, th = map(np.array, zip(*rows))
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.loglog(n_e, th, 'k-', label='M2 / N_e')
    ax.loglog(n_e, var, 'o', color='#2a78d6', label='SAOS')
    ax.set_xlabel('electrons per subaperture'); ax.set_ylabel('slope variance [px$^2$]')
    ax.legend(fontsize=7, frameon=False); ax.grid(True, which='both', color='0.92')
    report.figure(fig, 'photon_noise')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C7: reference slopes
# ----------------------------------------------------------------------------------------------------------------------

def test_reference_slopes_averaged(logger, report, tolerances):
    """
    With noiseFlag=True the reference slopes are the average of n_reference_frames detector frames, as in operation. The noise
    of the frames is independent, so rms(noisy - noise-free reference) = sigma_1 / sqrt(n), sigma_1 the single-frame noise rms
    (measured on 30 further frames). Checked as sqrt(n) * ratio = 1 within the estimator statistics (n_sigma / sqrt(2 M), M slopes).
    """
    tol = tolerances['wfs']['reference']
    det = dict(noiseFlag=True, quantumEfficiency=1.0, shotNoise=1, quantization_conversion=1, fullWellCapacity=1e12,
               nBits=32, darkCalibration=0, readoutNoise=0, darkCurrent=0, randomState=5)
    noisy, tel, src = make_wfs(logger, magnitude=8.0, n_reference_frames=tol['n_frames'], **det)
    clean, _, _ = make_wfs(logger, magnitude=8.0)
    frames = np.array([noisy.wfs_measure(np.zeros((RES, RES)), src)[0] for _ in range(30)])
    noise_rms = np.sqrt(frames.var(0, ddof=1).mean())
    diff = (noisy.reference_slopes_maps - clean.reference_slopes_maps)[noisy.valid_slopes_maps]
    m = diff.size
    band = tol['n_sigma'] / np.sqrt(2 * m)
    report.metric('single-frame noise rms', noise_rms, unit='px')
    report.metric('averaged frames n', tol['n_frames'])
    report.check('sqrt(n) * rms(reference noisy - noise-free) / single-frame noise rms', np.sqrt(tol['n_frames'] * np.mean(diff ** 2)) / noise_rms,
                 lo=1 - band, hi=1 + band)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C8: partial subapertures
# ----------------------------------------------------------------------------------------------------------------------

PUP_RES = (48, 96, 192)
OBS = 0.2


def annulus_mask(res):
    """Binary pupil (pixel centres inside the annulus 0.2 D/2 <= r < D/2) used as the aperture contract of the energy test."""
    c = np.arange(res) - (res - 1) / 2
    r = np.hypot(*np.meshgrid(c, c)) * D / res
    return (r < D / 2) & (r >= OBS * D / 2)


def partial_energy_discrepancy(logger, res, samp=2.0, fov_px=31, M=None):
    """
    Returns (energy ratio SAOS/expected of the valid subapertures, valid map, expected valid map, fraction map).
    Contract: the pupil is the piecewise-constant binary mask given to the Telescope (every pixel a uniform cell of the
    pupil plane); the field of subaperture (i, j) is the part of that mask inside the pixels [i np, (i+1) np) x
    [j np, (j+1) np), transmitted with unit amplitude, and the energy collected by the window is the sum of |E|^2 of its
    far field (Fourier sum of the cells, propagate_cog), expressed in units of a fully illuminated subaperture seen
    through the same window.
    """
    np_ = res // NSUB
    if M is None:
        # >= 256 Fourier-sum cells per subaperture: the reference itself converged (with 2 cells per pupil pixel, 16 per
        # subaperture at 48 px, its far field aliases and the window energy is off by ~1%)
        M = int(np.ceil(256 / np_))
    m = annulus_mask(res)
    tel = Telescope(resolution=res, diameter=D, samplingTime=1e-3, pupil=m, logger=logger)
    src = Source('R', 0, logger=logger)
    ps = src.wavelength / SUBAP * 206265 / samp
    wfs = ShackHartmann(nSubap=NSUB, telescope=tel, src=src, lightRatio=1.0, plate_scale=ps, fieldOfView=fov_px * ps, guardPx=1,
                        use_brightest=fov_px ** 2, logger=logger, device='cpu')
    psf = wfs.get_psf(np.zeros((res, res)), src.wavelength * 206265 / (SUBAP * wfs.plate_scale))
    en = psf.sum(axis=(1, 2)).reshape(NSUB, NSUB)
    full = propagate_cog(lambda xi, et: 0 * xi, fov_px, samp, M=M * np_, amp=np.ones((M * np_, M * np_)))[1]
    exp = np.zeros((NSUB, NSUB))
    frac = np.zeros((NSUB, NSUB))
    for i in range(NSUB):
        for j in range(NSUB):
            a = m[i * np_:(i + 1) * np_, j * np_:(j + 1) * np_].astype(float)
            frac[i, j] = a.mean()
            exp[i, j] = propagate_cog(lambda xi, et: 0 * xi, fov_px, samp, M=M * np_, amp=np.kron(a, np.ones((M, M))))[1] / full
    sc = en[exp > 0.999].mean()                  # energy of a fully illuminated subaperture in SAOS units
    v = wfs.valid_subapertures
    return (en[v] / sc) / exp[v], v, frac >= 0.5, frac


@pytest.mark.parametrize('res', PUP_RES, ids=[f'res{r}' for r in PUP_RES])
@known_issue('WFS-SUBAP-GRID', 'split_subapertures takes the phase/pupil of subaperture k on pixels [k np, k np + np] '
             '(shifted +0.5 px, the last one on [res-np-1, res-1]), so the energy of an edge subaperture is not that of its '
             'own pixels and a symmetric pupil gives asymmetric energies; error ~ 1/np (the bilinear resampling of the '
             'binary mask, align_corners=True, also blurs its edges)', 'ShackHartmann.py:131-133, 365-395')
def test_partial_subaperture_energy(logger, report, tolerances, res):
    """
    Circular pupil with a 20 % central obstruction, flat phase, 31 px window at 2 px per lambda/d: the energy of each
    valid subaperture image must equal the energy that the pixels of the aperture contract (partial_energy_discrepancy)
    transmit through that subaperture, relative to a full one; a subaperture is valid iff the fraction of its pixels inside
    the pupil is >= validSubapThreshold (0.5). Tolerance: precision of the reference (exact Fourier sum, float32 FFT).
    Run at 8, 16 and 32 px per subaperture (test_partial_energy_converges checks the trend).
    """
    r, v, vexp, frac = partial_energy_discrepancy(logger, res)
    report.check('valid map == (fraction >= 0.5)', np.mean(v == vexp), 1.0, atol=0)
    report.metric('max |ratio - 1|', np.abs(r - 1).max())
    report.check('energy / expected transmitted energy', r, 1.0, atol=tolerances['wfs']['energy_atol'])
    report.assert_ok()


def test_partial_energy_converges(logger, report, tolerances):
    """
    The energy of partially illuminated subapertures must not depend on the pupil resolution: at 8, 16 and 32 pupil px per
    subaperture the rms error (over the valid subapertures) against the converged reference stays below the precision of
    the reference (energy_atol). The energy is a property of the aperture, not of its sampling.
    """
    errs = [np.sqrt(np.mean((partial_energy_discrepancy(logger, res)[0] - 1) ** 2)) for res in PUP_RES]
    for res, e in zip(PUP_RES, errs):
        report.check(f'rms(energy ratio - 1), pupil {res} px', e, lo=0, hi=tolerances['wfs']['energy_atol'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# C10: parameter validation
# ----------------------------------------------------------------------------------------------------------------------

def test_undersampled_spot_rejected(logger, report):
    """A plate scale coarser than lambda/d (spot < 1 px) must be rejected at construction with a ValueError."""
    with pytest.raises(ValueError):
        make_wfs(logger, samp=0.8)
    report.note('ValueError raised at construction')


@pytest.mark.parametrize('n', [0, 15 ** 2 + 1], ids=['zero', 'more_than_window'])
def test_use_brightest_validated(logger, report, n):
    """
    use_brightest outside [1, npix^2] must either raise a ValueError at construction or behave as 'all pixels'
    (same slopes as use_brightest = npix^2).
    """
    try:
        wfs, tel, src = make_wfs(logger, use_brightest=n)
    except ValueError:
        report.note('ValueError at construction')
        return
    ref, _, _ = make_wfs(logger)
    s = tilt_response(wfs, src, 0.3, 'x')[0]
    r = tilt_response(ref, src, 0.3, 'x')[0]
    report.check('max |slopes - all-pixel slopes|', np.abs(s - r).max(), 0.0, atol=1e-6, unit='px')
    report.assert_ok()


def test_unknown_kwarg_reported(logger, report, caplog):
    """A misspelt option (validSubapTreshold) must raise a TypeError or at least log a warning naming it."""
    import logging
    with caplog.at_level(logging.WARNING, logger=logger.name):
        try:
            make_wfs(logger, validSubapTreshold=0.9)
        except TypeError:
            report.note('TypeError raised')
            return
    warned = any('validSubapTreshold' in r.getMessage() for r in caplog.records)
    report.check('warning naming the unknown option', float(warned), 1.0, atol=0)
    report.assert_ok()
