"""
T2 validation of the extended source (ExtendedSource) used with ScienceCam and Atmosphere.

Scenario of the owner (mcaobenchcodes/processingScripts/generateExtendedAsterism.py and
telescopeSims/anisoplanatism/anisoplanatismPointSource.py): an h5 image of delta-function "stars" at 0.0128"/px,
written as image[x_px, y_px] (first axis = sky x = r cos(az), second axis = sky y = r sin(az)), is observed by an
EST-like telescope (D 4.149 m, central obstruction 1.3 m, 3x3 sub-directions, FoV 9.975", padding 5", margin 4") with a
ScienceCam at 0.0128"/px. Besides the 17 stars of the owner a marker star at (x, y) = (+3.0", -1.2") breaks the D4
symmetry of the pattern, so that a transposed or mirrored image is detected. The same 18 directions are observed with
point sources (NGS) for reference.

Every expectation is computed independently of SAOS (own geometry, own FFT of an analytic annular pupil, own
sampling of the atmosphere screen, own convolution) with the conventions of the camera frames: [row, col], first
axis = image row, called x by the owner's generator, pixel nPix//2 = optical axis. In the SAOS sky frame of the
atmosphere and the DMs (x = column, y = row), the image row is sky y: the NGS observing the star written at
image[x_px, y_px] is placed at azimuth 90 - phi (fixed with EXT-TRANSPOSE, 2026-10-27).

Telescope resolution: 450 px (about 2 per camera pupil sample at 898 px), so that the known resampling errors of
ScienceCam (SCI-SCALE, SCI-SR-RESAMPLE, tests/test_sciencecam.py) stay below 0.2% of the PSF peak and do not mask
the extended-source mechanics; with the 200 px of the production scripts the reference mismatch is 0.7% (flat phase)
to 2.3% (r0 0.6 m).

Tests:
 B1 no atmosphere: positions, FWHM and PSF of every star vs the PSF of the obstructed pupil;
 B2 orientation: no transposition/mirroring of the image; sub-direction -> layer footprint geometry;
 B3 ground layer: isoplanatic, every star has the PSF of the (common) screen, equal to the NGS PSF;
 B4 altitude layer: expected mosaic of per-patch PSFs (tent weights) from the actual screen; growth of the
    star-NGS difference with the distance to the patch centre;
 B5 uniform object -> uniform image, structured object vs own convolution (no seams), filter partition of unity,
    flux.
"""
import numpy as np
import pytest
import h5py
from scipy import ndimage, signal, stats

import citools
from citools import known_issue, fwhm_2d
from SAOS.Atmosphere import Atmosphere
from SAOS.ExtendedSource import ExtendedSource
from SAOS.LightPath import LightPath
from SAOS.ScienceCam import ScienceCam
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.nightly

ARCSEC = np.pi / 180 / 3600
PS = 0.0128                  # camera and object plate scale ["/px]
D, OBS = 4.149, 1.3          # EST pupil [m]
RES = 450                    # telescope resolution (see module docstring)
LAM = 500e-9
FOV, PAD, MARGIN, NSUB = 9.975, 5.0, 4.0, 3
W = 39                       # half-width of the analysis windows [px] (1.0")
R0, L0 = 0.6, 25.0
H_ALT = 15000.0              # altitude of the single turbulent layer [m]
TS = 5e-4

TEXT_OWNER = [[0, 0],
              [1.872, 0], [1.872, 90], [1.872, 180], [1.872, -90],
              [2.647, 45], [2.647, 135], [2.647, -135], [2.647, -45],
              [3.744, 0], [3.744, 90], [3.744, 180], [3.744, -90],
              [5.294, 45], [5.294, 135], [5.294, -135], [5.294, -45]]       # [r ("), azimuth (deg)] of the owner
MARKER = [3.23, -21.8]                                                       # (x, y) = (+3.0", -1.2")
STAR_POLAR = TEXT_OWNER + [MARKER]


# ----------------------------------------------------------------------------------------------------------------------
# Independent references
# ----------------------------------------------------------------------------------------------------------------------

class RefPSF:
    """
    PSF of the annular pupil (D, obstruction) with a phase map given on the telescope grid [res x res, rad], on the
    camera pixel scale `ps` exactly (N * dx = lambda / ps). Own FFT; pupil = analytic annulus with 4x4 sub-pixel occupancy;
    phase pixel centres at (k - (res-1)/2) D / res, interpolated bilinearly. Output normalised to unit energy, peak at N//2.
    """

    def __init__(self, D=D, obs=OBS, lam=LAM, ps=PS, res=RES, N=1024, ss=4):
        self.N, self.res = N, res
        dx = lam / (ps * ARCSEC) / N
        m = int(np.ceil(D / dx)) + 4
        self.m = m
        g = (np.arange(m * ss) - (m * ss - 1) / 2) * dx / ss
        X, Y = np.meshgrid(g, g)
        R = np.hypot(X, Y)
        self.occ = ((R <= D / 2) & (R >= obs / 2)).astype(float).reshape(m, ss, m, ss).mean((1, 3))
        idx = ((np.arange(m) - (m - 1) / 2) * dx) / (D / res) + (res - 1) / 2
        self.coords = np.meshgrid(idx, idx, indexing='ij')

    def psf(self, phase=None):
        f = np.zeros((self.N, self.N), dtype=complex)
        if phase is None:
            f[:self.m, :self.m] = self.occ
        else:
            ph = ndimage.map_coordinates(np.asarray(phase, dtype=float), self.coords, order=1, mode='nearest')
            f[:self.m, :self.m] = self.occ * np.exp(1j * ph)
        p = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
        return p / p.sum()

    def window(self, p, w=W, shift=(0, 0)):
        """(2w+1)^2 window of the PSF, normalised to unit sum; `shift` = (drow, dcol) of the window centre in px."""
        c = self.N // 2
        a = p[c + shift[0] - w:c + shift[0] + w + 1, c + shift[1] - w:c + shift[1] + w + 1]
        return a / a.sum()


def window(frame, row, col, w=W, norm=True):
    a = frame[row - w:row + w + 1, col - w:col + w + 1]
    return a / a.sum() if norm else a


def peak_position(img, up=8):
    """Sub-pixel peak (row, col) of a band-limited image: Fourier upsampling + parabola (px of `img`)."""
    img = np.asarray(img, dtype=float)
    n0, n1 = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    G = np.zeros((n0 * up, n1 * up), dtype=complex)
    o0, o1 = (n0 * up - n0) // 2, (n1 * up - n1) // 2
    G[o0:o0 + n0, o1:o1 + n1] = F
    u = np.real(np.fft.ifft2(np.fft.ifftshift(G))) * up ** 2
    r, c = np.unravel_index(np.argmax(u), u.shape)

    def parab(m, z, p):
        den = m - 2 * z + p
        return 0.0 if den == 0 else 0.5 * (m - p) / den

    return (r + parab(u[r - 1, c], u[r, c], u[r + 1, c])) / up, (c + parab(u[r, c - 1], u[r, c], u[r, c + 1])) / up


def patch_geometry(fov=FOV, pad=PAD, n=NSUB):
    """Patch spacing: patch size s = 2 (fov + pad) / (n + 1) ["], centres every s/2 around the axis (50% overlap)."""
    s = 2 * (fov + pad) / (n + 1)
    return s, (np.arange(n) - (n - 1) / 2) * s / 2


def patch_weights(i, x, s, centres):
    """Tent weight of patch i (centre c_i, half-width s/2 = patch spacing) at sky coordinate x ["]; flat 1 beyond the
    outermost centres, so that the weights of all patches sum to one everywhere."""
    t = np.maximum(0.0, 1 - np.abs(x - centres[i]) / (s / 2))
    if i == 0:
        t = np.where(x < centres[0], 1.0, t)
    if i == len(centres) - 1:
        t = np.where(x > centres[-1], 1.0, t)
    return t


def layer_pitch(atm, layer=1):
    return getattr(atm, f'layer_{layer}').spatial_res


def screen_window(scr, off_row, off_col, res=RES):
    """Pupil-sized window of a layer screen whose first sample is (npix//2 - res//2) + offset (px, sub-pixel by cubic spline)."""
    st = scr.shape[0] // 2 - res // 2
    rr = st + off_row + np.arange(res)
    cc = st + off_col + np.arange(res)
    return ndimage.map_coordinates(scr, np.meshgrid(rr, cc, indexing='ij'), order=3, mode='nearest')


def layer_offset_px(x_arcsec, y_arcsec, altitude, pitch):
    """(row, col) offset in layer pixels of the line of sight to the sky position (x, y) ["], x along rows, y along columns."""
    return altitude * np.tan(x_arcsec * ARCSEC) / pitch, altitude * np.tan(y_arcsec * ARCSEC) / pitch


def expected_star_window(ref, psfs, star_px, s, centres, w=W):
    """
    Expected window around a star at the integer frame offset `star_px` = (dx, dy) px from the axis: sum over patches
    of weight(pixel) x PSF_k(pixel - star), PSF_k being the PSF of the screen seen from the centre of patch (i, j).
    """
    sx, sy = star_px
    rows = np.arange(sx - w, sx + w + 1)
    cols = np.arange(sy - w, sy + w + 1)
    ax, ay = rows * PS, cols * PS
    c = ref.N // 2
    E = np.zeros((2 * w + 1, 2 * w + 1))
    n = len(centres)
    for i in range(n):
        wx = patch_weights(i, ax, s, centres)
        for j in range(n):
            wy = patch_weights(j, ay, s, centres)
            if wx.max() == 0 or wy.max() == 0:
                continue
            E += (wx[:, None] * wy[None, :]) * psfs[i, j][c + rows[:, None] - sx, c + cols[None, :] - sy]
    return E / E.sum()


def patch_weight_table(star_px, s, centres):
    """Weights of the n x n patches at the star position and rms distance ["] to the patch centres."""
    ax, ay = star_px[0] * PS, star_px[1] * PS
    n = len(centres)
    wt = np.array([[patch_weights(i, np.array([ax]), s, centres)[0] * patch_weights(j, np.array([ay]), s, centres)[0]
                    for j in range(n)] for i in range(n)])
    d = np.array([[np.hypot(ax - centres[i], ay - centres[j]) for j in range(n)] for i in range(n)])
    return wt, d


# ----------------------------------------------------------------------------------------------------------------------
# SAOS set-ups
# ----------------------------------------------------------------------------------------------------------------------

def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def pattern_stars(npix):
    """Integer pixel offsets (ix, iy) from the image centre of the stars, rounded as in the owner's generator."""
    out = []
    for r, phi in STAR_POLAR:
        out.append(dict(r=r, phi=phi, x=r * np.cos(np.deg2rad(phi)), y=r * np.sin(np.deg2rad(phi)),
                        ix=int(np.round(r * np.cos(np.deg2rad(phi)) / PS)), iy=int(np.round(r * np.sin(np.deg2rad(phi)) / PS))))
    return out


def write_h5(path, data):
    with h5py.File(path, 'w') as f:
        g = f.create_group('image')
        g.attrs['FoV'] = data.shape[0] * PS
        g.attrs['resolution'] = PS
        g.create_dataset('data', data=data.astype(np.float32), dtype=np.float32)


def write_pattern(path):
    """Owner's recipe (generateExtendedAsterism.py): delta stars written as image[x_px, y_px], plus the marker star."""
    npix = int(np.round(20 / PS))
    img = np.zeros((npix, npix))
    stars = pattern_stars(npix)
    for st in stars:
        img[st['ix'] + npix // 2, st['iy'] + npix // 2] = 1.0
    write_h5(path, img / img.sum())
    return stars, npix


def make_telescope(logger):
    return Telescope(resolution=RES, diameter=D, centralObstruction=OBS / D, samplingTime=TS, fov=30, logger=logger)


def make_sun(path, logger, n=NSUB):
    return ExtendedSource('V', coordinates=[0, 0], nSubDirs=n, fov=FOV, subDir_margin=MARGIN, patch_padding=PAD,
                          img_path=path, logger=logger)


def make_cam(tel, logger, plate=PS, light_ratio=1):
    return ScienceCam(fieldOfView=FOV, plate_scale=plate, samplingTime=TS, telescope=tel, integrationTime=TS,
                      lightRatio=light_ratio, logger=logger, device=device())


def make_atmosphere(tel, logger, altitude, seed, r0=R0):
    atm = Atmosphere(r0=r0, L0=L0, fractionalR0=[1.0], altitude=[altitude], windDirection=[0], windSpeed=[0],
                     telescope=tel, zenith=0, logger=logger)
    atm.initializeAtmosphere(randomState=seed)
    return atm


def science_frame(src, atm, tel, cam, logger):
    lp = LightPath(logger)
    lp.initialize_path(src=src, atm=atm, tel=tel, sci=cam)
    lp.propagate(True)
    return lp.sci_frame.copy(), lp


def run_case(path, tel, logger, atm, stars):
    """Extended-source frame and the NGS frames (same atmosphere) of the 18 directions."""
    sun = make_sun(path, logger)
    cam = make_cam(tel, logger)
    ext, _ = science_frame(sun, atm, tel, cam, logger)
    # The star sits at image [row, col] = (r cos phi, r sin phi) (owner's image[x_px, y_px]); in the SAOS sky frame
    # (atmosphere, DMs: x = column, y = row) that direction is (x, y) = (r sin phi, r cos phi), i.e. azimuth 90 - phi
    ngs = [science_frame(Source('V0', 5, coordinates=[st['r'], 90.0 - st['phi']], logger=logger), atm, tel, cam, logger)[0]
           for st in stars]
    return dict(sun=sun, cam=cam, ext=ext, ngs=np.array(ngs), atm=atm, n=ext.shape[0])


def star_px(case, st):
    """Frame offset (dx, dy) of a star from the optical axis [px] (axis = pixel nPix//2)."""
    return st['ix'], st['iy']


def star_windows(case, stars, w=W):
    n = case['n']
    return np.array([window(case['ext'], n // 2 + st['ix'], n // 2 + st['iy'], w) for st in stars])


def ngs_windows(case, w=W):
    n = case['n']
    return np.array([window(f, n // 2, n // 2, w) for f in case['ngs']])


def mosaic(report, name, ext, ref, labels, title, ref_name='independent'):
    """Rows: extended-source star PSF, reference PSF, residual (ext - ref, in % of the reference peak)."""
    import matplotlib.pyplot as plt
    n = len(ext)
    fig, ax = plt.subplots(3, n, figsize=(1.25 * n, 4.2))
    for k in range(n):
        vmax = ref[k].max()
        ax[0, k].imshow(np.log10(np.maximum(ext[k] / vmax, 1e-5)), vmin=-4, vmax=0, cmap='magma')
        ax[1, k].imshow(np.log10(np.maximum(ref[k] / vmax, 1e-5)), vmin=-4, vmax=0, cmap='magma')
        res = 100 * (ext[k] - ref[k]) / vmax
        lim = max(1.0, np.abs(res).max())
        ax[2, k].imshow(res, vmin=-lim, vmax=lim, cmap='RdBu_r')
        ax[0, k].set_title(labels[k], fontsize=6)
        ax[2, k].set_xlabel(f'{np.abs(res).max():.2f}%', fontsize=6)
        for a in ax[:, k]:
            a.set_xticks([])
            a.set_yticks([])
    ax[0, 0].set_ylabel('extended', fontsize=7)
    ax[1, 0].set_ylabel(ref_name, fontsize=7)
    ax[2, 0].set_ylabel('residual', fontsize=7)
    fig.suptitle(title, fontsize=8)
    report.figure(fig, name)


def star_labels(stars):
    return [f'{st["r"]:.2f}",{st["phi"]:.0f}' for st in stars]


# ----------------------------------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tel(logger):
    return make_telescope(logger)


@pytest.fixture(scope='module')
def pattern(tmp_path_factory):
    path = str(tmp_path_factory.mktemp('ext') / 'asterism.h5')
    stars, npix = write_pattern(path)
    return path, stars, npix


@pytest.fixture(scope='module')
def ref():
    return RefPSF()


@pytest.fixture(scope='module')
def flat_case(pattern, tel, logger):
    return run_case(pattern[0], tel, logger, None, pattern[1])


@pytest.fixture(scope='module')
def ground_case(pattern, tel, logger, tolerances):
    atm = make_atmosphere(tel, logger, 0.0, tolerances['extended_source']['seed'])
    return run_case(pattern[0], tel, logger, atm, pattern[1])


@pytest.fixture(scope='module')
def altitude_case(pattern, tel, logger, tolerances):
    atm = make_atmosphere(tel, logger, H_ALT, tolerances['extended_source']['seed'])
    return run_case(pattern[0], tel, logger, atm, pattern[1])


# ----------------------------------------------------------------------------------------------------------------------
# B1  no atmosphere
# ----------------------------------------------------------------------------------------------------------------------

def test_b1_no_atmosphere_stars(flat_case, pattern, ref, tolerances, report):
    """
    No atmosphere. Every star of the object must appear at its pixel of the input image (rows = first image axis) with
    the PSF of the telescope: position within +-0.1 px of the input pixel, FWHM of the obstructed-pupil PSF (own FFT of the
    annulus 4.149 m / 1.3 m) and window PSF (unit sum, 79 px) equal to the independent PSF within a fraction of the peak.
    The NGS PSFs of the same directions are recorded for comparison (no criterion).
    """
    tol = tolerances['extended_source']['b1']
    _, stars, _ = pattern
    n = flat_case['n']
    P = ref.psf()
    E = ref.window(P, W)
    fwhm_ref = fwhm_2d(E)
    report.metric('independent PSF FWHM [px]', fwhm_ref)
    ext_w, ngs_w = star_windows(flat_case, stars), ngs_windows(flat_case)
    pos_err, fwhm_rel, diff_ref, diff_ngs = [], [], [], []
    for k, st in enumerate(stars):
        r0, c0 = n // 2 + st['ix'], n // 2 + st['iy']
        pr, pc = peak_position(window(flat_case['ext'], r0, c0, 16))
        pos_err.append([pr - 16, pc - 16])
        fwhm_rel.append(fwhm_2d(ext_w[k]) / fwhm_ref - 1)
        diff_ref.append(np.abs(ext_w[k] - E).max() / E.max())
        diff_ngs.append(np.abs(ext_w[k] - ngs_w[k]).max() / ngs_w[k].max())
    pos_err = np.array(pos_err)
    report.check('star position error (row, col) [px]', pos_err.ravel(), expected=0.0, atol=tol['pos_atol_px'])
    report.check('FWHM / independent FWHM - 1', fwhm_rel, expected=0.0, atol=tol['fwhm_rtol'])
    report.check('max |PSF - independent PSF| / peak, per star', diff_ref, expected=0.0, atol=tol['psf_atol'])
    report.metric('max |PSF - NGS PSF| / peak, per star (SAOS NGS, same direction)', diff_ngs)
    report.metric('FWHM NGS / independent FWHM - 1', fwhm_2d(ngs_w[0]) / fwhm_ref - 1)
    import matplotlib.pyplot as plt
    mosaic(report, 'b1_mosaic', ext_w, np.array([E] * len(stars)), star_labels(stars),
           'B1 no atmosphere: star PSFs (log10, 4 decades), window 1.0" half-width')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(np.arange(len(stars)), np.array(fwhm_rel) * 100, 'o-', label='extended source')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].axhspan(-100 * tol['fwhm_rtol'], 100 * tol['fwhm_rtol'], color='g', alpha=0.15, label='tolerance')
    ax[0].set_xticks(np.arange(len(stars)))
    ax[0].set_xticklabels(star_labels(stars), rotation=90, fontsize=6)
    ax[0].set_ylabel('FWHM / independent - 1 [%]')
    ax[0].legend()
    ax[1].plot(pos_err[:, 0], pos_err[:, 1], 'o')
    ax[1].set_xlabel('row error [px]')
    ax[1].set_ylabel('col error [px]')
    ax[1].set_title('peak position error (tolerance +-%.2f px)' % tol['pos_atol_px'])
    ax[1].set_xlim(-0.3, 0.3)
    ax[1].set_ylim(-0.3, 0.3)
    ax[1].grid(alpha=0.3)
    report.figure(fig, 'b1_fwhm_position')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# B2  orientation
# ----------------------------------------------------------------------------------------------------------------------

def test_b2_image_not_transposed_or_mirrored(flat_case, pattern, report):
    """
    The 18-star pattern (17 D4-symmetric stars + the asymmetric marker at x = +3.0", y = -1.2") is observed without
    atmosphere. The peaks found in the frame must coincide with the input pixels (first image axis -> frame rows) for the
    identity and for no other transformation of the dihedral group (transposition, mirrors, rotations).
    """
    _, stars, _ = pattern
    f, n = flat_case['ext'], flat_case['n']
    pk = (ndimage.maximum_filter(f, size=21) == f) & (f > 0.2 * f.max())
    found = np.argwhere(pk) - n // 2
    report.check('number of peaks in the frame', len(found), expected=len(stars), atol=0)
    exp = np.array([[st['ix'], st['iy']] for st in stars])
    transforms = {
        'identity': lambda a: a, 'transposed': lambda a: a[:, ::-1], 'mirror rows': lambda a: a * [-1, 1],
        'mirror cols': lambda a: a * [1, -1], 'rotation 180': lambda a: -a, 'anti-transposed': lambda a: -a[:, ::-1],
        'rotation 90': lambda a: np.stack([-a[:, 1], a[:, 0]], 1), 'rotation 270': lambda a: np.stack([a[:, 1], -a[:, 0]], 1)}
    for name, t in transforms.items():
        te = t(exp)
        m = sum(bool(np.any(np.hypot(found[:, 0] - p[0], found[:, 1] - p[1]) < 0.7)) for p in te)
        if name == 'identity':
            report.check('stars found at the input pixels (identity)', m, expected=len(stars), atol=0)
        else:
            report.check(f'stars matched by the image {name}', m, hi=len(stars) - 1)
    mk = stars[-1]
    sel = np.argmin(np.hypot(found[:, 0] - mk['ix'], found[:, 1] - mk['iy']))
    report.metric('marker expected (row, col) offset [px]', [mk['ix'], mk['iy']])
    report.metric('marker nearest peak (row, col) offset [px]', found[sel])
    report.assert_ok()


def _footprint_offsets(altitude_case):
    """
    Footprint offsets (row, col) [px] of the 9 sub-directions relative to the central one, measured by cross-correlating
    the OPD of each sub-direction with the on-axis one (whitened, integer + parabola), with the physical expectation
    (sky x on rows, exact h tan(theta)) and the one with transposed layer axes.
    """
    atm, sun = altitude_case['atm'], altitude_case['sun']
    pitch = layer_pitch(atm)
    opd = atm.getOPD(sun)
    assert opd.shape[0] == NSUB ** 2
    s, centres = patch_geometry()
    ref0 = whiten_opd(opd[(NSUB // 2) * NSUB + NSUB // 2])
    F0 = np.fft.fft2(ref0)
    meas, expd = [], []
    for i in range(NSUB ** 2):
        a, b = centres[i // NSUB], centres[i % NSUB]
        expd.append(np.array(layer_offset_px(a, b, H_ALT, pitch)))
        meas.append(measure_offset(opd[i], F0))
    return np.array(meas), np.array(expd), np.array(expd)[:, ::-1]


def whiten_opd(a):
    return a - ndimage.gaussian_filter(a, 8)


def measure_offset(a, F0):
    """Shift t [px] (row, col) maximising c(t) = sum_p a(p) ref(p + t) (whitened maps, parabolic sub-pixel peak)."""
    F = np.fft.fft2(whiten_opd(a))
    c = np.real(np.fft.ifft2(np.conj(F) * F0))
    r, q = np.unravel_index(np.argmax(c), c.shape)
    out = []
    for ax_, k in ((0, r), (1, q)):
        m, z, p = (c[(r - 1) % c.shape[0], q], c[r, q], c[(r + 1) % c.shape[0], q]) if ax_ == 0 else \
                  (c[r, (q - 1) % c.shape[1]], c[r, q], c[r, (q + 1) % c.shape[1]])
        den = m - 2 * z + p
        d = 0.0 if den == 0 else 0.5 * (m - p) / den
        k = k - c.shape[ax_] if k > c.shape[ax_] // 2 else k
        out.append(k + d)
    return np.array(out)


def _footprint_figure(report, expd, meas):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(expd[:, 1], expd[:, 0], 'ks', ms=9, mfc='none', label='expected (image axes)')
    ax.plot(meas[:, 1], meas[:, 0], 'rx', label='SAOS atmosphere')
    for i in range(NSUB ** 2):
        ax.annotate(f'{i // NSUB},{i % NSUB}', (expd[i, 1], expd[i, 0]), fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('column offset of the footprint [px]')
    ax.set_ylabel('row offset of the footprint [px]')
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.set_title('footprint of the sub-direction (row block, col block)', fontsize=8)
    report.figure(fig, 'b2_footprint_offsets')


def test_b2_subdirection_footprint_orientation(altitude_case, report):
    """
    One layer at 15 km. The sub-direction i of the extended source is imaged in row block i // 3 and column block i % 3 of
    the mosaic (ScienceCam), whose centre is at sky offset (a, b) = ((i//3 - 1) s/2, (i%3 - 1) s/2) with s the patch size and
    image rows along sky x. The line of sight crosses the layer displaced by h tan(a) rows and h tan(b) columns (layer
    axes = image axes). The footprint offset of each sub-direction is measured by cross-correlation with the on-axis one.
    Orientation criterion (no tolerance): for every sub-direction off the diagonal (a != b, where the transposed footprint
    differs by tens of pixels) the measured offset is closer to the physical one than to the one with transposed axes.
    """
    meas, expd, tr = _footprint_offsets(altitude_case)
    off = np.array([i // NSUB != i % NSUB for i in range(NSUB ** 2)])
    e_phys = np.abs(meas - expd).max(axis=1)
    e_tr = np.abs(meas - tr).max(axis=1)
    report.metric('measured footprint offsets (row, col) [px], sub-directions 0..8', meas)
    report.metric('expected footprint offsets (row, col) [px] (sky x on rows)', expd)
    report.metric('max |error| w.r.t. the physical offsets [px], off-diagonal sub-directions', e_phys[off])
    report.metric('max |error| w.r.t. the transposed-axes offsets [px], off-diagonal sub-directions', e_tr[off])
    report.check('sub-directions closer to the transposed footprint than to the physical one', int((e_tr[off] < e_phys[off]).sum()), expected=0, atol=0)
    _footprint_figure(report, expd, meas)
    report.assert_ok()


def test_b2_subdirection_footprint_subpixel(altitude_case, tolerances, report):
    """
    Same measurement as test_b2_subdirection_footprint_orientation, restricted to the diagonal sub-directions (a = b, 0, 4, 8)
    where the layer axes (transposed or not) give the same offset, so that only the footprint position is tested: the measured
    offset must match h tan(theta) rounded to the nearest layer pixel (integer-pixel model, owner's decision 2026-10-27). The
    tolerance is the precision of the offset estimator itself (whitened cross-correlation + parabola), 8 seeds.
    """
    tol = tolerances['extended_source']['b2']
    meas, expd, _ = _footprint_offsets(altitude_case)
    diag = np.array([i // NSUB == i % NSUB for i in range(NSUB ** 2)])
    report.metric('|h tan(theta) - rounded| [px] (integer-pixel model, <= 0.5)', np.abs(expd - np.round(expd))[diag])
    expd = np.round(expd)
    err = (meas - expd)[diag]
    report.metric('measured diagonal footprint offsets (row, col) [px]', meas[diag])
    report.metric('expected diagonal footprint offsets (row, col) [px]', expd[diag])
    report.check('footprint offset error, diagonal sub-directions [px]', err.ravel(), expected=0.0, atol=tol['offset_atol_px'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# B3  ground layer only
# ----------------------------------------------------------------------------------------------------------------------

def test_b3_ground_layer_isoplanatic(ground_case, pattern, ref, tel, tolerances, report):
    """
    A single frozen layer at 0 m (one realisation): every direction sees the same phase, so each star of the extended
    object has the PSF of that screen (own FFT of the annulus with the phase window of the screen) and equals the NGS PSF
    of its direction. Criteria on windows of 79 px with unit sum: max |ext - independent| / peak and
    max |ext - NGS| / peak below the tolerances.
    """
    tol = tolerances['extended_source']['b3']
    _, stars, _ = pattern
    scr = np.array(ground_case['atm'].layer_1.screen.scrn)
    assert scr.shape[0] == RES
    E = ref.window(ref.psf(scr), W)
    ext_w, ngs_w = star_windows(ground_case, stars), ngs_windows(ground_case)
    d_ref = np.array([np.abs(w - E).max() / E.max() for w in ext_w])
    d_ngs = np.array([np.abs(a - b).max() / b.max() for a, b in zip(ext_w, ngs_w)])
    d_ngs_ref = np.array([np.abs(w - E).max() / E.max() for w in ngs_w])
    report.metric('independent SR (peak / diffraction-limited peak)', ref.psf(scr).max() / ref.psf().max())
    report.check('max |ext - independent| / peak, per star', d_ref, expected=0.0, atol=tol['ref_atol'])
    report.check('max |ext - NGS| / peak, per star', d_ngs, expected=0.0, atol=tol['ngs_atol'])
    report.metric('max |NGS - independent| / peak, per star', d_ngs_ref)
    mosaic(report, 'b3_mosaic', ext_w, np.array([E] * len(stars)), star_labels(stars),
           'B3 ground layer (r0 0.6 m): extended-object star PSFs vs PSF of the screen')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(d_ref * 100, 'o-', label='extended vs independent')
    ax.plot(d_ngs * 100, 's-', label='extended vs NGS')
    ax.plot(d_ngs_ref * 100, '^-', label='NGS vs independent', alpha=0.6)
    ax.axhline(100 * tol['ref_atol'], color='C0', ls=':')
    ax.axhline(100 * tol['ngs_atol'], color='C1', ls=':')
    ax.set_xticks(np.arange(len(stars)))
    ax.set_xticklabels(star_labels(stars), rotation=90, fontsize=6)
    ax.set_ylabel('max |difference| / peak [%]')
    ax.legend(fontsize=7)
    report.figure(fig, 'b3_residual_per_star')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# B4  altitude layer
# ----------------------------------------------------------------------------------------------------------------------

def _patch_psfs(ref, atm, transposed=False):
    scr = np.array(atm.layer_1.screen.scrn)
    pitch = layer_pitch(atm)
    s, centres = patch_geometry()
    psfs = {}
    for i in range(NSUB):
        for j in range(NSUB):
            o = np.round(layer_offset_px(centres[i], centres[j], H_ALT, pitch))   # integer-pixel footprint model
            o = o[::-1] if transposed else o
            psfs[i, j] = ref.psf(screen_window(scr, *o))
    return psfs


@pytest.fixture(scope='module')
def b4_expect(altitude_case, pattern, ref):
    """Star windows of the extended frame and expectations (offsets rounded to the layer pixel; transposed layer axes) from the screen."""
    _, stars, _ = pattern
    s, centres = patch_geometry()
    psfs, psfs_t = _patch_psfs(ref, altitude_case['atm']), _patch_psfs(ref, altitude_case['atm'], transposed=True)
    ext_w = star_windows(altitude_case, stars)
    E = np.array([expected_star_window(ref, psfs, (st['ix'], st['iy']), s, centres) for st in stars])
    Et = np.array([expected_star_window(ref, psfs_t, (st['ix'], st['iy']), s, centres) for st in stars])
    return dict(stars=stars, ext=ext_w, E=E, Et=Et)


def _peak_diff(a, b):
    return np.array([np.abs(x - y).max() / y.max() for x, y in zip(a, b)])


def test_b4_altitude_layer_orientation(b4_expect, tolerances, report):
    """
    One frozen layer at 15 km (r0 0.6 m). The extended source sees, in each patch, the phase of the line of sight to the
    patch centre (sky offset (a, b), layer offset (h tan a, h tan b) rows/columns of the screen) and the mosaic blends the
    patch images with tent weights. Expected star window E from own FFT of the annulus and the cubic sub-pixel window of the
    screen (exact offsets), E_t the same with transposed layer axes. Orientation criterion: among the stars for which the two
    expectations are distinguishable (max |E - E_t| / peak above the numerical precision of the reference, yaml
    b4.ref_atol) the extended image is closer to E than to E_t.
    """
    tol = tolerances['extended_source']['b4']
    ext_w, E, Et = b4_expect['ext'], b4_expect['E'], b4_expect['Et']
    d, dt = _peak_diff(ext_w, E), _peak_diff(ext_w, Et)
    sep = _peak_diff(E, Et)
    # Orientation is decided only where the two expectations differ clearly (orientation_min_sep of the peak)
    sel = sep > max(tol['ref_atol'], tol['orientation_min_sep'])
    report.metric('max |E - E_t| / peak per star (distinguishability)', sep)
    report.metric('max |ext - E| / peak per star', d)
    report.metric('max |ext - E_t| / peak per star', dt)
    report.check('distinguishable stars closer to the transposed expectation than to the physical one', int((dt[sel] < d[sel]).sum()), expected=0, atol=0)
    report.assert_ok()


def test_b4_altitude_layer_psf_vs_screen(b4_expect, tolerances, report):
    """
    Same set-up as test_b4_altitude_layer_orientation. Absolute criterion on every star: max |ext - E| / peak <= b4.psf_atol
    (2 %, owner's criterion), E built from the screen with the footprint offsets h tan(theta) rounded to the layer pixel
    (integer-pixel model). The criterion covers the extended-image model error itself (patch resampling and PSF blending,
    validated on the ground layer in b3: 0.6-0.9 %); the reference precision is b4.ref_atol (0.11 %). The layer-axes
    orientation is fixed (EXT-TRANSPOSE solved) and tested separately by test_b4_altitude_layer_orientation.
    """
    tol = tolerances['extended_source']['b4']
    ext_w, E, Et = b4_expect['ext'], b4_expect['E'], b4_expect['Et']
    stars = b4_expect['stars']
    d = _peak_diff(ext_w, E)
    report.metric('max |ext - E| / peak, per star', d)
    report.check('max |ext - expected| / peak, all stars', d, expected=0.0, atol=tol['psf_atol'])
    mosaic(report, 'b4_mosaic', ext_w, E, star_labels(stars), 'B4 layer at 15 km: extended star PSF vs expected (tent-weighted patch PSFs)')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(d * 100, 'o-', label='expected, image axes')
    ax.plot(_peak_diff(ext_w, Et) * 100, 's-', label='expected, transposed layer axes')
    ax.axhline(100 * tol['psf_atol'], color='k', ls=':', label='tolerance')
    ax.set_xticks(np.arange(len(stars)))
    ax.set_xticklabels(star_labels(stars), rotation=90, fontsize=6)
    ax.set_ylabel('max |ext - expected| / peak [%]')
    ax.legend(fontsize=7)
    report.figure(fig, 'b4_residual_per_star')
    report.assert_ok()


def test_b4_anisoplanatic_difference_grows_with_patch_distance(altitude_case, pattern, ref, tolerances, report):
    """
    The star of the extended object sees the phase of its patch centre(s), the NGS the phase of its own direction: their PSF
    difference is zero at a patch centre and grows with the distance to the patch centres. Independent anisoplanatic phase
    variance of a star: sigma^2 = sum_k w_k S(h d_k), with d_k the distance to centre k and S(r) the mean squared phase
    difference at lag r measured on the actual screen (isotropic average of row and column lags). Criteria: difference at
    patch centres below the numerical floor; mean difference of the off-centre stars above ratio_min times that of the stars at
    patch centres; Spearman correlation of the difference with sigma^2 above the calibrated minimum. The peak ratio ext/NGS is recorded against exp(-sigma^2).
    """
    tol = tolerances['extended_source']['b4_growth']
    _, stars, _ = pattern
    s, centres = patch_geometry()
    atm = altitude_case['atm']
    scr = np.array(atm.layer_1.screen.scrn)
    pitch = layer_pitch(atm)
    lags = np.arange(1, 40)
    dx, dy = citools.structure_function(scr, lags)
    S = lambda r_m: np.interp(r_m / pitch, np.r_[0, lags], np.r_[0.0, 0.5 * (dx + dy)])
    ext_w, ngs_w = star_windows(altitude_case, stars), ngs_windows(altitude_case)
    diff = np.array([np.abs(a - b).max() / b.max() for a, b in zip(ext_w, ngs_w)])
    sig2, dist, sr_ratio = [], [], []
    sr_dl = ref.psf().max()
    for k, st in enumerate(stars):
        wt, dd = patch_weight_table((st['ix'], st['iy']), s, centres)
        sig2.append(float(np.sum(wt * S(H_ALT * np.tan(dd * ARCSEC)))))
        dist.append(float(np.sqrt(np.sum(wt * dd ** 2))))
        sr_ratio.append(ext_w[k].max() / ngs_w[k].max())
    sig2, dist, sr_ratio = np.array(sig2), np.array(dist), np.array(sr_ratio)
    centre = dist < 1e-3
    report.check('max difference at patch centres / peak', diff[centre].max(), lo=0.0, hi=tol['floor'])
    report.check('mean difference off-centre / mean difference at centres', diff[~centre].mean() / diff[centre].mean(),
                 lo=tol['ratio_min'])
    rho = stats.spearmanr(sig2, diff).statistic
    report.check('Spearman(difference, sigma^2)', rho, lo=tol['spearman_min'])
    report.metric('Spearman(difference, rms distance to centres)', stats.spearmanr(dist, diff).statistic)
    report.metric('max |ext - NGS| / peak per star', diff)
    report.metric('sigma^2 [rad^2] per star', sig2)
    report.metric('peak ratio ext/NGS per star', sr_ratio)
    report.metric('exp(-sigma^2) per star', np.exp(-sig2))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].scatter(dist, diff * 100, c=sig2, cmap='viridis')
    ax[0].set_xlabel('rms distance to the patch centres ["]')
    ax[0].set_ylabel('max |ext - NGS| / peak [%]')
    ax[1].scatter(sig2, diff * 100)
    ax[1].set_xlabel('anisoplanatic phase variance sigma^2 [rad^2] (from the screen)')
    ax[1].set_ylabel('max |ext - NGS| / peak [%]')
    ax[1].set_title(f'Spearman {rho:.2f}', fontsize=9)
    ax[2].scatter(np.exp(-sig2), sr_ratio)
    ax[2].plot([0, 1], [0, 1], 'k:', lw=0.7)
    ax[2].set_xlabel('exp(-sigma^2)')
    ax[2].set_ylabel('peak(ext) / peak(NGS)')
    for a in ax:
        a.grid(alpha=0.3)
    report.figure(fig, 'b4_difference_vs_distance')
    mosaic(report, 'b4_mosaic_ext_vs_ngs', ext_w, ngs_w, star_labels(stars), 'B4 extended star PSF vs NGS PSF (same direction)',
           ref_name='NGS')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# B5  uniform object, mosaic, flux
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('plate', [PS, 0.016])
def test_b5_uniform_object_gives_uniform_image(plate, altitude_case, tel, logger, tmp_path, tolerances, report):
    """
    Uniform object: the image is uniform whatever the PSF (convolution of a constant with a unit-energy PSF is the
    constant), so a mosaic of patches seen through different phases must show no seams. Criteria: no NaN, std/mean and
    max |frame/mean - 1| below the float64 floor, with the flat phase and with the altitude layer (different phase in each
    patch), for the object plate scale (0.0128"/px) and a coarser camera pixel (0.016"/px).
    """
    tol = tolerances['extended_source']['b5_uniform']
    npix = int(np.round(20 / PS))
    path = str(tmp_path / 'uniform.h5')
    write_h5(path, np.ones((npix, npix)))
    sun = make_sun(path, logger)
    cam = make_cam(tel, logger, plate=plate)
    for label, atm in (('flat', None), ('layer 15 km', altitude_case['atm'])):
        f, _ = science_frame(sun, atm, tel, cam, logger)
        m = f.mean()
        report.check(f'NaN count ({label})', int(np.isnan(f).sum()), expected=0, atol=0)
        report.check(f'std / mean ({label})', f.std() / m, lo=0.0, hi=tol['floor'])
        report.check(f'max |frame / mean - 1| ({label})', np.abs(f / m - 1).max(), lo=0.0, hi=tol['floor'])
        report.check(f'frame energy ({label})', f.sum(), expected=1.0, rtol=1e-12)
    report.assert_ok()


def test_b5_structured_object_mosaic_has_no_seams(ground_case, pattern, tel, logger, tmp_path, ref, tolerances, report):
    """
    Smooth random texture (+-30% contrast, correlation length 12 px) observed through the single ground layer: all patches
    have the same PSF, so the frame must be the convolution of the object with the PSF of the screen (own FFT convolution of
    the object window with the 601 px PSF). Criterion: max |frame - expected| / mean(expected) over the 779 px frame, which
    shows seams or offsets between patches as steps of the order of the object contrast.
    """
    tol = tolerances['extended_source']['b5_mosaic']
    npix = int(np.round(20 / PS))
    rng = np.random.default_rng(tol['texture_seed'])
    tex = ndimage.gaussian_filter(rng.standard_normal((npix, npix)), 12)
    tex = 1 + 0.3 * tex / tex.std()
    path = str(tmp_path / 'texture.h5')
    write_h5(path, tex)
    sun = make_sun(path, logger)
    cam = make_cam(tel, logger)
    atm = ground_case['atm']
    f, _ = science_frame(sun, atm, tel, cam, logger)
    n = f.shape[0]
    P = ref.psf(np.array(atm.layer_1.screen.scrn))
    h = 300
    c = ref.N // 2
    K = P[c - h:c + h + 1, c - h:c + h + 1]
    c0 = npix // 2
    tex32 = tex.astype(np.float32).astype(float)           # the h5 stores float32
    sub = tex32[c0 - n // 2 - h:c0 - n // 2 - h + n + 2 * h, c0 - n // 2 - h:c0 - n // 2 - h + n + 2 * h]
    expected = signal.fftconvolve(sub, K, mode='valid')
    expected /= expected.sum()
    m = f / f.sum()
    d = (m - expected) / expected.mean()
    report.metric('frame contrast (std/mean)', f.std() / f.mean())
    report.check('max |frame - expected| / mean', np.abs(d).max(), lo=0.0, hi=tol['max_atol'])
    report.check('rms |frame - expected| / mean', d.std(), lo=0.0, hi=tol['rms_atol'])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, im, t in zip(ax, (m, expected, d * 100), ('SAOS frame', 'independent convolution', 'difference / mean [%]')):
        h_ = a.imshow(im, cmap='RdBu_r' if im is d * 100 or t.startswith('diff') else 'gray')
        a.set_title(t, fontsize=9)
        plt.colorbar(h_, ax=a, fraction=0.046)
    report.figure(fig, 'b5_mosaic_texture')
    report.assert_ok()


@pytest.mark.parametrize('n_sub', [2, 3, 4, 5])
def test_b5_filters_partition_of_unity(n_sub, pattern, logger, report):
    """
    The filters that blend the n x n patches must sum to one at every pixel of the mosaic. Independent placement: patch
    (i, j) at the corner (i, j) x round(W / 2) px (50% overlap, as ScienceCam places them) of a mosaic of size
    (n - 1) round(W / 2) + W, with W the filter width in px; the sum of the filters of SAOS placed there must be 1 everywhere.
    """
    sun = make_sun(pattern[0], logger, n=n_sub)
    filt = sun.filter_2D
    W_ = filt.shape[0]
    assert filt.shape == (W_, W_, n_sub, n_sub)
    stride = int(np.round(W_ / 2))
    size = (n_sub - 1) * stride + W_
    acc = np.zeros((size, size))
    for i in range(n_sub):
        for j in range(n_sub):
            acc[i * stride:i * stride + W_, j * stride:j * stride + W_] += filt[:, :, i, j]
    report.check(f'min of the filter sum (n={n_sub})', acc.min(), expected=1.0, rtol=1e-12)
    report.check(f'max of the filter sum (n={n_sub})', acc.max(), expected=1.0, rtol=1e-12)
    report.metric('filter width [px]', W_)
    report.assert_ok()


def test_b5_flux(flat_case, pattern, ref, tel, logger, tolerances, report):
    """
    Flux: (a) photon flux of the source = E(500 nm) Dl (Omega_FoV / Omega_sun) lambda / (h c) with the ASTM G173 direct
    irradiance and the solar disk of 960"; the sub-directions share it; (b) the long-exposure frame of the LightPath holds
    flux T A_eff photons (ScienceCam lightRatio = 1); (c) the noise-free frame equals the independent convolution of the
    object with the PSF (own FFT) over the whole 779 px frame: star by star, flux within 1" (sum over the aperture / frame)
    and max |frame - expected| / peak.
    """
    import pvlib
    import scipy.constants as sc
    tol = tolerances['extended_source']['b5_flux']
    path, stars, npix = pattern
    sun, n = flat_case['sun'], flat_case['n']
    spec = pvlib.spectrum.get_reference_spectra(standard='ASTM G173-03')
    E500 = np.interp(500.0, spec.index.values, spec['direct'].values)
    N_exp = 10e-9 * 1e9 * E500 * (FOV / 206265.0) ** 2 / (np.pi * (960.0 / 206265.0) ** 2) * LAM / (sc.h * sc.c)
    report.check('source flux / expected [photons/m2/s]', sun.flux / N_exp, expected=1.0, rtol=tol['flux_rtol'])
    report.check('sum of sub-direction fluxes / source flux', sum(s_.flux for s_ in sun.subDirs_stars) / sun.flux, expected=1.0,
                 rtol=tol['flux_rtol'])
    cam = make_cam(tel, logger, light_ratio=1)
    _, lp = science_frame(sun, None, tel, cam, logger)
    le = lp.long_exposure_frame
    report.check('long exposure photons / (flux T A_eff)', le.sum() / (N_exp * TS * tel.area_eff), expected=1.0, rtol=tol['flux_rtol'])
    cam50 = make_cam(tel, logger)
    cam50.lightRatio = 50
    _, lp50 = science_frame(sun, None, tel, cam50, logger)
    report.metric('long exposure photons with ScienceCam lightRatio 50 (default before 2026-10-27) / (flux T A_eff) (context)',
                  lp50.long_exposure_frame.sum() / (N_exp * TS * tel.area_eff))
    f = flat_case['ext']
    P = ref.psf()
    h = 300
    c = ref.N // 2
    K = P[c - h:c + h + 1, c - h:c + h + 1]
    img = np.zeros((n + 2 * h, n + 2 * h))
    for st in stars:
        img[h + n // 2 + st['ix'], h + n // 2 + st['iy']] = 1.0
    expected = signal.fftconvolve(img, K, mode='valid')
    expected /= expected.sum()
    m = f / f.sum()
    report.check('max |frame - independent| / peak', np.abs(m - expected).max() / expected.max(), lo=0.0, hi=tol['frame_atol'])
    rr, cc = np.indices(f.shape)
    ratios = []
    for st in stars:
        ap = np.hypot(rr - (n // 2 + st['ix']), cc - (n // 2 + st['iy'])) <= 1.0 / PS
        ratios.append(m[ap].sum() / expected[ap].sum())
    report.check('star flux within 1" / independent', ratios, expected=1.0, rtol=tol['star_flux_rtol'])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ratios, 'o-')
    ax[0].axhline(1, color='k', lw=0.6)
    ax[0].set_xticks(np.arange(len(stars)))
    ax[0].set_xticklabels(star_labels(stars), rotation=90, fontsize=6)
    ax[0].set_ylabel('flux within 1" / independent')
    im = ax[1].imshow(100 * (m - expected) / expected.max(), cmap='RdBu_r')
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set_title('frame - independent [% of peak]', fontsize=9)
    report.figure(fig, 'b5_flux')
    report.assert_ok()


def test_b5_default_science_photon_budget(flat_case, tel, logger, report):
    """
    With the default ScienceCam arguments the long-exposure frame of a noise-free LightPath must hold the photons that the
    source delivers through the effective area in the exposure time, flux x T x A_eff (a camera that records all the light
    of the path): ScienceCam.lightRatio is the fraction of the photons reaching the camera (LightPath.py and
    ScienceCam.apply_noise), 1.0 by default (it was 50 before 2026-10-27).
    """
    sun = flat_case['sun']
    cam = ScienceCam(fieldOfView=FOV, plate_scale=PS, samplingTime=TS, telescope=tel, integrationTime=TS, logger=logger,
                     device=device())
    _, lp = science_frame(sun, None, tel, cam, logger)
    expected = sun.flux * TS * tel.area_eff
    report.metric('ScienceCam default lightRatio', cam.lightRatio)
    report.check('long exposure photons / (flux T A_eff), default camera', lp.long_exposure_frame.sum() / expected,
                 expected=1.0, rtol=1e-6)
    report.assert_ok()


def test_band_without_source_entry(logger, report):
    """
    ExtendedSource uses its own band table (C.9, owner: the wavelengths of the solar images are kept): a band absent from the
    Source table ('IR') constructs, and the sub-direction sources carry the wavelength of the extended source.
    """
    from SAOS.ExtendedSource import ExtendedSource
    for band in ('IR', 'V', 'R'):
        sun = ExtendedSource(optBand=band, coordinates=[0, 0], nSubDirs=2, img_PS=0.0128, fov=6.0, subDir_margin=4.0,
                             patch_padding=10.0, logger=logger)
        lam = {s.wavelength for s in sun.subDirs_stars}
        report.check(f'{band}: sub-direction wavelengths == extended source wavelength', float(lam == {sun.wavelength}), 1.0, atol=0)
    report.assert_ok()

