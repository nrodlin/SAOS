"""
T1 validation of the interaction-matrix measurement (SAOS.InteractionMatrixHandler) and of the control laws
(SAOS.Controller): IM columns against an independent physical forward model (documented Gaussian DM + Shack-Hartmann
spot centroid), linearity and push-pull symmetry of the IM, IM bookkeeping (identical DMs, save/load, per-DM modal
bases), reconstructors against independent SVD / Tikhonov / least-squares solutions of prescribed matrices, and the
dynamics of the leaky / PI controllers against their z-transforms.

Independent references (no SAOS code or outputs): `dm_surface` re-implements the documented DM (Gaussian influence
functions of coupling 0.6 over a regular grid, partition of unity) and `sh_slopes` propagates the surface over each
physical subaperture to the spot (zero-padded Fourier transform sampled at 2 px per lambda/d) and takes the centre of
gravity over the 15 px window. Reconstructor / control-law references are numpy SVD, Tikhonov normal equations and
difference equations written from the documented formulas. Tests that compare SAOS with another SAOS path are named
`test_regression_*`.

Configuration: 48 px square pupil, D = 0.6 m, 6x6-subaperture Shack-Hartmann (all pixels of the window used, so the
centroid is odd and continuous), 7x7 DM, zernike modal basis, CPU. The controller is driven with a linear plant in
modal space, s(k) = IM (a(k) + c(k-D)), where a is the disturbance, c the modal command and D the total pure delay of
the loop, so that the closed loop is exactly the one of the control textbook:
    c = C(z) e,  e = -R s,  residual u = a + z^-D c   =>   u / a = S(z) = 1 / (1 + C(z) z^-D)
with C_leaky = g / (1 - d z^-1), C_forwardPI = (g + (ki - g) z^-1) / (1 - z^-1), C_backwardPI = (g + ki - g z^-1) / (1 - z^-1).
"""
import types

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy import signal

from citools import known_issue
from SAOS.Controller import Controller
from SAOS.DeformableMirror import DeformableMirror
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.LightPath import LightPath
from SAOS.ShackHartmann import ShackHartmann
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

RES, NSUB, D = 48, 6, 0.6
P = D / RES                                    # phase pixel [m]
SUBAP = D / NSUB
COORD = np.arange(RES) - (RES - 1) / 2
X, Y = np.meshgrid(COORD, COORD)               # X: column coordinate [px], Y: row coordinate [px]
STROKE = 2e-8
NM = 12                                        # modes used by the controller tests


def make_tel(logger):
    return Telescope(resolution=RES, diameter=D, samplingTime=1e-3, pupil=np.ones((RES, RES), dtype=bool),
                     centralObstruction=0.0, logger=logger)


def make_wfs(logger, tel, src, samp=2.0, fov_px=15):
    ps = src.wavelength / SUBAP * 206265 / samp
    wfs = ShackHartmann(nSubap=NSUB, telescope=tel, src=src, lightRatio=1.0, plate_scale=ps, fieldOfView=fov_px * ps,
                        guardPx=1, use_brightest=fov_px ** 2, logger=logger, device='cpu')
    wfs.k_px = src.wavelength / (2 * np.pi * P) * 206265 / ps        # spot shift [px] per phase gradient [rad/px]
    wfs.samp = samp
    return wfs


def make_dm(logger, tel, nActs=7):
    return DeformableMirror(telescope=tel, nActs=nActs, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)


def make_lp(logger, tel, src, dm, wfs, delay=0, atm=None):
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, atm=atm, dm=dm, wfs=wfs, delay=delay)
    return lp


def split_xy(wfs, s):
    n = wfs.nValidSubaperture
    return s[:n], s[n:]


def zero_index(wfs):
    """Window pixel holding the zero-tilt spot (zero frequency of SAOS's FFT crop)."""
    nfft = int(np.round(wfs.samp * wfs.npix_lenslet))
    return nfft // 2 - (nfft - wfs.npix_lenslet) // 2 - (wfs.npix_lenslet - wfs.npix_subap) // 2


def reference_cog(wfs, shift_px):
    """Exact CoG shift [px] of a square-subaperture sinc^2 spot moved by shift_px, same window as SAOS."""
    x = np.arange(wfs.npix_subap) - zero_index(wfs)
    i1 = np.sinc((x - shift_px) / wfs.samp) ** 2
    i0 = np.sinc(x / wfs.samp) ** 2
    return (x * i1).sum() / i1.sum() - (x * i0).sum() / i0.sum()


class System:
    """tel, src, wfs, dm and light path of the single-WFS / single-DM configuration, plus its measured IM."""

    def __init__(self, logger):
        self.logger = logger
        self.tel = make_tel(logger)
        self.src = Source('R', 0, logger=logger)
        self.wfs = make_wfs(logger, self.tel, self.src)
        self.dm = make_dm(logger, self.tel)
        self.lp = make_lp(logger, self.tel, self.src, self.dm, self.wfs)
        self.handler = InteractionMatrixHandler(logger)
        self.handler.initialize_im_class([self.lp])
        self.handler.measure('zernike', STROKE)
        self.IM = self.handler.interaction_matrix_warehouse[0][0]['IM']
        self.basis = self.handler.modal_basis[0]['zernike']


@pytest.fixture(scope='module')
def sysm(logger):
    return System(logger)


def measure_im(handler, basis, stroke, nModes=None):
    handler.measure(basis, stroke, nModes=nModes)
    return [[None if e['IM'] is None else e['IM'].copy() for e in row] for row in handler.interaction_matrix_warehouse]


# ----------------------------------------------------------------------------------------------------------------------
# Independent physical forward model: documented DM surface -> Shack-Hartmann slopes
# ----------------------------------------------------------------------------------------------------------------------

def dm_surface(cmd_valid, X, Y, nacts=7, coupling=0.6, thresh=0.5):
    """
    Surface [m] of the documented DM (Gaussian influence functions, value `coupling` at one pitch, partition of unity)
    at the points (X, Y) [m]: nacts x nacts actuators over [-D/2, D/2], row-major, valid if r <= D/2 + thresh pitch.
    """
    x = np.linspace(-D / 2, D / 2, nacts)
    ax, ay = np.meshgrid(x, x)
    ax, ay = ax.ravel(), ay.ravel()
    pitch = D / (nacts - 1)
    valid = np.hypot(ax, ay) <= D / 2 + thresh * pitch
    r2 = (X.reshape(-1, 1) - ax[valid]) ** 2 + (Y.reshape(-1, 1) - ay[valid]) ** 2
    g = np.exp(np.log(coupling) * r2 / pitch ** 2)
    return ((g @ np.asarray(cmd_valid, dtype=float)) / g.sum(axis=1)).reshape(X.shape), valid


SAMP, FOV_PX = 2.0, 15                           # pixels per lambda/d and window size of make_wfs
ZERO_PX = FOV_PX // 2                            # window pixel of the zero frequency: the lenslet axis at the window centre (WFS-SPOT-OFFCENTRE, solved)
M_FINE = 32                                      # samples per subaperture side of the forward model


def sh_slopes(surface, wavelength, m=M_FINE):
    """
    Slopes [px] (x then y, subapertures row-major, rows = y) of a square Shack-Hartmann seeing the OPD `surface(X, Y)`
    [m] over the physical pupil [-D/2, D/2]^2 divided in NSUB x NSUB subapertures of side d. Per subaperture: field
    exp(2 pi i OPD / lambda) on an m x m midpoint grid, intensity at the angles k lambda / (SAMP d) (Fourier transform
    evaluated at the window pixels k - ZERO_PX, k = 0..FOV_PX-1; an x-gradient g gives a spot at +g), centre of gravity
    over the window, minus the value of a flat wavefront. A phase gradient g (OPD slope) moves the spot by g SAMP d /
    lambda pixels.
    """
    d = D / NSUB
    um = (np.arange(m) + 0.5) / m * d
    kx = np.arange(FOV_PX) - ZERO_PX
    Ek = np.exp(-2j * np.pi * np.outer(kx / (SAMP * d), um))
    out = np.zeros((2, NSUB, NSUB))
    for r in range(NSUB):
        for c in range(NSUB):
            X, Y = np.meshgrid(-D / 2 + c * d + um, -D / 2 + r * d + um)
            F = Ek @ np.exp(2j * np.pi * surface(X, Y) / wavelength) @ Ek.T         # [ky, kx]
            I = np.abs(F) ** 2
            out[0, r, c] = (I.sum(0) * kx).sum() / I.sum()
            out[1, r, c] = (I.sum(1) * kx).sum() / I.sum()
    return out.reshape(2, -1)


def expected_im(cmds_valid, wavelength, stroke, m=M_FINE):
    """Independent IM [px/m] of the given modal commands (columns of cmds_valid, valid actuators): slopes / stroke."""
    flat = sh_slopes(lambda X, Y: np.zeros_like(X), wavelength, m)
    cols = []
    for k in range(cmds_valid.shape[1]):
        sl = sh_slopes(lambda X, Y: dm_surface(stroke * cmds_valid[:, k], X, Y)[0], wavelength, m) - flat
        cols.append(sl.ravel() / stroke)
    return np.array(cols).T


# ----------------------------------------------------------------------------------------------------------------------
# Interaction matrix
# ----------------------------------------------------------------------------------------------------------------------

def _valid_mask():
    return dm_surface(np.zeros(37), np.zeros((1, 1)), np.zeros((1, 1)))[1]


def test_im_columns_vs_independent_forward_model(sysm, report, tolerances):
    """
    Absolute IM (slopes [px] per metre of mode amplitude) of the first NM zernike modes against the independent
    forward model of the documented DM and a physical Shack-Hartmann (see module docstring): the surface of the commanded
    mode (command = the applied basis column x stroke) is propagated over the physical subapertures. No normalisation
    by SAOS gains: all 2 x 36 slopes of every mode must agree in absolute terms, error normalised to the largest
    expected entry of the same mode (tolerance: precision of the reference, see yaml).
    """
    tol = tolerances['im_controller']['forward_model']
    valid = _valid_mask()
    modes = sysm.basis[valid, :NM]
    E = expected_im(modes, sysm.src.wavelength, STROKE)
    S = sysm.IM[:, :NM]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in range(NM):
        err = np.max(np.abs(S[:, m] - E[:, m])) / np.max(np.abs(E[:, m]))
        gain = (S[:, m] @ E[:, m]) / (E[:, m] @ E[:, m])
        report.check(f'mode {m}: max |IM - expected| / max |expected|', err, hi=tol['abs_rel'])
        report.metric(f'mode {m}: least-squares gain SAOS / expected (information)', gain)
    ax.plot(E.ravel(), S.ravel(), '.', ms=2)
    lim = np.abs(E).max()
    ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.6)
    ax.set_xlabel('independent forward model [px/m]'); ax.set_ylabel('SAOS IM [px/m]'); ax.set_aspect('equal')
    report.figure(fig, 'im_vs_forward_model')
    report.assert_ok()


@pytest.mark.parametrize('mode,label', [(2, 'defocus'), (9, 'spherical')])
@known_issue('WFS-SUBAP-GRID', 'split_subapertures samples subaperture k on phase pixels [k np, k np + np] (one duplicated border '
             'row/column, centre shifted +0.5 px from the pupil grid), so a mirror-symmetric phase gives slope maps that are not '
             'mirror-antisymmetric along the gradient axis (x-mirror of s_x; the y-mirror of s_y alike)', 'ShackHartmann.py:365-405')
def test_im_symmetry_of_symmetric_modes(mode, label, sysm, report, tolerances):
    """
    A DM shape with the symmetry of the sensor (defocus and spherical aberration: even under x -> -x, y -> -y and x <-> y)
    must produce slope maps with the same symmetry: s_x(r, c) = -s_x(r, 5 - c) = s_x(5 - r, c) = s_y(c, r), whatever
    the scale of the sensor. Errors normalised to the largest slope (float32 pipeline: yaml bound). The DM surface itself is
    symmetric to 1e-13 (independent model), checked first.

    Diagnosis of the failure (scratch experiment, SAOS untouched): the asymmetry is in the sensor, not in the DM or the IM.
    An analytic symmetric phase (0.1 rad defocus / spherical on the 48 px pupil) fed directly to ShackHartmann.wfs_measure gives
    x-mirror 5.1e-2 (defocus) and 5.7e-2 (spherical), y-mirror of s_x 1e-4 / 4.9e-2 (spherical), x<->y 1e-4: only the axis
    along which the +0.5 px shift of the subaperture grid matters is broken (s_x of a defocus depends on x only). With
    split_subapertures replaced by an independent symmetric split (subaperture k = value at the pixel edges
    [k np - 1/2, k np + np - 1/2], replicate-padded, same bilinear resampling afterwards) the same computation gives x-mirror
    2.3e-4 (defocus) / 2.3e-4 (spherical, 0.01 rad, linear regime) and y-mirror 8e-5 / 1e-4, all below the yaml bound: the
    asymmetry is WFS-SUBAP-GRID (see test_wfs.py::test_defocus_subaperture_geometry).
    """
    tol = tolerances['im_controller']['symmetry']
    valid = _valid_mask()
    xs = np.linspace(-D / 2, D / 2, 64)
    Xp, Yp = np.meshgrid(xs, xs)
    w = dm_surface(STROKE * sysm.basis[valid, mode], Xp, Yp)[0]
    report.check(f'{label}: independent DM surface asymmetry / max', max(np.abs(w - w[:, ::-1]).max(), np.abs(w - w[::-1]).max(), np.abs(w - w.T).max()) / np.abs(w).max(), hi=1e-10)
    sx = split_xy(sysm.wfs, sysm.IM[:, mode])[0].reshape(NSUB, NSUB)
    sy = split_xy(sysm.wfs, sysm.IM[:, mode])[1].reshape(NSUB, NSUB)
    scale = max(np.abs(sx).max(), np.abs(sy).max())
    report.check(f'{label}: x-mirror antisymmetry of s_x, max |s(r,c) + s(r,5-c)| / max', np.abs(sx + sx[:, ::-1]).max() / scale, hi=tol['rel'])
    report.check(f'{label}: y-mirror symmetry of s_x, max |s(r,c) - s(5-r,c)| / max', np.abs(sx - sx[::-1]).max() / scale, hi=tol['rel'])
    report.check(f'{label}: x <-> y symmetry, max |s_x(r,c) - s_y(c,r)| / max', np.abs(sx - sy.T).max() / scale, hi=tol['rel'])
    report.assert_ok()


def test_regression_im_column_equals_direct_wfs_response(sysm, report, tolerances):
    """
    REGRESSION (equal-path, not a physics check): the IM column of each of the first modes equals the slopes of the
    sensor for the OPD of the same DM shape applied directly (IM = slopes / stroke).
    """
    s = sysm
    for m in range(NM):
        s.dm.updateDMShape(s.basis[:, m] * STROKE, dynamicResponse=False)
        opd = s.dm.dm_layer.OPD.copy()
        s.dm.updateDMShape(np.zeros(s.dm.nValidAct), dynamicResponse=False)
        direct = s.wfs.wfs_measure(opd * 2 * np.pi / s.src.wavelength, s.src)[0] / STROKE
        report.check(f'mode {m}: IM column vs direct WFS response (max rel)', np.max(np.abs(s.IM[:, m] - direct)) / np.max(np.abs(direct)),
                     hi=tolerances['im_controller']['tilt']['direct'])
    report.assert_ok()


def test_im_linear_in_stroke_and_push_pull(logger, report, tolerances):
    """
    IM = slopes / stroke is independent of the stroke where the sensor is linear: strokes of 2e-8 and 4e-8 m (0.02-0.05
    px) give the same matrix, and the push (+stroke) and pull (-stroke) matrices coincide (odd response of the
    all-pixels centroid). Errors normalised to the largest entry.
    """
    tol = tolerances['im_controller']['linearity']
    tel = make_tel(logger)
    src = Source('R', 0, logger=logger)
    dm = make_dm(logger, tel)
    lp = make_lp(logger, tel, src, dm, make_wfs(logger, tel, src))
    h = InteractionMatrixHandler(logger)
    h.initialize_im_class([lp])
    nm = 12
    im1 = measure_im(h, 'zernike', 2e-8, [nm])[0][0]
    im2 = measure_im(h, 'zernike', 4e-8, [nm])[0][0]
    imn = measure_im(h, 'zernike', -2e-8, [nm])[0][0]
    ref = np.max(np.abs(im1))
    report.check('max |IM(4e-8) - IM(2e-8)| / max', np.max(np.abs(im2 - im1)) / ref, hi=tol['stroke'])
    report.check('max |IM(-2e-8) - IM(+2e-8)| / max', np.max(np.abs(imn - im1)) / ref, hi=tol['push_pull'])
    report.metric('IM shape', np.array(im1.shape))
    report.assert_ok()


@known_issue('IM-DEDUP', 'two distinct DM objects with the same (nActs, altitude, nValidAct, mechCoupling) are merged '
             'into one scanned DM: the second one is never poked and its light path gets an all-zero IM',
             'InteractionMatrixHandler.py:103-106')
def test_identical_dms_in_two_lightpaths_both_measured(logger, report, tolerances):
    """
    Two light paths, each with its own DM object and WFS (DMs with identical properties, identical optics). Expected:
    two distinct scanned DMs; warehouse[i][j] holds an IM only where DM i is seen by light path j, i.e. i == j (DM 0
    in light path 0, DM 1 in light path 1, None in the crossed entries); both IMs non-zero and, the optics being
    identical, equal to each other.
    """
    tol = tolerances['im_controller']
    tel = make_tel(logger)
    src = Source('R', 0, logger=logger)
    dms = [make_dm(logger, tel), make_dm(logger, tel)]
    lps = [make_lp(logger, tel, src, dm, make_wfs(logger, tel, src)) for dm in dms]
    h = InteractionMatrixHandler(logger)
    h.initialize_im_class(lps)
    report.check('number of scanned DMs', len(h.dm_scanned_list), expected=2, atol=0)
    report.check('scanned DMs are the two distinct objects', len({id(d) for d in h.dm_scanned_list}) == 2 and all(any(d is x for x in h.dm_scanned_list) for d in dms), 1.0, atol=0)
    h.measure('zonal', STROKE)
    wh = h.interaction_matrix_warehouse

    def im(i, j):
        try:
            return wh[i][j]['IM']
        except IndexError:
            return None
    for i in range(2):
        for j in range(2):
            got = im(i, j)
            report.check(f'warehouse[{i}][{j}] holds an IM (expected {i == j})', got is not None, float(i == j), atol=0)
    a, b = im(0, 0), im(1, 1)
    for k, m in enumerate((a, b)):
        report.check(f'light path {k}: max |IM| [px/m]', np.max(np.abs(m)) if m is not None else 0.0, lo=tol['nonzero_min'])
    if a is not None and b is not None and a.shape == b.shape:
        report.check('IM(DM 0, LP 0) == IM(DM 1, LP 1): max |diff| / max', np.max(np.abs(a - b)) / np.max(np.abs(a)), hi=tol['reconstructor']['matrix'])
    else:
        report.check('both IMs available with the same shape', 0.0, expected=1.0, atol=0)
    report.assert_ok()


def _two_lp_system(logger, science_first):
    tel = make_tel(logger)
    src = Source('R', 0, logger=logger)
    dm = make_dm(logger, tel)
    lps = []
    if science_first:
        lps.append(make_lp(logger, tel, src, dm, None))
    lps.append(make_lp(logger, tel, src, dm, make_wfs(logger, tel, src)))
    return lps


def _save_load(logger, lps, path):
    h = InteractionMatrixHandler(logger)
    h.initialize_im_class(lps)
    h.measure('zonal', STROKE, nModes=[10])
    h.save_IM(str(path))
    ref = [[None if e['IM'] is None else e['IM'].copy() for e in row] for row in h.interaction_matrix_warehouse]
    h2 = InteractionMatrixHandler(logger)
    h2.initialize_im_class(lps)
    h2.load_IM(str(path))
    return h, h2, ref


def _compare_warehouse(h2, ref, report):
    worst = 0.0
    for i, row in enumerate(ref):
        for j, im in enumerate(row):
            got = h2.interaction_matrix_warehouse[i][j]['IM']
            if im is None:
                report.check(f'DM {i} / LP {j}: no IM stays None', got is None, 1.0, atol=0)
            else:
                worst = max(worst, np.max(np.abs(got - im)) / np.max(np.abs(im))) if got is not None else np.inf
    report.check('max |IM_loaded - IM_saved| / max', worst, hi=0.0)


def test_regression_save_load_im_round_trip(logger, tmp_path, report):
    """REGRESSION (round trip): save_IM + load_IM into a fresh handler of the same optical configuration restores the IM bit by bit."""
    lps = _two_lp_system(logger, science_first=False)
    h, h2, ref = _save_load(logger, lps, tmp_path / 'im.h5')
    _compare_warehouse(h2, ref, report)
    report.check('modal basis name restored', h2.interaction_matrix_warehouse[0][0]['modalBasis'] in ('zonal', b'zonal'), 1.0, atol=0)
    report.assert_ok()


@known_issue('IM-LOAD', 'load_IM looks the light paths up as LP0..LP(n-1) in the file, but save_IM names the groups with the '
             'light-path index: a science-only light path placed first leaves the file without LP0 -> KeyError', 
             'InteractionMatrixHandler.py:440-451')
def test_regression_save_load_im_with_science_lightpath_first(logger, tmp_path, report):
    """REGRESSION (round trip): a science-only light path (no WFS) before the WFS light path must not break the save/load round trip."""
    lps = _two_lp_system(logger, science_first=True)
    h, h2, ref = _save_load(logger, lps, tmp_path / 'im_sci.h5')
    _compare_warehouse(h2, ref, report)
    report.assert_ok()


@known_issue('IM-LIST', 'measure() documents a per-DM list of modal bases but tests `modal_basis not in self.modal_list` '
             'with the list itself, so every list raises ValueError (Unsupported modal basis)',
             'InteractionMatrixHandler.py:239')
def test_regression_per_dm_modal_basis_list(logger, report, tolerances):
    """REGRESSION (equal path): measure(['zonal'], ...) for a one-DM system is accepted and equals measure('zonal', ...)."""
    tel = make_tel(logger)
    src = Source('R', 0, logger=logger)
    dm = make_dm(logger, tel)
    lp = make_lp(logger, tel, src, dm, make_wfs(logger, tel, src))
    h = InteractionMatrixHandler(logger)
    h.initialize_im_class([lp])
    ref = measure_im(h, 'zonal', STROKE, [8])[0][0]
    got = measure_im(h, ['zonal'], STROKE, [8])[0][0]
    report.check('max |IM(list) - IM(str)| / max', np.max(np.abs(got - ref)) / np.max(np.abs(ref)), hi=1e-12)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Controller: linear modal plant
# ----------------------------------------------------------------------------------------------------------------------

class MockLP:
    """Light-path stand-in: the controller only calls get_wavefront_error()."""

    def __init__(self, n):
        self.s = np.zeros(n)

    def get_wavefront_error(self):
        return self.s


def make_ctrl(sysm, logger, ctype, gain=0.4, decay=0.99, ki=0.0, method='inversion', op='closed', delay=1, **kw):
    return Controller(telescope=sysm.tel, interactionMatrix=sysm.handler, reconstructionMethod=method, controllerType=ctype,
                      operationType=op, logger=logger, gain=[gain], decay=[decay], ki=[ki], rcond=1e-3, nModes=NM,
                      delay=delay, **kw)


def modal_cmd(ctrl):
    return ctrl.command_previous[0].cpu().numpy().ravel().copy()


def run_modal_loop(ctrl, IM, a_seq, D, closed=True):
    """
    Linear plant s(k) = IM (a(k) + c(k-D)) (closed) or s(k) = IM a(k) (open), controller called once per step.
    Returns the true modal residual u(k) = a(k) + c(k-D) and the modal command c(k).
    """
    K = a_seq.shape[0]
    lp = MockLP(IM.shape[0])
    c = np.zeros((K, a_seq.shape[1]))
    u = np.zeros_like(c)
    for k in range(K):
        u[k] = a_seq[k] + (c[k - D] if (closed and k - D >= 0) else 0.0)
        lp.s = IM @ u[k]
        ctrl.computeControlAction([lp])
        c[k] = modal_cmd(ctrl)
    return u, c


def controller_tf(ctype, gain, decay, ki):
    """C(z) as ascending powers of z^-1: (numerator, denominator)."""
    if ctype == 'leaky':
        return np.array([gain, 0.0]), np.array([1.0, -decay])
    if ctype == 'forwardPI':
        return np.array([gain, ki - gain]), np.array([1.0, -1.0])
    return np.array([gain + ki, -gain]), np.array([1.0, -1.0])           # backwardPI


def sensitivity_tf(num, den, D):
    """S = den / (den + num z^-D) as same-length ascending polynomials in z^-1 (= descending in z)."""
    L = max(len(den), D + len(num))
    A = np.zeros(L); A[:len(den)] = den
    Bz = np.zeros(L); Bz[D:D + len(num)] = num
    return A, A + Bz


CTRL_CASES = [('leaky', 0.4, 0.99, 0.0, 2), ('forwardPI', 0.1, 1.0, 0.2, 2), ('backwardPI', 0.1, 1.0, 0.2, 1)]
CTRL_IDS = [f'{c[0]}-D{c[4]}' for c in CTRL_CASES]


def ctrl_mats(sysm):
    return sysm.IM[:, :NM]


@pytest.mark.parametrize('ctype,gain,decay,ki,D', CTRL_CASES, ids=CTRL_IDS)
def test_controller_step_response_vs_ztransform(ctype, gain, decay, ki, D, sysm, logger, report, tolerances):
    """
    The control law is the difference equation of its z-transform. (a) Open-loop feed (slopes IM u, u = step in mode 0
    plus a random sequence in the others): the modal command equals scipy dlsim of C(z) applied to e = -u. (b) Closed
    loop with the linear plant of pure delay D: the residual of a unit step in mode 0 equals dlsim of
    S(z) = 1/(1 + C(z) z^-D) (u = a + z^-D c). Also R IM = I on the controlled modes (inversion).
    """
    tol = tolerances['im_controller']['controller']
    IM = ctrl_mats(sysm)
    rng = np.random.default_rng(3)
    ctrl = make_ctrl(sysm, logger, ctype, gain, decay, ki)
    R = ctrl.reconstructor[0].cpu().numpy()
    report.check('max |R IM - I|', np.max(np.abs(R @ IM - np.eye(NM))), hi=tol['reconstructor'])
    num, den = controller_tf(ctype, gain, decay, ki)
    # (a) open loop
    K = 80
    u = rng.standard_normal((K, NM)); u[:, 0] = np.where(np.arange(K) >= 5, 1.0, 0.0)
    _, c = run_modal_loop(ctrl, IM, u, D, closed=False)
    ref = np.stack([signal.dlsim(signal.dlti(num, den, dt=1), -u[:, m])[1][:, 0] for m in range(NM)], axis=1)
    report.check('open loop: max |c - dlsim(C, -u)| / max', np.max(np.abs(c - ref)) / np.max(np.abs(ref)), hi=tol['difference_eq'])
    # (b) closed loop step
    ctrl2 = make_ctrl(sysm, logger, ctype, gain, decay, ki)
    K = 120
    a = np.zeros((K, NM)); a[3:, 0] = 1.0
    ur, _ = run_modal_loop(ctrl2, IM, a, D, closed=True)
    A, B = sensitivity_tf(num, den, D)
    poles = np.roots(B)
    report.metric('max |pole| of 1 + C z^-D', np.max(np.abs(poles)))
    refu = signal.dlsim(signal.dlti(A, B, dt=1), a[:, 0])[1][:, 0]
    report.check('closed loop: max |u - dlsim(S, a)| (unit step)', np.max(np.abs(ur[:, 0] - refu)), hi=tol['closed_step'])
    report.check('closed loop: other modes stay at zero (max |u|)', np.max(np.abs(ur[:, 1:])), hi=tol['closed_step'])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    axs[0].plot(ref[:, 0], 'k-', label='dlsim(C, -u)')
    axs[0].plot(c[:, 0], 'o', ms=3, mfc='none', label='SAOS (open-loop feed)')
    axs[0].set_title('open-loop command, mode 0', fontsize=8); axs[0].legend(fontsize=7, frameon=False)
    axs[1].plot(refu, 'k-', label='dlsim 1/(1+C z^-D)'); axs[1].plot(ur[:, 0], 'o', ms=3, mfc='none', label='SAOS loop')
    axs[1].set_title(f'closed-loop residual, unit step ({ctype}, D = {D})', fontsize=8); axs[1].legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.set_xlabel('frame')
    report.figure(fig, 'step_response')
    report.assert_ok()


REJ_CASES = [('leaky', 0.4, 0.99, 0.0, 2), ('leaky', 0.3, 0.995, 0.0, 3), ('forwardPI', 0.1, 1.0, 0.2, 2), ('backwardPI', 0.1, 1.0, 0.2, 1)]


@pytest.mark.parametrize('ctype,gain,decay,ki,D', REJ_CASES, ids=[f'{c[0]}-g{c[1]}-D{c[4]}' for c in REJ_CASES])
def test_rejection_transfer_function(ctype, gain, decay, ki, D, sysm, logger, report, tolerances):
    """
    Rejection |S(e^jwT)| measured with injected sinusoids (mode 0, bins 2..90 of a 256-frame window, one multisine,
    lock-in over an integer number of periods after the transient) equals the analytic 1/(1 + C(z) z^-D)
    (magnitude and phase) of the loop with total pure delay D.
    """
    tol = tolerances['im_controller']['controller']
    IM = ctrl_mats(sysm)
    N, skip, win = 256, 1024, 512
    bins = np.array([2, 5, 10, 20, 40, 64, 90])
    rng = np.random.default_rng(11)
    ph = rng.uniform(0, 2 * np.pi, len(bins))
    K = skip + win
    k = np.arange(K)
    a = np.zeros((K, NM))
    a[:, 0] = sum(0.3 * np.sin(2 * np.pi * b * k / N + p) for b, p in zip(bins, ph))
    ctrl = make_ctrl(sysm, logger, ctype, gain, decay, ki)
    u, _ = run_modal_loop(ctrl, IM, a, D, closed=True)
    num, den = controller_tf(ctype, gain, decay, ki)
    A, B = sensitivity_tf(num, den, D)
    report.metric('max |pole|', np.max(np.abs(np.roots(B))))
    w = 2 * np.pi * bins / N
    z = np.exp(1j * w)
    zz = lambda p: sum(c * z ** (-i) for i, c in enumerate(p))
    S = zz(A) / zz(B)
    kk = np.arange(skip, K)
    basis = np.exp(-1j * 2 * np.pi * np.outer(bins, kk) / N)
    Ua = basis @ u[skip:, 0]
    Aa = basis @ a[skip:, 0]
    Sm = Ua / Aa
    report.check('|S| measured / analytic - 1', np.abs(Sm) / np.abs(S) - 1, expected=0, atol=tol['rejection_mag'])
    report.check('phase(S) measured - analytic [rad]', np.angle(Sm / S), expected=0, atol=tol['rejection_phase'])
    f = bins / N                                                       # cycles per frame
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    fd = np.linspace(0.002, 0.5, 400)
    zd = np.exp(2j * np.pi * fd)
    Sd = sum(c * zd ** (-i) for i, c in enumerate(A)) / sum(c * zd ** (-i) for i, c in enumerate(B))
    axs[0].semilogx(fd, 20 * np.log10(np.abs(Sd)), 'k-', label='analytic')
    axs[0].semilogx(f, 20 * np.log10(np.abs(Sm)), 'o', ms=4, mfc='none', label='SAOS, injected sinusoids')
    axs[0].set_ylabel('|S| [dB]')
    axs[1].semilogx(fd, np.angle(Sd), 'k-'); axs[1].semilogx(f, np.angle(Sm), 'o', ms=4, mfc='none')
    axs[1].set_ylabel('phase(S) [rad]')
    for ax in axs:
        ax.set_xlabel('frequency [cycles / frame]'); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'{ctype} g = {gain}, decay = {decay}, ki = {ki}, pure delay D = {D}', fontsize=8)
    report.figure(fig, 'rejection_transfer_function')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Reconstructors and control law against independent linear algebra on prescribed matrices
# ----------------------------------------------------------------------------------------------------------------------

def mock_handler(H, bases):
    """Interaction-matrix handler stand-in: H[i][j] = IM of DM i seen by light path j (None = not seen), bases[i] the modal basis of DM i."""
    wh = [[dict(IM=h, modalBasis='m', discarded_modes=0) for h in row] for row in H]
    return types.SimpleNamespace(interaction_matrix_warehouse=wh, modal_basis=[dict(m=b) for b in bases])


TEL = types.SimpleNamespace(samplingTime=1e-3)


def ref_pinv(H, rcond):
    """Minimum-norm least-squares inverse: singular values below rcond * s_max discarded."""
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    keep = s > rcond * s.max()
    return (Vt.T[:, keep] / s[keep]) @ U.T[keep]


def ref_tikhonov(H, beta):
    """V diag(s / (s^2 + alpha)) U^T with alpha = beta s_max^2."""
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    return (Vt.T * (s / (s ** 2 + beta * s.max() ** 2))) @ U.T


def ref_joint_tikhonov(H, sizes, betas):
    """(H^T H + diag(alpha_i)) \\ H^T with alpha_i = beta_i s_max(columns of DM i)^2; returns the blocks of rows per DM."""
    gam = np.concatenate([np.full(n, b * np.linalg.svd(H[:, o:o + n], compute_uv=False)[0] ** 2)
                          for n, b, o in zip(sizes, betas, np.cumsum([0] + list(sizes[:-1])))])
    R = np.linalg.solve(H.T @ H + np.diag(gam), H.T)
    return np.split(R, np.cumsum(sizes)[:-1], axis=0)


def first_command(ctrl, mocks, s_list):
    """One call of a leaky controller with gain 1 and decay 0: modal command per DM and DM command."""
    for m, s in zip(mocks, s_list):
        m.s = s
    dm_cmd = [c.cpu().numpy().ravel() for c in ctrl.computeControlAction(mocks)]
    return [c.cpu().numpy().ravel() for c in ctrl.command_previous], dm_cmd


def prescribed(rng, n, sizes, rank_deficient=False):
    """Random prescribed matrix blocks [n x size_i]; rank_deficient: the last column of the last DM is a combination of the first two."""
    blocks = [rng.standard_normal((n, k)) for k in sizes]
    if rank_deficient:
        blocks[-1][:, -1] = 0.7 * blocks[0][:, 0] - 1.3 * blocks[0][:, 1]
    return blocks


@pytest.mark.parametrize('method', ['inversion', 'tikhonov'])
@pytest.mark.parametrize('joint', [True, False], ids=['joint', 'per-DM'])
@pytest.mark.parametrize('rank_deficient', [False, True], ids=['full-rank', 'rank-deficient'])
def test_single_dm_reconstructor_vs_independent_svd(method, joint, rank_deficient, logger, report, tolerances):
    """
    One DM, one WFS, prescribed IM H (30 slopes x 8 modes; rank-deficient case: last column = 0.7 col0 - 1.3 col1).
    The reconstructor and the first command of a gain-1 leaky controller (c = R (-s), DM command = basis c) equal the
    independent numpy solutions: inversion = minimum-norm pseudo-inverse with singular values below rcond s_max dropped;
    tikhonov = V diag(s / (s^2 + beta s_max^2)) U^T. For the rank-deficient inversion also the projector R H onto the row
    space of H, the Penrose condition H R H = H and the absence of any null-space component of the command (minimum norm).
    """
    tol = tolerances['im_controller']['reconstructor']
    rng = np.random.default_rng(100 + 4 * (method == 'inversion') + 2 * joint + rank_deficient)
    H = prescribed(rng, 30, [8], rank_deficient)[0]
    basis = rng.standard_normal((20, 8))
    rcond, beta = 1e-3, 1e-3
    ctrl = Controller(telescope=TEL, interactionMatrix=mock_handler([[H]], [basis]), reconstructionMethod=method, controllerType='leaky',
                      logger=logger, gain=[1.0], decay=[0.0], rcond=rcond, beta=beta, joint_reconstruction=joint)
    ref = ref_pinv(H, rcond) if method == 'inversion' else ref_tikhonov(H, beta)
    R = ctrl.reconstructor[0].cpu().numpy()
    report.check('max |R - R_ref| / max |R_ref|', np.max(np.abs(R - ref)) / np.max(np.abs(ref)), hi=tol['matrix'])
    s = rng.standard_normal(30)
    modal, dm_cmd = first_command(ctrl, [MockLP(30)], [s])
    c_ref = ref @ (-s)
    report.check('modal command: max |c - c_ref| / max |c_ref|', np.max(np.abs(modal[0] - c_ref)) / np.max(np.abs(c_ref)), hi=tol['matrix'])
    report.check('DM command: max |basis c - basis c_ref| / max', np.max(np.abs(dm_cmd[0] - basis @ c_ref)) / np.max(np.abs(basis @ c_ref)), hi=tol['matrix'])
    if method == 'inversion' and rank_deficient:
        U, sv, Vt = np.linalg.svd(H)
        r = int((sv > rcond * sv.max()).sum())
        report.metric('rank kept', r)
        report.check('rank kept == rank of H', r, expected=7, atol=0)
        report.check('R H - V_r V_r^T (row-space projector), max', np.max(np.abs(R @ H - Vt[:r].T @ Vt[:r])), hi=tol['matrix'])
        report.check('H R H - H, max / max |H|', np.max(np.abs(H @ R @ H - H)) / np.max(np.abs(H)), hi=tol['matrix'])
        null = Vt[r:].T
        report.check('null-space component of the command, max |N^T c| / |c|', np.max(np.abs(null.T @ modal[0])) / np.linalg.norm(modal[0]), hi=tol['matrix'])
    report.assert_ok()


@pytest.mark.parametrize('method', ['inversion', 'tikhonov'])
@pytest.mark.parametrize('rank_deficient', [False, True], ids=['full-rank', 'rank-deficient'])
def test_two_dm_coupled_reconstructors_vs_independent_solutions(method, rank_deficient, logger, report, tolerances):
    """
    Two DMs (6 and 4 modes), two WFS (14 slopes each), both WFS see both DMs; prescribed blocks (rank-deficient case:
    the last mode of DM 2 is a combination of two modes of DM 1, in both WFS). Independent references on the stacked H:
    joint inversion = minimum-norm pseudo-inverse (rcond s_max); joint tikhonov = (H^T H + diag(beta_i s_max,i^2))^-1 H^T
    with s_max,i of the columns of DM i; per-DM reconstruction = the same on the stack of the columns of one DM. The
    commands of a gain-1 leaky controller (c = R (-s)) must equal the independent ones (joint and per-DM), and the joint
    command must fit the slopes at least as well as the least-squares optimum allows (residual equal to the independent
    one for the same reconstructor).
    """
    tol = tolerances['im_controller']['reconstructor']
    rng = np.random.default_rng(200 + 2 * (method == 'inversion') + rank_deficient)
    sizes, n = [6, 4], 14
    cols = [prescribed(rng, n, sizes, False) for _ in range(2)]            # per WFS: [block DM1, block DM2]
    if rank_deficient:
        for blk in cols:
            blk[1][:, -1] = 0.7 * blk[0][:, 0] - 1.3 * blk[0][:, 1]
    H = np.vstack([np.hstack(blk) for blk in cols])
    rcond, betas = 1e-3, [1e-3, 3e-3]
    bases = [rng.standard_normal((12, 6)), rng.standard_normal((9, 4))]
    hand = mock_handler([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]], bases)
    s = rng.standard_normal(2 * n)
    for joint in (True, False):
        ctrl = Controller(telescope=TEL, interactionMatrix=hand, reconstructionMethod=method, controllerType='leaky', logger=logger,
                          gain=[1.0, 1.0], decay=[0.0, 0.0], rcond=rcond, beta=betas, joint_reconstruction=joint)
        if joint:
            refs = [ref_pinv(H, rcond)[:6], ref_pinv(H, rcond)[6:]] if method == 'inversion' else ref_joint_tikhonov(H, sizes, betas)
        else:
            refs = [ref_pinv(H[:, :6], rcond) if method == 'inversion' else ref_tikhonov(H[:, :6], betas[0]),
                    ref_pinv(H[:, 6:], rcond) if method == 'inversion' else ref_tikhonov(H[:, 6:], betas[1])]
        name = 'joint' if joint else 'per-DM'
        for i in range(2):
            R = ctrl.reconstructor[i].cpu().numpy()
            report.check(f'{name}: DM {i} max |R - R_ref| / max |R_ref|', np.max(np.abs(R - refs[i])) / np.max(np.abs(refs[i])), hi=tol['matrix'])
        modal, dm_cmd = first_command(ctrl, [MockLP(n), MockLP(n)], [s[:n], s[n:]])
        c_ref = [r @ (-s) for r in refs]
        for i in range(2):
            report.check(f'{name}: DM {i} modal command vs reference (max rel)', np.max(np.abs(modal[i] - c_ref[i])) / np.max(np.abs(c_ref[i])), hi=tol['matrix'])
            report.check(f'{name}: DM {i} DM command vs basis c_ref (max rel)', np.max(np.abs(dm_cmd[i] - bases[i] @ c_ref[i])) / np.max(np.abs(bases[i] @ c_ref[i])), hi=tol['matrix'])
        res = np.linalg.norm(s - H @ np.concatenate([-m for m in modal])) / np.linalg.norm(s)
        res_ref = np.linalg.norm(s - H @ np.concatenate([-c for c in c_ref])) / np.linalg.norm(s)
        report.check(f'{name}: slope fit residual - independent', res - res_ref, expected=0, atol=tol['matrix'])
        if joint and method == 'inversion':
            lsq = np.linalg.lstsq(H, -s, rcond=rcond)[0]                  # numpy minimum-norm least squares
            report.check('joint inversion command vs numpy lstsq minimum-norm solution (max rel)',
                         np.max(np.abs(np.concatenate(modal) - lsq)) / np.max(np.abs(lsq)), hi=tol['matrix'])
    report.assert_ok()


@pytest.mark.parametrize('method', ['inversion', 'tikhonov'])
def test_block_diagonal_reconstructors_vs_independent_solutions(method, logger, report, tolerances):
    """
    WFS i sees only DM i (block-diagonal IM, prescribed 14 x 6 and 14 x 4 blocks): the joint reconstructor of DM i is
    the independent reconstructor of its block on the slopes of WFS i and exactly zero on the other WFS, and the per-DM
    one is that block alone; commands of a gain-1 leaky controller equal the independent ones.
    """
    tol = tolerances['im_controller']['reconstructor']
    rng = np.random.default_rng(300 + (method == 'inversion'))
    n, sizes = 14, [6, 4]
    H0, H1 = rng.standard_normal((n, 6)), rng.standard_normal((n, 4))
    bases = [rng.standard_normal((12, 6)), rng.standard_normal((9, 4))]
    hand = mock_handler([[H0, None], [None, H1]], bases)
    rcond, betas = 1e-3, [1e-3, 3e-3]
    ref_blk = [ref_pinv(H0, rcond), ref_pinv(H1, rcond)] if method == 'inversion' else [ref_tikhonov(H0, betas[0]), ref_tikhonov(H1, betas[1])]
    s = rng.standard_normal(2 * n)
    for joint in (True, False):
        ctrl = Controller(telescope=TEL, interactionMatrix=hand, reconstructionMethod=method, controllerType='leaky', logger=logger,
                          gain=[1.0, 1.0], decay=[0.0, 0.0], rcond=rcond, beta=betas, joint_reconstruction=joint)
        name = 'joint' if joint else 'per-DM'
        for i in range(2):
            R = ctrl.reconstructor[i].cpu().numpy()
            if joint:
                full = np.zeros((sizes[i], 2 * n)); full[:, i * n:(i + 1) * n] = ref_blk[i]
                report.check(f'{name}: DM {i} max |R - R_ref| / max |R_ref| (zero on the other WFS)', np.max(np.abs(R - full)) / np.max(np.abs(full)), hi=tol['matrix'])
            else:
                report.check(f'{name}: DM {i} max |R - R_ref| / max |R_ref|', np.max(np.abs(R - ref_blk[i])) / np.max(np.abs(ref_blk[i])), hi=tol['matrix'])
        modal, dm_cmd = first_command(ctrl, [MockLP(n), MockLP(n)], [s[:n], s[n:]])
        for i in range(2):
            c_ref = ref_blk[i] @ (-s[i * n:(i + 1) * n])
            report.check(f'{name}: DM {i} modal command vs reference (max rel)', np.max(np.abs(modal[i] - c_ref)) / np.max(np.abs(c_ref)), hi=tol['matrix'])
            report.check(f'{name}: DM {i} DM command vs basis c_ref (max rel)', np.max(np.abs(dm_cmd[i] - bases[i] @ c_ref)) / np.max(np.abs(bases[i] @ c_ref)), hi=tol['matrix'])
    report.assert_ok()


def test_coupled_joint_reconstruction_beats_per_dm(logger, report):
    """
    Physics of the coupling: with both WFS seeing both DMs, the per-DM reconstructors R_i = H_i^+ ignore the cross-term
    that the other DM puts into the slopes. For slopes s = H c_true (consistent data, full-rank prescribed H), the
    joint command recovers c_true exactly (independent expectation: H has full column rank, R H = I), while the per-DM
    commands are wrong by the cross-coupling (information, must be larger than 1 % to make the comparison meaningful).
    """
    rng = np.random.default_rng(400)
    sizes, n = [6, 4], 14
    cols = [prescribed(rng, n, sizes, False) for _ in range(2)]
    H = np.vstack([np.hstack(b) for b in cols])
    bases = [rng.standard_normal((12, 6)), rng.standard_normal((9, 4))]
    hand = mock_handler([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]], bases)
    c_true = rng.standard_normal(10)
    s = H @ c_true
    out = {}
    for joint in (True, False):
        ctrl = Controller(telescope=TEL, interactionMatrix=hand, reconstructionMethod='inversion', controllerType='leaky', logger=logger,
                          gain=[1.0, 1.0], decay=[0.0, 0.0], rcond=1e-6, joint_reconstruction=joint)
        modal, _ = first_command(ctrl, [MockLP(n), MockLP(n)], [s[:n], s[n:]])
        out[joint] = np.linalg.norm(-np.concatenate(modal) - c_true) / np.linalg.norm(c_true)
        report.metric(f'{"joint" if joint else "per-DM"}: |-c - c_true| / |c_true|', out[joint])
    report.check('joint: |-c - c_true| / |c_true|', out[True], hi=1e-9)
    report.check('per-DM error (information: cross-coupling) > 1 %', out[False], lo=0.01)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Controller sizing
# ----------------------------------------------------------------------------------------------------------------------

@known_issue('CTRL-NMODES1', 'with nModes = 1 the IM block (n, 1) is squeezed to (n,) and unsqueezed to (1, n): im_per_dm '
             '(POLC prediction) and the per-DM reconstructor have the transposed shape', 'Controller.py:326-328, 417')
def test_single_mode_shapes(sysm, logger, report):
    """nModes = 1: the IM of the DM is (nSignal, 1), the reconstructor (1, nSignal), for joint and per-DM reconstruction."""
    n = sysm.IM.shape[0]
    shapes = {}
    for joint in (True, False):
        c = Controller(telescope=sysm.tel, interactionMatrix=sysm.handler, reconstructionMethod='tikhonov', controllerType='leaky',
                       logger=logger, gain=[0.3], decay=[0.99], nModes=1, joint_reconstruction=joint)
        shapes[joint] = (tuple(c.im_per_dm[0].shape), tuple(c.reconstructor[0].shape))
        report.metric(f'joint={joint}: im_per_dm shape', np.array(shapes[joint][0]))
        report.metric(f'joint={joint}: reconstructor shape', np.array(shapes[joint][1]))
        report.check(f'joint={joint}: im_per_dm == (nSignal, 1)', shapes[joint][0] == (n, 1), 1.0, atol=0)
        report.check(f'joint={joint}: reconstructor == (1, nSignal)', shapes[joint][1] == (1, n), 1.0, atol=0)
    report.assert_ok()


@known_issue('CTRL-SIZING', 'the joint reconstructor sizes every DM block with the modal basis (minus discarded modes) '
             'instead of the number of modes of the measured IM: an IM with fewer modes than the basis cannot be used '
             'unless nModes is also given', 'Controller.py:340-345, failing at 366-391')
def test_joint_sizing_with_fewer_im_modes_than_basis(sysm, logger, report):
    """IM measured with 10 modes of a 37-mode basis: the controller works with 10 modes (reconstructor (10, nSignal))."""
    h = InteractionMatrixHandler(logger)
    h.initialize_im_class([sysm.lp])
    h.measure('zernike', STROKE, nModes=[10])
    c = Controller(telescope=sysm.tel, interactionMatrix=h, reconstructionMethod='tikhonov', controllerType='leaky',
                   logger=logger, gain=[0.3], decay=[0.99])
    report.metric('reconstructor shape', np.array(c.reconstructor[0].shape))
    report.check('reconstructor shape == (10, nSignal)', tuple(c.reconstructor[0].shape) == (10, sysm.IM.shape[0]), 1.0, atol=0)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# POLC
# ----------------------------------------------------------------------------------------------------------------------

def _polc_runs(sysm, logger, D=1, K=60):
    """Closed and POLC controllers (tikhonov, leaky) on the same modal disturbance with plant s = IM (a + c(k-D))."""
    IM = ctrl_mats(sysm)
    rng = np.random.default_rng(21)
    a = np.zeros((K, NM)); a[:, :4] = np.cumsum(0.2 * rng.standard_normal((K, 4)), axis=0) * 0.1 + 0.5
    out = {}
    for op in ('closed', 'polc'):
        c = make_ctrl(sysm, logger, 'leaky', gain=0.3, decay=0.95, method='tikhonov', op=op, delay=D)
        out[op] = (c,) + run_modal_loop(c, IM, a, D, closed=True)
    return IM, a, out


def test_polc_reconstructs_the_open_loop_turbulence(sysm, logger, report, tolerances):
    """
    POLC (tikhonov, leaky configuration, pure delay D = 1 = Controller delay) reconstructs from the pseudo-open-loop slopes
    s_pol = s - IM c(k-D) = IM a with the POLC law (Ellerbroek & Vogel 2003) c(k) = (1 - g) c(k-1) + g (-R s_pol(k)) (unit DC
    gain; the decay of the configuration is not used): the command is that filter of -R IM a, independent of the loop, so it
    differs from the closed-loop law. Reference: the recursion with g = 0.3.
    """
    tol = tolerances['im_controller']['polc']
    IM, a, out = _polc_runs(sysm, logger)
    R = out['polc'][0].reconstructor[0].cpu().numpy()
    e = -(R @ IM @ a.T).T
    ref = np.zeros_like(a)
    prev = np.zeros(NM)
    for k in range(a.shape[0]):
        prev = 0.3 * e[k] + (1 - 0.3) * prev
        ref[k] = prev
    c_cl, c_polc = out['closed'][2], out['polc'][2]
    scale = np.max(np.abs(ref))
    report.check('POLC command vs the POLC filter of -R s_turb (max rel)', np.max(np.abs(c_polc - ref)) / scale, hi=tol['match'])
    report.check('POLC command differs from the closed-loop command (max rel)', np.max(np.abs(c_polc - c_cl)) / scale, lo=tol['differs'])
    report.assert_ok()


def test_polc_slopes_attribute_is_minus_the_turbulence(sysm, logger, report, tolerances):
    """After each call controller.slopes_polc must be -s_turb (error sign convention), i.e. -IM a."""
    IM, a, out = _polc_runs(sysm, logger, K=30)
    c = out['polc'][0]
    # last step: u = a + c(k-D); slopes_polc was computed with the history before the last update
    k = a.shape[0] - 1
    got = c.slopes_polc[0].cpu().numpy().ravel()
    expected = -(IM @ a[k])
    report.check('slopes_polc vs -s_turb (max rel)', np.max(np.abs(got - expected)) / np.max(np.abs(expected)), hi=tolerances['im_controller']['polc']['sign'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Tomography law against the stated equations on prescribed matrices
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('joint', [True, False], ids=['joint', 'per-DM'])
def test_tomography_law_vs_stated_equations(joint, logger, tmp_path, report, tolerances):
    """
    Tomographic control law from prescribed matrices (one WFS with 24 slopes, two DMs with 6 and 5 modes, R_tomo
    18 x 24, target IMs D_i 18 x n_i), leaky gains g_i, decays d_i, 25 frames of random slopes. Stated equations (L&A):
      reconstructor of the target slopes  R = (D^T D + diag(beta_i s_max(D_i)^2))^-1 D^T, D = [D_1 .. D_n]   (joint)
                                          R_i = V diag(s / (s^2 + beta_i s_max^2)) U^T of D_i                 (per DM);
      e_i(k) = R_i ( R_tomo (-s(k)) - sum_j D_j c_j(k-1) ),   c_i(k) = g_i e_i(k) + d_i c_i(k-1).
    The modal commands of every frame must equal this recursion (numpy), including the DM commands basis_i c_i.
    """
    tol = tolerances['im_controller']['reconstructor']
    rng = np.random.default_rng(500 + joint)
    n, sizes, nt = 24, [6, 5], 18
    IM = [rng.standard_normal((n, k)) for k in sizes]
    bases = [rng.standard_normal((12, 6)), rng.standard_normal((9, 5))]
    Rt = rng.standard_normal((nt, n)) / np.sqrt(n)
    Dt = [rng.standard_normal((nt, k)) for k in sizes]
    betas, g, dk = [1e-3, 3e-3], [0.4, 0.3], [0.98, 0.95]
    path = tmp_path / 'rtomo.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('R_tomo', data=Rt)
    ctrl = Controller(telescope=TEL, interactionMatrix=mock_handler([[IM[0]], [IM[1]]], bases), reconstructionMethod='tomography',
                      controllerType='leaky', logger=logger, gain=g, decay=dk, beta=betas, R_tomo_path=str(path), delay=1)
    ctrl.set_tomography_targets(Dt, joint=joint, release_R_tomo=False)
    if joint:
        R = ref_joint_tikhonov(np.hstack(Dt), sizes, betas)
    else:
        R = [ref_tikhonov(Dt[i], betas[i]) for i in range(2)]
    for i in range(2):
        report.check(f'DM {i}: target reconstructor vs independent', np.max(np.abs(ctrl.reconstructor[i].cpu().numpy() - R[i])) / np.max(np.abs(R[i])), hi=tol['matrix'])
    lp = MockLP(n)
    c = [np.zeros(k) for k in sizes]
    worst_m, worst_d = 0.0, 0.0
    for _ in range(25):
        s = rng.standard_normal(n) * 0.2
        lp.s = s
        dm_cmd = [x.cpu().numpy().ravel() for x in ctrl.computeControlAction([lp])]
        modal = [x.cpu().numpy().ravel() for x in ctrl.command_previous]
        resid = Rt @ (-s) - sum(Dt[j] @ c[j] for j in range(2))
        c = [g[i] * (R[i] @ resid) + dk[i] * c[i] for i in range(2)]
        for i in range(2):
            worst_m = max(worst_m, np.max(np.abs(modal[i] - c[i])) / np.max(np.abs(c[i])))
            worst_d = max(worst_d, np.max(np.abs(dm_cmd[i] - bases[i] @ c[i])) / np.max(np.abs(bases[i] @ c[i])))
    report.check('modal commands, 25 frames: max rel deviation from the recursion', worst_m, hi=tol['matrix'])
    report.check('DM commands, 25 frames: max rel deviation from basis c', worst_d, hi=tol['matrix'])
    report.assert_ok()
