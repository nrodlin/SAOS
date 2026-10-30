"""
T1 validation of the deformable mirror: linearity of the influence-function model (piston, poke, superposition), sign
of the DM contribution in a LightPath, fitting of tilt and defocus, footprints of altitude DMs, grid geometry, temporal
dynamics (state-space model) and actuator-level misregistration.

Module under test: SAOS.DeformableMirror (with SAOS.MisRegistration and SAOS.LightPath where noted).
Conventions verified here: the DM is transmissive (OPD = IF @ command, no factor 2) and its OPD is added to the
atmospheric OPD; source azimuth 0 deg moves the altitude footprint along +x (columns), 90 deg along +y (rows).

Independent reference (no SAOS code, no SAOS coordinates): the DOCUMENTED actuator model is re-implemented in this file
(`model_opd`) from the specification in the DeformableMirror docstrings:
  * nActs x nActs actuators on a regular grid spanning the metapupil diameter D_fov, pitch p = D_fov / (nActs - 1),
    index order = row-major (y rows, x columns); valid actuators r <= D_fov/2 + thresh * p;
  * Gaussian influence function g_k(x) = exp(-(eps |x - x_k|)^2), eps = sqrt(-ln c) / p, i.e. g = c at one pitch
    (c = mechanical coupling);
  * partition of unity: W(x) = sum_k a_k g_k(x) / sum_k g_k(x) (the denominator runs over the valid actuators).
The pupil grid is the telescope pixel grid, samples at the cell centres (i - (N-1)/2) D/N (the grid of the pupil, the
atmosphere and the WFS), with a pupil mask built here. The reference never
uses dm.coordinates, dm.high_res_coords, dm.influence_functions or any SAOS-generated OPD. Tests that compare SAOS with
its own alternative code path are named `test_regression_*` and are not physics checks.
"""
import types

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy import signal

from citools import known_issue
from SAOS.DeformableMirror import DeformableMirror
from SAOS.LightPath import LightPath
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

N, D = 64, 4.0                   # pupil sampling and diameter of the small test telescope
ARCSEC = np.pi / 180 / 3600      # rad per arcsec (exact)


# ----------------------------------------------------------------------------------------------------------------------
# Independent model of the documented actuator geometry and Gaussian / partition-of-unity influence functions
# ----------------------------------------------------------------------------------------------------------------------

def pupil_grid(n, diameter):
    """Pupil sample coordinates [m]: centres of the n pixels of D/n across D, (i - (n-1)/2) D/n (x along columns, y along rows)."""
    x = (np.arange(n) - (n - 1) / 2) * diameter / n
    return np.meshgrid(x, x)


def circular_pupil(n, diameter):
    X, Y = pupil_grid(n, diameter)
    return np.hypot(X, Y) <= diameter / 2


def actuators(nacts, d_fov, thresh=0.5):
    """Actuator coordinates [m] of the full nacts^2 grid (row-major), validity mask and pitch."""
    x = np.linspace(-d_fov / 2, d_fov / 2, nacts)
    ax, ay = np.meshgrid(x, x)
    ax, ay = ax.ravel(), ay.ravel()
    pitch = d_fov / (nacts - 1)
    return ax, ay, np.hypot(ax, ay) <= d_fov / 2 + thresh * pitch, pitch


def model_opd(cmd_valid, X, Y, nacts, d_fov, coupling=0.6, thresh=0.5, shift=(0.0, 0.0), rot_deg=0.0, scale=1.0):
    """
    OPD [m] at the points (X, Y) [m] of the Gaussian / partition-of-unity DM for the command of the valid actuators.
    Misregistration (same order as the SAOS docstring): the actuator positions are moved to x' = s R x + d, R a rotation
    by rot_deg clockwise (x right, y up), and the Gaussian width is scaled by s.
    """
    ax, ay, valid, pitch = actuators(nacts, d_fov, thresh)
    th = np.deg2rad(rot_deg)
    px = scale * (ax * np.cos(th) + ay * np.sin(th)) + shift[0]           # clockwise: (x, y) -> (x cos + y sin, -x sin + y cos)
    py = scale * (-ax * np.sin(th) + ay * np.cos(th)) + shift[1]
    px, py = px[valid], py[valid]
    r2 = (X.reshape(-1, 1) - px) ** 2 + (Y.reshape(-1, 1) - py) ** 2
    g = np.exp(np.log(coupling) * r2 / (scale * pitch) ** 2)              # = c at one (scaled) pitch
    return ((g @ np.asarray(cmd_valid, dtype=float)) / g.sum(axis=1)).reshape(X.shape)


def lattice_peak(coupling):
    """Surface value at an interior actuator of an infinite lattice poked with unit command: 1 / (sum_i c^(i^2))^2."""
    i = np.arange(-30, 31)
    return 1.0 / np.sum(coupling ** (i ** 2.0)) ** 2


@pytest.fixture(scope='module')
def tel(logger):
    return Telescope(resolution=N, diameter=D, samplingTime=1 / 2000, centralObstruction=0.0, fov=60, logger=logger,
                     pupil=circular_pupil(N, D))


@pytest.fixture(scope='module')
def dm(tel, logger):
    """11x11 ground DM, default mechanical coupling (0.6), actuators within D/2 + 0.5 pitch."""
    return DeformableMirror(telescope=tel, nActs=11, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)


@pytest.fixture(scope='module')
def nvalid():
    return int(actuators(11, D)[2].sum())


def shape(dm, cmd):
    dm.updateDMShape(np.asarray(cmd, dtype=float), dynamicResponse=False)
    return dm.dm_layer.OPD.copy()


def src_at(arcsec, az_deg, wavelength=500e-9):
    return types.SimpleNamespace(coordinates=[arcsec, az_deg], tag='NGS', wavelength=wavelength)


# ----------------------------------------------------------------------------------------------------------------------
# Linearity and sign
# ----------------------------------------------------------------------------------------------------------------------

def test_piston_gives_flat_opd(dm, tel, report, tolerances):
    """Partition of unity: an equal command a on every actuator gives OPD = a everywhere in the pupil."""
    a = 1e-6
    opd = shape(dm, np.full(dm.nValidAct, a))
    pup = circular_pupil(N, D)
    report.check('peak-to-valley / a inside the pupil', (opd[pup].max() - opd[pup].min()) / a, hi=tolerances['dm']['linearity']['rel'])
    report.check('mean / a inside the pupil', opd[pup].mean() / a, expected=1, atol=tolerances['dm']['linearity']['rel'])
    report.assert_ok()


def test_poke_matches_independent_actuator_model(dm, nvalid, rng, report, tolerances):
    """
    OPD of a random command over the whole grid against the independent Gaussian / partition-of-unity model (see the
    module docstring), absolute amplitude in metres: max |OPD - model| / max |model| over the pupil grid.
    """
    X, Y = pupil_grid(N, D)
    c = rng.standard_normal(nvalid) * 1e-7
    got = shape(dm, c)
    ref = model_opd(c, X, Y, 11, D)
    report.check('max |OPD - model| / max |model|', np.max(np.abs(got - ref)) / np.max(np.abs(ref)), hi=tolerances['dm']['model']['rel'])
    report.assert_ok()


def _poke_index(valid_xy, pos):
    """Index (among the valid actuators) of the actuator closest to pos = (x, y) [m]."""
    return int(np.argmin(np.hypot(valid_xy[0] - pos[0], valid_xy[1] - pos[1])))


def test_single_actuator_surface_central_and_edge(logger, report, tolerances):
    """
    Surface of a single poked actuator (11x11 DM, c = 0.6, 61 px). Independent expectations:
      * the whole surface equals the model (Gaussian of width fixed by c, normalised by the sum over the valid
        actuators), absolute amplitude in metres, on the pupil samples;
      * interior actuator (infinite-lattice limit): the value at the actuator is 1 / (sum_i c^(i^2))^2 = 0.1626 a and the
        value at the n-th neighbour on an axis is c^(n^2) times it (mechanical coupling = neighbour / actuator ratio);
      * edge actuator (valid, on the border): the value at the actuator is 1 / sum over valid actuators of c^(d/p)^2.
    On the cell-centred pupil grid (D/N) only the central actuator falls on a sample (61 px, odd): its value is read from
    SAOS; the values at the other actuators (between samples, the edge one at D/2 beyond the last sample centre) are those of
    the model, which SAOS equals on every sample (first check).
    """
    n61 = 61
    tel61 = Telescope(resolution=n61, diameter=D, samplingTime=1 / 2000, logger=logger, pupil=circular_pupil(n61, D))
    d61 = DeformableMirror(telescope=tel61, nActs=11, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)
    X, Y = pupil_grid(n61, D)
    ax, ay, valid, pitch = actuators(11, D)
    vx, vy = ax[valid], ay[valid]
    a = 1e-6
    tol = tolerances['dm']['model']['rel']
    c0 = 0.6
    for name, pos in (('central', (0.0, 0.0)), ('edge (+x border)', (D / 2, 0.0)), ('inner ring', (2 * pitch, pitch))):
        k = _poke_index((vx, vy), pos)
        cmd = np.zeros(len(vx)); cmd[k] = a
        got = shape(d61, cmd)
        ref = model_opd(cmd, X, Y, 11, D)
        report.check(f'{name}: max |OPD - model| / peak', np.max(np.abs(got - ref)) / np.max(np.abs(ref)), hi=tol)
        at = lambda x, y: model_opd(cmd, np.array([[x]]), np.array([[y]]), 11, D)[0, 0]      # model at a point [m]
        if name == 'central':
            ic = (n61 - 1) // 2                                                              # sample at x = y = 0
            report.check('central: SAOS value at the actuator / a', got[ic, ic] / a, expected=lattice_peak(c0), rtol=1e-3)
            for step in (1, 2):
                report.check(f'central: model value {step} pitch away / value at the actuator', at(step * pitch, 0.0) / at(0.0, 0.0),
                             expected=c0 ** (step ** 2), rtol=1e-3)
            report.check('central: model diagonal neighbour / value at the actuator', at(pitch, pitch) / at(0.0, 0.0), expected=c0 ** 2, rtol=1e-3)
        if name.startswith('edge'):
            d2 = ((vx - vx[k]) ** 2 + (vy - vy[k]) ** 2) / pitch ** 2
            report.check('edge: model value at the actuator / a', at(vx[k], vy[k]) / a, expected=1.0 / np.sum(c0 ** d2), rtol=1e-9)
    report.assert_ok()


def test_regression_poke_superposition_and_scaling(dm, nvalid, rng, report, tolerances):
    """
    REGRESSION (linearity, equal-path): OPD(c1 + c2) = OPD(c1) + OPD(c2), OPD(k c) = k OPD(c), a full nActs^2 command
    equals the same command restricted to the valid actuators, and a poke reproduces the stored influence-function
    column (SAOS internal consistency; the physical IF is checked against the independent model above).
    """
    tol = tolerances['dm']['linearity']['rel']
    k = nvalid // 2
    poke = np.zeros(nvalid); poke[k] = 1.3e-7
    o = shape(dm, poke)
    col = dm.influence_functions[:, k].cpu().numpy().reshape(o.shape) * poke[k]
    report.check('poke vs stored IF column (max rel)', np.max(np.abs(o - col)) / np.max(np.abs(col)), hi=tol)
    c1, c2 = rng.standard_normal((2, nvalid)) * 1e-7
    o1, o2, o12 = shape(dm, c1), shape(dm, c2), shape(dm, c1 + c2)
    report.check('superposition (max rel)', np.max(np.abs(o12 - o1 - o2)) / np.max(np.abs(o12)), hi=tol)
    report.check('scaling x3 (max rel)', np.max(np.abs(shape(dm, 3 * c1) - 3 * o1)) / np.max(np.abs(3 * o1)), hi=tol)
    full = np.zeros(dm.validAct.shape[0]); full[dm.validAct] = c1
    report.check('full-grid command == valid-only command', np.max(np.abs(shape(dm, full) - o1)) / np.max(np.abs(o1)), hi=tol)
    report.assert_ok()


def test_dm_opd_enters_lightpath_with_positive_sign(dm, tel, nvalid, rng, logger, report, tolerances):
    """
    Transmissive convention: the OPD seen by the WFS/science path is +W(x) (no factor 2 for reflection), W the
    independent actuator model, masked by the pupil, and it adds to the atmosphere (OPD_wfs = (OPD_atm + W) P).
    """
    c = rng.standard_normal(nvalid) * 1e-7
    dm.updateDMShape(c, dynamicResponse=False)
    X, Y = pupil_grid(N, D)
    w = model_opd(c, X, Y, 11, D)
    atm_opd = rng.standard_normal((N, N)) * 1e-7
    atm = types.SimpleNamespace(getOPD=lambda src: atm_opd)
    lp = LightPath(logger)
    lp.initialize_path(src=src_at(0, 0), tel=tel, atm=atm, dm=[dm], delay=0)
    lp.propagate(temporal_tick=False)
    expected = (atm_opd + w) * circular_pupil(N, D)
    report.check('max |OPD_wfs - (OPD_atm + W) P| / max', np.max(np.abs(lp.wfs_opd - expected)) / np.max(np.abs(expected)),
                 hi=tolerances['dm']['model']['rel'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------------------------------------------------

def test_tilt_and_defocus_fitting(dm, tel, report, tolerances):
    """
    Low orders sampled on the actuators (c = a x_act for tilt, c = a r_act^2 for defocus, actuator coordinates of the
    independent model). (1) Over the whole pupil, OPD equals the independent actuator model (absolute, in metres).
    (2) The finite actuator model reproduces the targets a x and a r^2 (up to piston) inside 0.7 R with a relative rms
    error (same region for residual and signal). The fit quality is a property of the finite Gaussian actuator model
    (reported: inner region and edge annulus 0.7 R - R where the actuator support ends); SAOS must reproduce the model
    value in both regions.
    """
    tol = tolerances['dm']
    X, Y = pupil_grid(N, D)
    R = D / 2
    r = np.hypot(X, Y)
    pup = r <= R
    inner, edge = pup & (r < 0.7 * R), pup & (r >= 0.7 * R)
    ax, ay, valid, _ = actuators(11, D)
    xa, ya = ax[valid], ay[valid]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax_, (name, cmd, target) in zip(axes, (('tilt', 1e-6 * xa, 1e-6 * X), ('defocus', 1e-6 * (xa ** 2 + ya ** 2), 1e-6 * r ** 2))):
        got = shape(dm, cmd)
        ref = model_opd(cmd, X, Y, 11, D)
        report.check(f'{name}: max |OPD - model| / rms(model), whole pupil', np.max(np.abs((got - ref)[pup])) / np.std(ref[pup]),
                     hi=tol['model']['rel'])
        for label, reg in (('r < 0.7R', inner), ('0.7R < r < R', edge)):
            vals = []
            for opd in (got, ref):                                   # SAOS and independent model, same region for residual and signal
                rr = (opd - target)[reg]
                vals.append(np.std(rr - rr.mean()) / np.std(target[reg]))
            report.metric(f'{name}: model residual rms / target rms, {label}', vals[1])
            report.check(f'{name}: SAOS residual rms / target rms - model value, {label}', vals[0] - vals[1], expected=0, atol=tol['fitting']['ratio_atol'])
        im = ax_.imshow(np.where(pup, (got - target) / np.std(target[pup]), np.nan), origin='lower', cmap='RdBu_r')
        ax_.set_title(f'{name}: (SAOS - target) / target rms', fontsize=9)
        fig.colorbar(im, ax=ax_, fraction=0.046)
    report.figure(fig, 'fitting_residuals')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Altitude DMs and grid geometry
# ----------------------------------------------------------------------------------------------------------------------

ALT, NACT_ALT, FOV = 10000.0, 15, 60.0


@pytest.fixture(scope='module')
def dm_alt(tel, logger):
    """15x15 DM conjugated at 10 km (metapupil for the 60" telescope field)."""
    return DeformableMirror(telescope=tel, nActs=NACT_ALT, altitude=ALT, logger=logger, device='cpu', validActThreshpercentage=0.5)


# (azimuth [deg], displacement [grid steps]); fractions away from 0.5 so that the rounded offset is unambiguous, none rounding to 0
FOOTPRINT_CASES = [(0, 0.7), (0, 1.6), (90, 2.3), (180, 1.3), (270, 0.8)]


@pytest.mark.parametrize('az,disp', FOOTPRINT_CASES, ids=[f'az{a}-{d}px' for a, d in FOOTPRINT_CASES])
def test_altitude_dm_footprint_physical_displacement(az, disp, dm_alt, tel, report, tolerances):
    """
    A source at field angle theta and azimuth az sees the DM at altitude h through the pupil displaced by
    delta = h tan(theta) (cos az, sin az): azimuth 0 -> +x (columns), 90 -> +y (rows). Prescribed surface: a ramp
    command c = g (x_act + 0.7 y_act) (actuators of the independent model, metapupil D_fov = D + 2 h tan(fov/2)), whose exact surface
    W(x, y) is the independent model. Expectation for the change of the OPD relative to the on-axis one:
    dW = W(x_p + delta_r) - W(x_p) at the pupil points x_p. Integer-pixel model (as the atmosphere footprint): delta_r is delta rounded to the nearest grid step D/N (the pupil pixel), so |delta_r - delta| <= half a step, unbiased
    (truncation would give 0 for 0.7-0.8 steps). Fractional displacements 0.7-2.3 steps. Criterion: relative L2 error of the
    change over the pupil < yaml bound.
    """
    d_fov = D + 2 * ALT * np.tan(FOV / 2 * ARCSEC)
    ax, ay, valid, _ = actuators(NACT_ALT, d_fov)
    g = 2e-6
    cmd = g * (ax[valid] + 0.7 * ay[valid])
    dm_alt.updateDMShape(cmd, dynamicResponse=False)
    step = D / N
    delta = disp * step
    delta_r = np.round(disp) * step                        # integer-pixel model
    theta = np.arctan(delta / ALT) / ARCSEC
    pup = circular_pupil(N, D)
    on, _ = dm_alt.get_dm_opd(src_at(0.0, 0.0))
    off, _ = dm_alt.get_dm_opd(src_at(theta, az))
    X, Y = pupil_grid(N, D)
    ex, ey = np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az))
    w0 = model_opd(cmd, X, Y, NACT_ALT, d_fov)
    w1 = model_opd(cmd, X + delta_r * ex, Y + delta_r * ey, NACT_ALT, d_fov)
    dw_ref = (w1 - w0)[pup]
    dw_got = (off - on)[pup]
    err = np.linalg.norm(dw_got - dw_ref) / np.linalg.norm(dw_ref)
    report.metric('rounded displacement - physical [grid steps] (integer-pixel model, |.| <= 0.5)', (delta_r - delta) / step)
    report.metric('expected change: rms [m]', np.std(dw_ref))
    report.metric('SAOS change: rms [m]', np.std(dw_got))
    report.check(f'azimuth {az}, displacement {disp} steps: |dOPD - dW| / |dW|', err, hi=tolerances['dm']['footprint']['rel'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Layer geometry (documented formulas)
# ----------------------------------------------------------------------------------------------------------------------

def _layer(tel_like, altitude):
    """DM layer geometry from DeformableMirror.buildLayer without building influence functions."""
    ns = types.SimpleNamespace(logger=types.SimpleNamespace(debug=lambda *a, **k: None))
    return DeformableMirror.buildLayer(ns, tel_like, altitude)


EST_TEL = types.SimpleNamespace(D=4.149, resolution=216, fov=60.0, pupil=np.ones((216, 216), dtype=bool))


def test_altitude_dm_metapupil_diameter(report, tolerances):
    """D_fov = D + 2 h tan(fov/2) for the telescope field (EST: 4.149 m, 60", h = 20 km), relative error < 1e-6."""
    L = _layer(EST_TEL, 20000)
    expected = EST_TEL.D + 2 * 20000 * np.tan(EST_TEL.fov / 2 / 206265)
    report.check('D_fov / (D + 2 h tan(fov/2)) - 1', L.D_fov / expected - 1, expected=0, atol=tolerances['dm']['geometry']['metapupil_rel'])
    report.assert_ok()


def test_altitude_dm_grid_step_matches_ground(report, tolerances):
    """
    Every DM grid samples the same physical pixel as the pupil, the atmosphere and the WFS (D/N, cell centres): EST, h = 0, 5
    and 20 km. The grid covers the metapupil with a symmetric whole-pixel margin: 0 <= D_px p - D_fov < 2 p, D_px - N even.
    """
    pix = EST_TEL.D / EST_TEL.resolution
    for h in (0, 5000, 20000):
        L = _layer(EST_TEL, h)
        report.check(f'h = {h / 1e3:g} km: step / (D/N) - 1', L.pixel_size / pix - 1, expected=0, atol=tolerances['dm']['geometry']['grid_step_rel'])
        report.check(f'h = {h / 1e3:g} km: (D_px p - D_fov) / p (in [0, 2))', (L.D_px * pix - L.D_fov) / pix, lo=-1e-9, hi=2.0 - 1e-9)
        report.check(f'h = {h / 1e3:g} km: (D_px - N) mod 2', (L.D_px - EST_TEL.resolution) % 2, expected=0, atol=0)
    report.assert_ok()


def test_misregistration_pixel_size_matches_dm_grid(dm, tel, report, tolerances):
    """The metre-to-pixel conversion of MisRegistration must use the DM grid step (relative difference < 1e-3), measured on the grid."""
    step = dm.high_res_coords[1, 0] - dm.high_res_coords[0, 0]
    report.check('misReg pixel / DM grid step - 1', dm.misReg.spatialSampling / step - 1, expected=0,
                 atol=tolerances['dm']['geometry']['grid_step_rel'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------------------------------------------------

def test_dynamics_matches_dlsim(tel, logger, rng, tmp_path, report, tolerances):
    """
    A DM with a state-space dynamic model (second order, 800 Hz, damping 0.3, ZOH at the loop rate) responds to any
    command sequence exactly as scipy.signal.dlsim of the same model, actuator by actuator (max rel error < 1e-12),
    and the OPD is the independent actuator model applied to the dynamic output (model tolerance: SAOS adds 1e-12 to
    the partition-of-unity normalisation).
    """
    Ts = tel.samplingTime
    w, z = 2 * np.pi * 800, 0.3
    A, B, C, Dm, _ = signal.cont2discrete((np.array([[0, 1], [-w ** 2, -2 * z * w]]), np.array([[0], [w ** 2]]),
                                           np.array([[1, 0]]), np.array([[0]])), Ts, method='zoh')
    path = tmp_path / 'dm_dynamics.h5'
    with h5py.File(path, 'w') as f:
        for key, val in (('A', A), ('B', B.ravel()), ('C', C), ('D', Dm)):
            f.create_group(key).create_dataset('data', data=val)
        f['A'].attrs['Ts'] = Ts
    dmd = DeformableMirror(telescope=tel, nActs=7, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5,
                           dynamicModel=str(path))
    n_steps = 60
    nv = int(actuators(7, D)[2].sum())
    u = rng.standard_normal((n_steps, nv)) * 1e-7
    u[:20] = 1e-7                                                    # step, then random
    # Reference: dlsim actuator by actuator
    ref = np.stack([signal.dlsim((A, B, C, Dm, Ts), u[:, j])[1][:, 0] for j in range(nv)], axis=1)
    # Sample-by-sample dynamic output on a fresh state
    dmd.curr_state.zero_()
    yy = np.stack([dmd.applyDynamics(torch.as_tensor(u[k][:, None])).cpu().numpy()[:, 0] for k in range(n_steps)])
    report.check('applyDynamics vs dlsim (max rel)', np.max(np.abs(yy - ref)) / np.max(np.abs(ref)), hi=tolerances['dm']['dynamics']['rel'])
    # OPD of the last update equals the independent actuator model applied to the dlsim output
    dmd.curr_state.zero_()
    for k in range(n_steps):
        dmd.updateDMShape(u[k])
    X, Y = pupil_grid(N, D)
    opd_ref = model_opd(ref[-1], X, Y, 7, D)
    report.check('OPD after the sequence vs independent model of the dlsim output (max rel)',
                 np.max(np.abs(dmd.dm_layer.OPD - opd_ref)) / np.max(np.abs(opd_ref)), hi=tolerances['dm']['model']['rel'])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    t = np.arange(n_steps) * Ts * 1e3
    ax.step(t, u[:, 0] * 1e9, where='post', color='0.6', label='command')
    ax.plot(t, ref[:, 0] * 1e9, 'k-', label='dlsim')
    ax.plot(t, yy[:, 0] * 1e9, 'o', ms=3, mfc='none', label='SAOS applyDynamics')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('actuator 0 [nm]'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('DM dynamics: 2nd order 800 Hz, zeta 0.3, Ts 0.5 ms', fontsize=9)
    report.figure(fig, 'dynamics_step_response')
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Actuator-level misregistration
# ----------------------------------------------------------------------------------------------------------------------

MISREG_CASES = [('shift 0.3 pitch', 0.12, -0.08, 0, 1), ('magnification 0.95', 0, 0, 0, 0.95),
                ('rotation 7 deg', 0, 0, 7.0, 1), ('combined', 0.03, -0.02, 7.0, 1.02)]


@pytest.mark.parametrize('name,dx,dy,rot,s', MISREG_CASES, ids=[c[0] for c in MISREG_CASES])
def test_actuator_misregistration_matches_independent_model(name, dx, dy, rot, s, tel, nvalid, logger, rng, report, tolerances):
    """
    set_actuator_misregistration(shift, rotation, magnification) gives the DM shape of the independent model with its
    actuators moved to x' = s R x + d (R: rotation clockwise for a positive angle, x right / y up; Gaussian width scaled
    by s; partition of unity over the moved actuators), evaluated on the pupil grid, as documented. Resetting restores the
    original shape. Error relative to the rms of the unmisregistered surface.
    """
    d = DeformableMirror(telescope=tel, nActs=11, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)
    X, Y = pupil_grid(N, D)
    c = rng.standard_normal(nvalid) * 1e-7
    o0 = shape(d, c)
    d.set_actuator_misregistration(dx, dy, rot, s)
    o1 = shape(d, c)
    ref0 = model_opd(c, X, Y, 11, D)
    ref1 = model_opd(c, X, Y, 11, D, shift=(dx, dy), rot_deg=rot, scale=s)
    rms0 = np.std(ref0)
    report.check('max |OPD - model| / rms', np.max(np.abs(o1 - ref1)) / rms0, hi=tolerances['dm']['misregistration']['exact'])
    report.metric('shape change rms / rms (model)', np.std(ref1 - ref0) / rms0)
    d.set_actuator_misregistration(0, 0, 0, 1)
    report.check('reset: max |OPD - original model| / rms', np.max(np.abs(shape(d, c) - ref0)) / rms0, hi=tolerances['dm']['misregistration']['exact'])
    report.assert_ok()


def test_actuator_misregistration_moves_a_poke_as_specified(tel, logger, report, tolerances):
    """
    Geometric check of the sense of every misregistration parameter on a single poked actuator at (+2 pitch, 0): a
    shift (dx, dy) moves the surface centroid by (dx, dy), a +90 deg rotation (clockwise, x right / y up) takes it to
    (0, -2 pitch), a -90 deg rotation to (0, +2 pitch), and a magnification s to (2 s pitch, 0). Centroid of the surface
    peak region (values above half the maximum) from the pupil coordinates of this test, tolerance in pupil samples.
    """
    X, Y = pupil_grid(N, D)
    ax, ay, valid, pitch = actuators(11, D)
    k = _poke_index((ax[valid], ay[valid]), (2 * pitch, 0.0))
    cases = (('shift (+0.2 p, -0.1 p)', (0.2 * pitch, -0.1 * pitch, 0, 1), (2.2 * pitch, -0.1 * pitch)),
             ('rotation +90 deg', (0, 0, 90.0, 1), (0.0, -2 * pitch)), ('rotation -90 deg', (0, 0, -90.0, 1), (0.0, 2 * pitch)),
             ('magnification 1.1', (0, 0, 0, 1.1), (2.2 * pitch, 0.0)))
    for name, args, expected in cases:
        d = DeformableMirror(telescope=tel, nActs=11, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)
        cmd = np.zeros(d.nValidAct); cmd[k] = 1e-6
        d.set_actuator_misregistration(*args)
        o = shape(d, cmd)
        m = o > 0.5 * o.max()
        cx, cy = (X * o)[m].sum() / o[m].sum(), (Y * o)[m].sum() / o[m].sum()
        report.check(f'{name}: peak centroid x [m]', cx, expected=expected[0], atol=tolerances['dm']['misregistration']['centroid_m'])
        report.check(f'{name}: peak centroid y [m]', cy, expected=expected[1], atol=tolerances['dm']['misregistration']['centroid_m'])
    report.assert_ok()


CONVENTION_CASES = [('shift +1 px x', 1, 0, 0), ('shift +2 px y', 0, 2, 0), ('rotation +90 deg', 0, 0, 90.0), ('rotation -90 deg', 0, 0, -90.0)]


@pytest.mark.parametrize('name,dxp,dyp,rot', CONVENTION_CASES, ids=[c[0] for c in CONVENTION_CASES])
def test_regression_actuator_vs_map_misregistration_convention(name, dxp, dyp, rot, tel, nvalid, logger, rng, report, tolerances):
    """
    REGRESSION (equal-path, not a physics check; the physical sense is verified by the geometric test above): the
    actuator-level misregistration has the same sense as the OPD-map misregistration (MisRegistration.apply_misreg)
    where the map interpolation is exact (integer-pixel shifts, +-90 deg): interior difference < 1e-9 of the rms, and
    the opposite transformation differs by more than 10x. Shifts are given in DM grid pixels.
    """
    d = DeformableMirror(telescope=tel, nActs=11, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)
    c = rng.standard_normal(nvalid) * 1e-7
    c += 3e-7 * actuators(11, D)[0][actuators(11, D)[2]] / D          # asymmetric shape
    o0 = shape(d, c)
    rms0 = np.std(o0)
    step = D / N
    px = tel.pixelSize
    d.set_actuator_misregistration(dxp * step, dyp * step, rot, 1)
    o1 = shape(d, c)
    inner = (slice(10, -10), slice(10, -10))
    d.misReg.set_params(dxp * px, dyp * px, rot, 1)
    same = np.max(np.abs(o1 - d.misReg.apply_misreg(o0))[inner]) / rms0
    d.misReg.set_params(-dxp * px, -dyp * px, -rot, 1)
    opposite = np.max(np.abs(o1 - d.misReg.apply_misreg(o0))[inner]) / rms0
    d.misReg.set_params(0, 0, 0, 1)
    report.check('same sense: max interior |diff| / rms', same, hi=tolerances['dm']['misregistration']['convention'])
    report.check('opposite sense / same sense', opposite / max(same, 1e-300), lo=10)
    report.assert_ok()


def test_print_properties_runs(dm, logger, caplog):
    """print_properties logs the DM configuration and its map-level misregistration without errors."""
    import logging
    with caplog.at_level(logging.INFO, logger=logger.name):
        dm.print_properties()
    assert any('Controlled Actuators' in r.getMessage() for r in caplog.records)

