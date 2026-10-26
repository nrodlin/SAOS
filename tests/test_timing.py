"""
T1 validation of the loop timing: an impulse in the input OPD reaches the DM after exactly d + 1 frames for a light-path
delay d (SAOS convention: total pure delay d + 1, Controller delay = LightPath delay + 1).

Chain: atmosphere stand-in -> LightPath(delay d) -> Shack-Hartmann (6x6, 48 px) -> Controller (zernike IM, tikhonov, leaky)
-> DM (7x7). Per frame k: propagate(temporal_tick=True) (WFS measures frame k), computeControlAction (reads the
measurement of frame k - d), updateDMShape (the command shapes the DM seen by the WFS from frame k + 1 on). A one-frame
tilt injected at frame k0 therefore changes the first controller output at call k0 + d and the DM OPD in the frame
k0 + d + 1: latency d + 1 frames. The impulse response (the run with the impulse minus the run without it, which removes
the constant offset of the WFS) is exactly zero before.
Independent expectations: theoretical arrival frames, exact zero maps before arrival, DM OPD of the arrival frame from an
independent actuator model, sign and size of the correction (see test docstring).
"""
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SAOS.Controller import Controller
from SAOS.DeformableMirror import DeformableMirror
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.LightPath import LightPath
from SAOS.ShackHartmann import ShackHartmann
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.fast

RES, NSUB, D = 48, 6, 0.6
SUBAP = D / NSUB
COORD = np.arange(RES) - (RES - 1) / 2
X, _ = np.meshgrid(COORD, COORD)
K0, K = 4, 14                                   # impulse frame and number of frames
GAIN = 0.5                                      # controller gain of the loop
TILT_OPD = 2e-9                                 # OPD gradient of the impulse [m/px]: ~0.1 px of spot shift


@pytest.fixture(scope='module')
def chain(logger):
    """Telescope, source, WFS, DM and the IM (zernike) shared by the three delays; light paths are built per run."""
    tel = Telescope(resolution=RES, diameter=D, samplingTime=1e-3, pupil=np.ones((RES, RES), dtype=bool), centralObstruction=0.0, logger=logger)
    src = Source('R', 0, logger=logger)
    ps = src.wavelength / SUBAP * 206265 / 2.0
    wfs = ShackHartmann(nSubap=NSUB, telescope=tel, src=src, lightRatio=1.0, plate_scale=ps, fieldOfView=15 * ps, guardPx=1,
                        use_brightest=225, logger=logger, device='cpu')
    dm = DeformableMirror(telescope=tel, nActs=7, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, atm=None, dm=dm, wfs=wfs, delay=0)
    h = InteractionMatrixHandler(logger)
    h.initialize_im_class([lp])
    h.measure('zernike', 2e-8)
    return types.SimpleNamespace(tel=tel, src=src, wfs=wfs, dm=dm, handler=h)


def run(chain, logger, delay, impulse):
    """Closed loop of K frames; returns the controller output, the full DM OPD map seen by the WFS and the WFS slopes per frame."""
    box = types.SimpleNamespace(opd=np.zeros((RES, RES)))
    atm = types.SimpleNamespace(getOPD=lambda src: box.opd)
    lp = LightPath(logger)
    lp.initialize_path(src=chain.src, tel=chain.tel, atm=atm, dm=chain.dm, wfs=chain.wfs, delay=delay)
    chain.dm.updateDMShape(np.zeros(chain.dm.nValidAct), dynamicResponse=False)
    ctrl = Controller(telescope=chain.tel, interactionMatrix=chain.handler, reconstructionMethod='tikhonov', controllerType='leaky',
                      logger=logger, gain=[GAIN], decay=[0.9], nModes=12, delay=delay + 1)
    cmd, dm_opd, slopes = [], [], []
    for k in range(K):
        box.opd = TILT_OPD * X if (impulse and k == K0) else np.zeros((RES, RES))
        lp.propagate(temporal_tick=True)
        dm_opd.append(lp.dm_opd[0].copy())
        slopes.append(lp.slopes_1D.copy())
        out = ctrl.computeControlAction([lp])
        cmd.append(out[0].cpu().numpy().ravel())
        chain.dm.updateDMShape(out[0], dynamicResponse=False)
    return np.array(cmd), np.array(dm_opd), np.array(slopes)


def dm_model(cmd_full, nacts=7):
    """Independent documented DM (Gaussian, coupling 0.6, partition of unity) on the pupil grid; cmd_full = nacts^2 command."""
    xs = np.linspace(-D / 2, D / 2, RES)
    Xp, Yp = np.meshgrid(xs, xs)
    xa = np.linspace(-D / 2, D / 2, nacts)
    ax, ay = np.meshgrid(xa, xa)
    ax, ay = ax.ravel(), ay.ravel()
    pitch = D / (nacts - 1)
    valid = np.hypot(ax, ay) <= D / 2 + 0.5 * pitch
    r2 = (Xp.reshape(-1, 1) - ax[valid]) ** 2 + (Yp.reshape(-1, 1) - ay[valid]) ** 2
    g = np.exp(np.log(0.6) * r2 / pitch ** 2)
    return ((g @ cmd_full[valid]) / g.sum(axis=1)).reshape(RES, RES)


@pytest.mark.parametrize('delay', [0, 1, 2])
def test_impulse_latency_is_delay_plus_one(delay, chain, logger, report, tolerances):
    """
    Impulse (one-frame tilt) at frame k0, light-path delay d (Controller delay = d + 1). Theoretical arrival frames
    (from the loop definition, not measured): the WFS sees the impulse in frame k0; the controller output of call k0 + d
    is the first to change; the DM OPD in the WFS path first changes in frame k0 + d + 1 (latency d + 1 frames).
    Compared as full signed arrays, run with the impulse minus run without it:
      * slopes before frame k0, controller output before call k0 + d and DM OPD maps before frame k0 + d + 1: exactly 0;
      * slopes in frame k0, controller output at call k0 + d, DM OPD map at frame k0 + d + 1: non-zero;
      * the DM OPD map at frame k0 + d + 1 equals the independent DM model (documented Gaussian actuators) applied to
        the controller output of call k0 + d (the command shapes the DM seen from the next frame on, no dynamics);
      * the correction has the sign and size of the loop: pupil-mean x-gradient of that map = -gain x injected
        gradient (a leaky integrator of gain `gain` on a pure tilt that the 12 modes represent), within the yaml bound.
    """
    tol = tolerances['timing']['latency']
    c1, o1, s1 = run(chain, logger, delay, True)
    c0, o0, s0 = run(chain, logger, delay, False)
    dc, do, ds = c1 - c0, o1 - o0, s1 - s0
    k_cmd, k_dm = K0 + delay, K0 + delay + 1
    report.check('slopes before frame k0 (max |diff|, signed)', np.max(np.abs(ds[:K0])), expected=0, atol=tol['zero_atol'])
    report.check('slopes at frame k0 (max |diff|)', np.max(np.abs(ds[K0])), lo=tol['signal_min'])
    report.check('controller output before call k0 + d (max |diff|, signed)', np.max(np.abs(dc[:k_cmd])), expected=0, atol=tol['zero_atol'])
    report.check('controller output at call k0 + d (max |diff|)', np.max(np.abs(dc[k_cmd])), lo=tol['signal_min'])
    report.check('DM OPD maps before frame k0 + d + 1 (max |diff|, signed)', np.max(np.abs(do[:k_dm])), expected=0, atol=tol['zero_atol'])
    report.check('DM OPD map at frame k0 + d + 1 (max |diff|)', np.max(np.abs(do[k_dm])), lo=tol['signal_min'])
    exp_map = dm_model(dc[k_cmd])
    report.check('DM OPD at k0 + d + 1 vs independent DM model of the command of call k0 + d (max rel)',
                 np.max(np.abs(do[k_dm] - exp_map)) / np.max(np.abs(exp_map)), hi=tol['model_rel'])
    gx = np.gradient(do[k_dm], axis=1).mean()
    report.check('pupil-mean x-gradient of the correction / (-gain x injected gradient)', gx / (-GAIN * TILT_OPD), expected=1.0, atol=tol['gain_atol'])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    t = np.arange(K)
    nrm = lambda v: v / max(v.max(), 1e-300)
    ax.stem(t - 0.12, nrm(np.abs(ds).max(axis=1)), linefmt='C0-', markerfmt='C0o', basefmt=' ', label='WFS slopes')
    ax.stem(t, nrm(np.abs(dc).max(axis=1)), linefmt='C1-', markerfmt='C1s', basefmt=' ', label='controller output')
    ax.stem(t + 0.12, nrm(np.abs(do).max(axis=(1, 2))), linefmt='C2-', markerfmt='C2^', basefmt=' ', label='DM OPD in the WFS path')
    ax.axvline(K0, color='k', ls=':', lw=1)
    ax.set_xlabel('frame'); ax.set_ylabel('response to the impulse (normalised)'); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f'impulse at frame {K0} (dotted), light-path delay d = {delay}: expected DM latency {delay + 1} frames', fontsize=8)
    report.figure(fig, 'impulse_response')
    report.assert_ok()
