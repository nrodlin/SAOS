"""
T2 validation of the parallel (joblib threads) and GPU execution of SAOS.

Equivalence tests (the only tests whose reference is SAOS itself, by policy documented here): the result of the
same configuration must not depend on how it is executed.
  * threads vs sequential: LightPaths propagated with joblib n_jobs = 1, 2, 8 give bit-identical results (noise off), and
    with detector noise and per-object seeds the results are reproducible across n_jobs;
  * CPU vs GPU: each module on identical inputs within the arithmetic precision of the computation (float64 for DM,
    ScienceCam and Controller; float32 inside the ShackHartmann cube and centroid). ShackHartmann, ScienceCam and
    DeformableMirror take `device`; Controller and InteractionMatrixHandler hard-code "cuda if available", so the CPU
    version is built with torch.cuda.is_available patched to False during construction (documented here).
Physical / resource criteria:
  * the VRAM peak of a fixed configuration is above the analytic size of its dominant tensor and below a recorded budget;
  * the number of threads alive during Atmosphere.getOPD inside threaded LightPaths stays within os.cpu_count().
"""
import os
import threading
import time
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from joblib import Parallel, delayed

from citools import known_issue
from SAOS.Atmosphere import Atmosphere
from SAOS.Controller import Controller
from SAOS.DeformableMirror import DeformableMirror
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.LightPath import LightPath
from SAOS.ScienceCam import ScienceCam
from SAOS.ShackHartmann import ShackHartmann
from SAOS.Source import Source
from SAOS.Telescope import Telescope

pytestmark = pytest.mark.nightly

RAD2ARCSEC = 206265
HAS_CUDA = torch.cuda.is_available()


# ----------------------------------------------------------------------------------------------------------------------
# Systems
# ----------------------------------------------------------------------------------------------------------------------

def make_wfs(logger, tel, src, device='cpu', nsub=6, samp=2.0, fov_px=12, **kw):
    sub = tel.D / nsub
    ps = src.wavelength / sub * RAD2ARCSEC / samp
    return ShackHartmann(nSubap=nsub, telescope=tel, src=src, lightRatio=1.0, plate_scale=ps, fieldOfView=fov_px * ps, guardPx=1,
                         use_brightest=fov_px ** 2, logger=logger, device=device, **kw)


def make_cam(logger, tel, src, device='cpu', fw=4.0, n_pix=None):
    n_pix = n_pix or tel.resolution
    ps = src.wavelength / tel.D * RAD2ARCSEC / fw
    return ScienceCam(fieldOfView=n_pix * ps, plate_scale=ps, samplingTime=tel.samplingTime, telescope=tel, logger=logger, device=device)


class Multi:
    """
    Telescope + 3-layer atmosphere + a shared ground DM + n LightPaths (one source direction, one WFS and one ScienceCam each).
    Everything is rebuilt with the same seeds for every execution mode.
    """

    def __init__(self, logger, n_paths=6, noise=False, seed=100, shared_wfs=False, res=48, D=0.6):
        self.tel = Telescope(resolution=res, diameter=D, samplingTime=1e-3, fov=40, logger=logger)
        self.atm = Atmosphere(r0=0.12, L0=25, windSpeed=[8, 15, 25], fractionalR0=[0.6, 0.25, 0.15], windDirection=[0, 70, 200],
                              altitude=[0, 4000, 11000], telescope=self.tel, logger=logger)
        self.atm.initializeAtmosphere(randomState=7)
        self.dm = DeformableMirror(telescope=self.tel, nActs=7, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)
        self.dm.updateDMShape(np.random.default_rng(1).standard_normal(self.dm.nValidAct) * 5e-8, dynamicResponse=False)
        self.lps = []
        shared = None
        for i in range(n_paths):
            src = Source('R', 5 if noise else 0, coordinates=[6.0 * (1 + i % 3), 360.0 * i / n_paths], logger=logger)
            if shared_wfs and shared is not None:
                wfs = shared
            else:
                wfs = make_wfs(logger, self.tel, src, noiseFlag=noise, randomState=seed + i, darkCalibration=0 if not noise else 5)
                shared = wfs
            cam = make_cam(logger, self.tel, src, n_pix=self.tel.resolution)
            lp = LightPath(logger)
            lp.initialize_path(src=src, tel=self.tel, atm=self.atm, dm=[self.dm], wfs=wfs, sci=cam)
            self.lps.append(lp)

    def run(self, n_jobs, steps=3):
        out = []
        for _ in range(steps):
            Parallel(n_jobs=n_jobs, prefer='threads')(delayed(lp.propagate)(True) for lp in self.lps)
            out.append([(lp.wfs_opd.copy(), lp.slopes_1D.copy(), lp.wfs_frame.copy(), lp.sci_frame.copy()) for lp in self.lps])
            self.atm.update()
        return out


def max_diff(a, b):
    """Largest absolute difference over all recorded arrays (steps x paths x 4 arrays)."""
    return max(float(np.max(np.abs(x - y))) for sa, sb in zip(a, b) for pa, pb in zip(sa, sb) for x, y in zip(pa, pb))


# ----------------------------------------------------------------------------------------------------------------------
# Threads vs sequential
# ----------------------------------------------------------------------------------------------------------------------

def test_threads_bit_identical_to_sequential_noise_off(logger, report, tolerances):
    """
    EQUIVALENCE TEST (SAOS vs SAOS, by design): six LightPaths sharing the atmosphere, the telescope and one DM, each with its
    own WFS and science camera, propagated for 3 steps with joblib threads n_jobs = 1, 2 and 8 (noise off): OPD, slopes,
    WFS frames and science frames are bit-identical.
    """
    tol = tolerances['parallel_gpu']['threads']['exact']
    res = {}
    for nj in (1, 2, 8):
        res[nj] = Multi(logger).run(nj)
    for nj in (2, 8):
        report.check(f'n_jobs={nj} vs 1: max |difference| over OPD, slopes, frames', max_diff(res[1], res[nj]), expected=0, atol=tol)
    report.metric('max |slope| [px] (signal present)', float(max(np.abs(p[1]).max() for s in res[1] for p in s)))
    report.assert_ok()


def test_threads_reproducible_with_per_object_seeds_and_noise(logger, report, tolerances):
    """
    EQUIVALENCE TEST (SAOS vs SAOS): with detector noise on and a distinct seed per WFS (seed = 100 + path), the results of n_jobs = 1 and
    n_jobs = 8 are bit-identical, and two runs with n_jobs = 8 coincide: the noise stream of a detector does not depend on the
    scheduling when each detector is used by one LightPath.
    """
    tol = tolerances['parallel_gpu']['threads']['exact']
    r1 = Multi(logger, noise=True).run(1)
    r8 = Multi(logger, noise=True).run(8)
    r8b = Multi(logger, noise=True).run(8)
    report.check('n_jobs=8 vs 1: max |difference|', max_diff(r1, r8), expected=0, atol=tol)
    report.check('n_jobs=8 run 2 vs run 1: max |difference|', max_diff(r8, r8b), expected=0, atol=tol)
    report.metric('mean WFS frame level [DN] (noise on)', float(np.mean([p[2].mean() for s in r1 for p in s])))
    report.assert_ok()


@known_issue('DET-RNG', 'a Detector owns a single numpy Generator (default_rng(seed)) that is used without a lock: LightPaths that share a WFS/Detector and run in '
             'threads draw the noise in scheduling order, so the results depend on n_jobs and are not reproducible', 'SAOS/Detector.py:143')
def test_threads_reproducible_with_shared_detector(logger, report, tolerances):
    """
    EQUIVALENCE TEST (SAOS vs SAOS): four LightPaths share one WFS (one Detector, one seed) and are propagated with n_jobs = 1
    and n_jobs = 8: a reproducible simulator gives the same noisy slopes for the same path, whatever the scheduling. Two
    runs with n_jobs = 8 must also coincide.
    """
    tol = tolerances['parallel_gpu']['threads']['exact']
    r1 = Multi(logger, n_paths=4, noise=True, shared_wfs=True).run(1)
    r8 = Multi(logger, n_paths=4, noise=True, shared_wfs=True).run(8)
    r8b = Multi(logger, n_paths=4, noise=True, shared_wfs=True).run(8)
    report.metric('n_jobs=8 vs 1: max |slope difference| [px]', max(float(np.max(np.abs(a[1] - b[1]))) for sa, sb in zip(r1, r8) for a, b in zip(sa, sb)), unit='px')
    report.check('n_jobs=8 vs 1: max |difference| (OPD, slopes, frames)', max_diff(r1, r8), expected=0, atol=tol)
    report.check('n_jobs=8 run 2 vs run 1: max |difference|', max_diff(r8, r8b), expected=0, atol=tol)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Thread count
# ----------------------------------------------------------------------------------------------------------------------

class ThreadSampler:
    """Samples threading.active_count() (and the OS thread count of the process, psutil) every millisecond."""

    def __init__(self):
        try:
            import psutil
            self.proc = psutil.Process()
        except ImportError:
            self.proc = None
        self.stop = False
        self.peak_py = 0
        self.peak_os = 0

    def _run(self):
        while not self.stop:
            self.peak_py = max(self.peak_py, threading.active_count())
            if self.proc is not None:
                self.peak_os = max(self.peak_os, self.proc.num_threads())
            time.sleep(0.001)

    def __enter__(self):
        self.base_py = threading.active_count()
        self.base_os = self.proc.num_threads() if self.proc else 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


@known_issue('ATM-NESTED', 'Atmosphere.getOPD opens its own joblib thread pool (up to 0.7 cpu_count threads, once per stage) inside every LightPath thread: '
             'the pools nest (outer n_jobs x inner n_jobs) and the thread count exceeds os.cpu_count()', 'SAOS/Atmosphere.py:410-422')
def test_thread_count_inside_threaded_lightpaths_within_cpu_count(logger, report, tolerances):
    """
    Resource criterion: 8 LightPaths of an extended (sun-like, 9 point directions) source are propagated with 8 joblib threads;
    each Atmosphere.getOPD parallelises its 9 directions. The worker threads created by the call (peak thread count - baseline,
    measured with threading.active_count and, as OS threads, psutil) must not exceed os.cpu_count().
    """
    ncpu = os.cpu_count()
    limit = tolerances['parallel_gpu']['threads_count']['max_threads'] or ncpu
    tel = Telescope(resolution=32, diameter=0.6, samplingTime=1e-3, fov=40, logger=logger)
    atm = Atmosphere(r0=0.12, L0=25, windSpeed=[8, 15, 25], fractionalR0=[0.6, 0.25, 0.15], windDirection=[0, 70, 200],
                     altitude=[0, 4000, 11000], telescope=tel, logger=logger)
    atm.initializeAtmosphere(randomState=7)
    stars = [Source('R', 0, coordinates=[6.0 * (1 + k % 3), 40.0 * k], logger=logger) for k in range(9)]
    sun = types.SimpleNamespace(tag='sun', subDirs_stars=stars, wavelength=stars[0].wavelength)
    atm.getOPD(sun)                                                     # warm-up (lazy initialisations)
    with ThreadSampler() as sampler:
        for _ in range(3):
            Parallel(n_jobs=8, prefer='threads')(delayed(atm.getOPD)(sun) for _ in range(8))
    extra_py = sampler.peak_py - sampler.base_py
    report.metric('os.cpu_count()', ncpu)
    report.metric('baseline Python threads', sampler.base_py)
    report.check('Python worker threads alive during getOPD (peak - baseline)', extra_py, hi=limit)
    if sampler.proc is not None:
        report.metric('baseline OS threads', sampler.base_os)
        report.check('OS threads alive during getOPD (peak - baseline)', sampler.peak_os - sampler.base_os, hi=limit)
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# CPU vs GPU
# ----------------------------------------------------------------------------------------------------------------------

def smooth_phase(res, rng, rms=1.0):
    f = np.fft.fftfreq(res) * res
    F = np.hypot(*np.meshgrid(f, f))
    amp = np.where(F > 0, np.maximum(F, 1e-9) ** (-11 / 6), 0.0)
    ph = np.real(np.fft.ifft2(amp * (rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res)))))
    return ph / ph.std() * rms


@pytest.mark.gpu
def test_gpu_matches_cpu_shackhartmann(logger, rng, report, tolerances):
    """
    EQUIVALENCE TEST (CPU vs GPU): the same phase (1 rad rms, Kolmogorov-like) through ShackHartmann(device='cpu') and
    ShackHartmann(device='cuda'), noise off. The cube of subaperture fields is complex64 (ShackHartmann.py:465) and the
    centroid float32, so the agreement is at float32 precision: frame relative to its peak, slopes in px.
    """
    tol = tolerances['parallel_gpu']['cpu_gpu']
    tel = Telescope(resolution=48, diameter=0.6, samplingTime=1e-3, logger=logger)
    src = Source('R', 0, logger=logger)
    wc, wg = make_wfs(logger, tel, src, 'cpu'), make_wfs(logger, tel, src, 'cuda')
    phase = smooth_phase(48, rng, 1.0) * tel.pupil
    sc, _, fc = wc.wfs_measure(phase, src)
    sg, _, fg = wg.wfs_measure(phase, src)
    report.check('slopes: max |cpu - gpu| [px]', np.max(np.abs(sc - sg)), hi=tol['shackhartmann_slope_px'], unit='px')
    report.check('frame: max |cpu - gpu| / peak', np.max(np.abs(fc - fg)) / np.max(fc), hi=tol['shackhartmann_frame_rel'])
    report.metric('max |slope| [px]', np.max(np.abs(sc)))
    report.assert_ok()


@pytest.mark.gpu
def test_gpu_matches_cpu_sciencecam(logger, rng, report, tolerances):
    """EQUIVALENCE TEST (CPU vs GPU): ScienceCam.get_frame on a 2 rad rms phase, complex128 FFT: relative to the peak."""
    tol = tolerances['parallel_gpu']['cpu_gpu']['sciencecam_rel']
    tel = Telescope(resolution=64, diameter=1.0, samplingTime=1e-3, logger=logger)
    src = Source('R', 5, logger=logger)
    cc, cg = make_cam(logger, tel, src, 'cpu', n_pix=64), make_cam(logger, tel, src, 'cuda', n_pix=64)
    phase = smooth_phase(64, rng, 2.0) * tel.pupil
    fc = np.asarray(cc.get_frame(src, phase).detach().cpu() if torch.is_tensor(cc.get_frame(src, phase)) else cc.get_frame(src, phase))
    fg = cg.get_frame(src, phase)
    fg = np.asarray(fg.detach().cpu() if torch.is_tensor(fg) else fg)
    report.check('frame: max |cpu - gpu| / peak', np.max(np.abs(fc - fg)) / np.max(fc), hi=tol)
    report.assert_ok()


@pytest.mark.gpu
def test_gpu_matches_cpu_deformablemirror(logger, rng, report, tolerances):
    """EQUIVALENCE TEST (CPU vs GPU): DeformableMirror OPD of a random command, ground and 4 km layers (footprint shift), float64."""
    tol = tolerances['parallel_gpu']['cpu_gpu']['dm_rel']
    tel = Telescope(resolution=48, diameter=0.6, samplingTime=1e-3, fov=40, logger=logger)
    src = Source('R', 0, coordinates=[15.0, 30.0], logger=logger)
    for alt in (0.0, 4000.0):
        dc = DeformableMirror(telescope=tel, nActs=9, altitude=alt, logger=logger, device='cpu', validActThreshpercentage=0.5)
        dg = DeformableMirror(telescope=tel, nActs=9, altitude=alt, logger=logger, device='cuda', validActThreshpercentage=0.5)
        cmd = rng.standard_normal(dc.nValidAct) * 1e-7
        dc.updateDMShape(cmd, dynamicResponse=False)
        dg.updateDMShape(cmd, dynamicResponse=False)
        oc, pc = dc.get_dm_opd(src)
        og, pg = dg.get_dm_opd(src)
        oc, og = np.asarray(oc), np.asarray(og)
        report.check(f'altitude {alt:.0f} m: max |OPD cpu - gpu| / max', np.max(np.abs(oc - og)) / np.max(np.abs(oc)), hi=tol)
    report.assert_ok()


class MockLP:
    def __init__(self, n):
        self.s = np.zeros(n)

    def get_wavefront_error(self):
        return self.s


def build_loop(logger, cpu):
    """Single-WFS / single-DM system and its IM; cpu=True patches torch.cuda.is_available to False while SAOS reads the device."""
    orig = torch.cuda.is_available
    if cpu:
        torch.cuda.is_available = lambda: False
    try:
        dev = 'cpu' if cpu else 'cuda'
        tel = Telescope(resolution=48, diameter=0.6, samplingTime=1e-3, pupil=np.ones((48, 48), dtype=bool), logger=logger)
        src = Source('R', 0, logger=logger)
        wfs = make_wfs(logger, tel, src, dev, nsub=6, samp=2.0, fov_px=15)
        dm = DeformableMirror(telescope=tel, nActs=7, altitude=0, logger=logger, device=dev, validActThreshpercentage=0.5)
        lp = LightPath(logger)
        lp.initialize_path(src=src, tel=tel, dm=dm, wfs=wfs)
        h = InteractionMatrixHandler(logger)
        h.initialize_im_class([lp])
        h.measure('zernike', 2e-8)
        return tel, h
    finally:
        torch.cuda.is_available = orig


@pytest.mark.gpu
def test_gpu_matches_cpu_interaction_matrix_and_controller(logger, rng, report, tolerances):
    """
    EQUIVALENCE TEST (CPU vs GPU). InteractionMatrixHandler and Controller hard-code `cuda if available`: the CPU system is built
    with torch.cuda.is_available patched to False (WFS and DM with device='cpu'), the GPU system normally (device='cuda').
    (i) The measured zernike IMs agree to float32 precision (the WFS slopes are float32).
    (ii) Controllers (inversion, leaky, 12 modes) built on the SAME IM, one per device, driven by the same 10 slope vectors give
    the same modal and DM commands (float64).
    """
    tol = tolerances['parallel_gpu']['cpu_gpu']
    tel_c, hc = build_loop(logger, cpu=True)
    tel_g, hg = build_loop(logger, cpu=False)
    imc = hc.interaction_matrix_warehouse[0][0]['IM']
    img = hg.interaction_matrix_warehouse[0][0]['IM']
    stroke = 2e-8
    report.metric('IM: max |cpu - gpu| / max', np.max(np.abs(imc - img)) / np.max(np.abs(imc)))
    report.check('IM: max |cpu - gpu| x stroke [px] (slope-level difference)', np.max(np.abs(imc - img)) * stroke, hi=tol['im_slope_px'], unit='px')
    kw = dict(telescope=tel_c, interactionMatrix=hc, reconstructionMethod='inversion', controllerType='leaky', operationType='closed',
              logger=logger, gain=[0.4], decay=[0.99], ki=[0.0], rcond=1e-3, nModes=12, delay=1)
    orig = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        cc = Controller(**kw)
    finally:
        torch.cuda.is_available = orig
    cg = Controller(**kw)
    report.note(f'controller devices: cpu={cc.device}, gpu={cg.device}')
    assert cc.device.type == 'cpu' and cg.device.type == 'cuda'
    n = imc.shape[0]
    lc, lg = MockLP(n), MockLP(n)
    mc, mg, dc_, dg_ = [], [], [], []
    for _ in range(10):
        lc.s = lg.s = rng.standard_normal(n) * 0.05
        dcm = cc.computeControlAction([lc])
        dgm = cg.computeControlAction([lg])
        mc.append(cc.command_previous[0].cpu().numpy().ravel())
        mg.append(cg.command_previous[0].cpu().numpy().ravel())
        dc_.append(dcm[0].cpu().numpy().ravel())
        dg_.append(dgm[0].cpu().numpy().ravel())
    mc, mg, dc_, dg_ = map(np.array, (mc, mg, dc_, dg_))
    report.check('modal command: max |cpu - gpu| / max', np.max(np.abs(mc - mg)) / np.max(np.abs(mc)), hi=tol['controller_rel'])
    report.check('DM command: max |cpu - gpu| / max', np.max(np.abs(dc_ - dg_)) / np.max(np.abs(dc_)), hi=tol['controller_rel'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# VRAM
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.gpu
def test_vram_peak_within_budget(logger, rng, report, tolerances):
    """
    Fixed small configuration on the GPU (64 px pupil, 8x8 SH with 2x oversampled 16 px windows, 9x9 DM, 64 px science camera), 5 steps
    through a LightPath: torch.cuda.max_memory_allocated after the steps is (i) larger than the analytic size of the dominant
    tensor, the complex64 field cube of the SH (n_subap^2 nFFT^2 x 8 B), and (ii) below the recorded budget (regression guard:
    a leak or a per-step copy). The recorded budget is yaml vram.budget_mib x headroom.
    """
    tol = tolerances['parallel_gpu']['vram']
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    tel = Telescope(resolution=64, diameter=0.8, samplingTime=1e-3, fov=20, logger=logger)
    src = Source('R', 0, logger=logger)
    wfs = make_wfs(logger, tel, src, 'cuda', nsub=8, samp=2.0, fov_px=16)
    cam = make_cam(logger, tel, src, 'cuda', n_pix=64)
    dm = DeformableMirror(telescope=tel, nActs=9, altitude=0, logger=logger, device='cuda', validActThreshpercentage=0.5)
    atm = Atmosphere(r0=0.12, L0=25, windSpeed=[8, 15], fractionalR0=[0.7, 0.3], windDirection=[0, 90], altitude=[0, 5000], telescope=tel, logger=logger)
    atm.initializeAtmosphere(randomState=3)
    lp = LightPath(logger)
    lp.initialize_path(src=src, tel=tel, atm=atm, dm=[dm], wfs=wfs, sci=cam)
    for _ in range(5):
        lp.propagate(True)
        atm.update()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    nfft = int(np.round(src.wavelength * RAD2ARCSEC / (wfs.subaperture_size * wfs.plate_scale) * wfs.npix_lenslet))
    cube = wfs.nSubap ** 2 * nfft ** 2 * 8
    mib = peak / 2 ** 20
    report.metric('peak allocated [MiB] (max_memory_allocated - baseline)', mib)
    report.metric('analytic SH cube (complex64) [MiB]', cube / 2 ** 20)
    report.metric('allocated at the end [MiB]', (torch.cuda.memory_allocated() - base) / 2 ** 20)
    report.check('peak >= analytic SH field cube [MiB]', mib, lo=cube / 2 ** 20)
    report.check('peak below recorded budget x headroom [MiB]', mib, hi=tol['budget_mib'] * tol['headroom'])
    report.assert_ok()
