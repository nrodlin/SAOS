"""
T1 validation of SAOS.Detector: photon, dark and readout noise statistics, quantisation, saturation and reproducibility.

Detector.integrate(frame, photons) normalises `frame` to unit sum, scales it to `photons`, converts to electrons with QE
(Poisson(QE * N) with shot noise), adds dark current (Poisson) and readout noise (Gaussian), clips to the full well and
quantises to DN (electrons / quantization_conversion). Statistics are checked on uniform frames: every pixel is an
independent sample with the same expected value.
"""
import numpy as np
import pytest

from citools import known_issue
from SAOS.Detector import Detector

pytestmark = pytest.mark.fast

NPIX = 128          # 16384 independent pixels per frame


def make_detector(logger, **kw):
    params = dict(nPix=NPIX, samplingTime=1e-3, fullWellCapacity=1e9, nBits=24, quantumEfficiency=1.0, shotNoise=0,
                  darkCurrent=0, readoutNoise=0, gain=1, quantization_conversion=1, sensorType='CCD', darkCalibration=0,
                  noiseFlag=True, logger=logger, randomState=1234)
    params.update(kw)
    return Detector(**params)


def uniform_frames(det, photons_per_px, n_frames):
    flat = np.ones((NPIX, NPIX))
    return np.array([det.integrate(flat, photons_per_px * NPIX ** 2).astype(float) for _ in range(n_frames)])


def test_shot_noise_statistics(logger, report, tolerances):
    """
    Shot noise: detected electrons ~ Poisson(QE * N). On a uniform frame of N = 400 photons/px with QE = 0.6 the mean and
    the variance of the counts (1 e-/DN, no other noise) must both equal QE * N = 240 e-.
    """
    tol = tolerances['detector']['shot_noise']
    qe, n = 0.6, 400
    det = make_detector(logger, quantumEfficiency=qe, shotNoise=1)
    f = uniform_frames(det, n, 4)
    report.check('mean / (QE N)', f.mean() / (qe * n), 1.0, atol=tol['mean_rtol'], trend=True)
    report.check('variance / (QE N)', f.var() / (qe * n), 1.0, atol=tol['var_rtol'], trend=True)
    report.assert_ok()


def test_dark_current_statistics(logger, report, tolerances):
    """
    Dark current: Poisson(dark * t_int) electrons per pixel. 2000 e-/s/px over t_int = 10 ms gives mean = variance = 20 e-.
    """
    tol = tolerances['detector']['dark_current']
    det = make_detector(logger, darkCurrent=2000, integrationTime=0.01)
    f = uniform_frames(det, 0, 4)
    report.check('mean / (dark t)', f.mean() / 20.0, 1.0, atol=tol['mean_rtol'])
    report.check('variance / (dark t)', f.var() / 20.0, 1.0, atol=tol['var_rtol'])
    report.assert_ok()


def test_readout_noise_on_signal(logger, report, tolerances):
    """
    Readout noise on a bright uniform frame (no clipping at 0): variance = sigma_RON^2 + q^2/12 (quantisation of a
    dithered signal), here 25 + 1/12 e-^2 with q = 1 e-/DN.
    """
    tol = tolerances['detector']['readout_noise']
    det = make_detector(logger, readoutNoise=5.0)
    f = uniform_frames(det, 1000, 4)
    report.check('variance / (sigma^2 + q^2/12)', f.var() / (25 + 1 / 12), 1.0, atol=tol['var_rtol'])
    report.assert_ok()


@known_issue('DET-RON-CLIP', 'readout noise is clipped at 0 before quantisation (no bias pedestal) and the output is '
             'unsigned, so dark pixels lose the negative half of the noise', 'Detector.py:269-286')
def test_readout_noise_dark_frame(logger, report, tolerances):
    """
    Readout noise on a dark frame: after readout the signal must stay zero-mean with variance sigma^2 (a camera adds a
    bias pedestal before the ADC and removes it after, so negative excursions are kept). Clipping at 0 gives mean
    sigma/sqrt(2 pi) and variance sigma^2 (1/2 - 1/(2 pi)), which biases centroids towards the window centre.
    """
    tol = tolerances['detector']['readout_noise']
    sigma = 5.0
    det = make_detector(logger, readoutNoise=sigma, quantization_conversion=0.01)
    f = uniform_frames(det, 0, 4) * 0.01
    report.check('mean / sigma', f.mean() / sigma, 0.0, atol=tol['dark_mean_atol'])
    report.check('variance / sigma^2', f.var() / sigma ** 2, 1.0, atol=tol['var_rtol'])
    report.assert_ok()


@known_issue('DET-TRUNC', 'quantisation truncates (astype) instead of rounding: -0.5 DN bias', 'Detector.py:286')
def test_quantisation_unbiased(logger, report, tolerances):
    """
    Quantisation with q = 10 e-/DN of 1000 e- per pixel (readout noise 10 e- dithers the ADC): the mean output must be
    100 DN (rounding is unbiased); truncation gives 99.5 DN.
    """
    tol = tolerances['detector']['quantisation']
    det = make_detector(logger, readoutNoise=10.0, quantization_conversion=10)
    f = uniform_frames(det, 1000, 4)
    report.check('mean DN - e-/q', f.mean() - 100.0, 0.0, atol=tol['bias_atol_dn'], unit='DN')
    report.assert_ok()


@known_issue('DET-WRAP', 'dark-calibrated frames are cast to unsigned after subtracting the dark: negative values wrap '
             'to ~2^16', 'Detector.py:281-286')
def test_dark_calibration_no_wrap(logger, report, tolerances):
    """
    Dark calibration (20 frames) with readout noise 10 e-, 12 bits, 1 e-/DN, no signal: the calibrated output must stay
    within [0, 2^12 - 1] (or be signed) with mean ~ 0. Unsigned wrap of negative values produces counts near 65535.
    """
    tol = tolerances['detector']['dark_calibration']
    det = make_detector(logger, readoutNoise=10.0, nBits=12, darkCalibration=20, fullWellCapacity=60000)
    f = uniform_frames(det, 0, 2)
    report.check('max DN', f.max(), hi=2 ** 12 - 1, unit='DN')
    report.check('mean DN', f.mean(), 0.0, atol=tol['mean_atol_dn'], unit='DN')
    report.assert_ok()


def test_full_well_saturation(logger, report):
    """Saturation: 5000 e- per pixel on a 3000 e- full well must read exactly FWC / q = 3000 DN (q = 1)."""
    det = make_detector(logger, fullWellCapacity=3000)
    f = uniform_frames(det, 5000, 1)
    report.check('max DN', f.max(), 3000, atol=0, unit='DN')
    report.check('min DN', f.min(), 3000, atol=0, unit='DN')
    report.assert_ok()


@pytest.mark.parametrize('fwc', [np.inf, float('inf')], ids=['np.inf', 'float_inf'])
@known_issue('DET-FWC-INF', 'with an infinite full well and q = 0 every frame is normalised by its own maximum (np.inf), '
             'or the output is all zeros (float("inf") fails the `is np.inf` test)', 'Detector.py:263-268')
def test_infinite_full_well_scale(logger, report, fwc):
    """
    Infinite full well with automatic quantisation (q = 0): the output must keep the absolute signal scale, so a frame
    with twice the photons reads twice the counts (ratio 2) and is not zero.
    """
    det = make_detector(logger, fullWellCapacity=fwc, quantization_conversion=0, nBits=16)
    flat = np.ones((NPIX, NPIX))
    a = det.integrate(flat, 100 * NPIX ** 2).astype(float)
    b = det.integrate(flat, 200 * NPIX ** 2).astype(float)
    report.check('mean(2N) / mean(N)', b.mean() / max(a.mean(), 1e-30), 2.0, rtol=1e-3)
    report.assert_ok()


def test_seeded_reproducibility(logger, report):
    """Same randomState gives bit-identical noisy frames; a different seed gives different frames."""
    kw = dict(quantumEfficiency=0.7, shotNoise=1, readoutNoise=3.0, darkCurrent=1000)
    flat = np.ones((NPIX, NPIX))
    a = make_detector(logger, randomState=7, **kw).integrate(flat, 50 * NPIX ** 2)
    b = make_detector(logger, randomState=7, **kw).integrate(flat, 50 * NPIX ** 2)
    c = make_detector(logger, randomState=8, **kw).integrate(flat, 50 * NPIX ** 2)
    report.check('max |a - b| (same seed)', np.abs(a.astype(float) - b).max(), 0.0, atol=0)
    report.check('fraction of equal pixels (other seed)', np.mean(a == c), hi=0.5)
    report.assert_ok()


def test_noise_flag_off_is_identity(logger, report):
    """noiseFlag=False: integrate returns the input frame untouched (noise-free simulations)."""
    det = make_detector(logger, noiseFlag=False, shotNoise=1, readoutNoise=10.0)
    x = np.random.default_rng(0).random((NPIX, NPIX))
    report.check('max |out - in|', np.abs(det.integrate(x, 1e6) - x).max(), 0.0, atol=0)
    report.assert_ok()
