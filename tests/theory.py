"""
Analytic references of the validation suite, written independently of SAOS (no SAOS imports).

Atmosphere: von Karman / Kolmogorov phase statistics (phase in rad at the wavelength of r0).
Optics: diffraction-limited and seeing-limited PSF widths.
Units: lengths in m, frequencies in cycles/m, angles in rad unless stated.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma, kv

ARCSEC = np.pi / 180 / 3600

# Kolmogorov constants
KOLMO_D = 2 * ((24 / 5) * gamma(6 / 5)) ** (5 / 6)                                                  # 6.8839
KOLMO_PSD = ((24 / 5) * gamma(6 / 5)) ** (5 / 6) * gamma(11 / 6) ** 2 / (2 * np.pi ** (11 / 3))   # 0.022896


def vk_psd(f, r0, L0):
    """von Karman phase PSD [rad^2 m^2] at spatial frequency f [cycles/m]."""
    return KOLMO_PSD * r0 ** (-5 / 3) * (np.asarray(f, dtype=float) ** 2 + 1 / L0 ** 2) ** (-11 / 6)


def vk_variance(r0, L0):
    """Phase variance of von Karman turbulence: C(0) = 0.0863 (L0/r0)^(5/3)."""
    return (L0 / r0) ** (5 / 3) * KOLMO_PSD * np.pi * 6 / 5      # = int 2 pi f PSD df


def vk_covariance(r, r0, L0):
    """von Karman phase covariance C(r) = C(0) 2^(1/6)/Gamma(5/6) x^(5/6) K_5/6(x), x = 2 pi r / L0."""
    r = np.asarray(r, dtype=float)
    x = 2 * np.pi * np.maximum(r, 1e-12) / L0
    return vk_variance(r0, L0) * 2 ** (1 / 6) / gamma(5 / 6) * x ** (5 / 6) * kv(5 / 6, x)


def vk_structure(r, r0, L0):
    """von Karman phase structure function D(r) = 2 [C(0) - C(r)]."""
    return 2 * (vk_variance(r0, L0) - vk_covariance(r, r0, L0))


def kolmogorov_structure(r, r0):
    """D(r) = 6.88 (r/r0)^(5/3)."""
    return KOLMO_D * (np.asarray(r, dtype=float) / r0) ** (5 / 3)


def r0_at(r0_500, wavelength):
    """r0 scaled from 500 nm to `wavelength` [m]: r0 propto lambda^(6/5)."""
    return r0_500 * (wavelength / 500e-9) ** (6 / 5)


def fit_r0_L0(r, d, r0_guess, L0_guess, fix_L0=False):
    """Least-squares fit (in log D) of r0 and L0 (or r0 only, L0 fixed) to a measured structure function."""
    if fix_L0:
        sol = least_squares(lambda p: np.log(vk_structure(r, np.exp(p[0]), L0_guess)) - np.log(d), [np.log(r0_guess)])
        return float(np.exp(sol.x[0])), float(L0_guess)
    sol = least_squares(lambda p: np.log(vk_structure(r, np.exp(p[0]), np.exp(p[1]))) - np.log(d),
                        [np.log(r0_guess), np.log(L0_guess)],
                        bounds=([np.log(1e-3), np.log(0.1)], [np.log(10), np.log(1e4)]))
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))


def layer_r0(r0, fraction, zenith_rad=0.0):
    """r0 of a layer holding `fraction` of the Cn2 integral, along the line of sight at zenith angle z."""
    return r0 * np.cos(zenith_rad) ** 0.6 * fraction ** (-3 / 5)


def seeing_fwhm(wavelength, r0_500, L0=np.inf):
    """Long-exposure seeing FWHM [rad]: 0.976 lambda/r0 sqrt(1 - 2.183 (r0/L0)^0.356) (Tokovinin 2002)."""
    r0 = r0_at(r0_500, wavelength)
    if np.isinf(L0):
        corr = 1.0
    else:
        # Validity domain: 0 < r0/L0 <= 0.1 (L0 >= 10 r0). The radicand 1 - 2.183 (r0/L0)^0.356 vanishes at
        # r0/L0 = 0.112 and the fit of Tokovinin (2002) is only meaningful for L0 >> r0.
        ratio = r0 / L0
        if not 0 < ratio <= 0.1:
            raise ValueError(f'seeing_fwhm: r0/L0 = {ratio:.3g} outside the validity range (0, 0.1] of the outer-scale '
                             f'correction (r0 = {r0:.4g} m, L0 = {L0:.4g} m)')
        corr = np.sqrt(1 - 2.183 * ratio ** 0.356)
    return 0.976 * wavelength / r0 * corr


def airy_fwhm(wavelength, D):
    """FWHM of the unobstructed Airy pattern [rad]: 1.029 lambda/D."""
    return 1.02899 * wavelength / D


def airy_first_zero(wavelength, D):
    """First dark ring of the unobstructed Airy pattern [rad]: 1.2197 lambda/D."""
    return 1.21967 * wavelength / D


def psf_reference(pupil, phase=None, oversampling=8):
    """
    PSF of a sampled pupil by zero-padded FFT, `oversampling` pixels per lambda/D (D = pupil array size), normalised
    to peak 1 for the unaberrated pupil. Used as numerical reference for obstructed pupils.
    """
    n = pupil.shape[0]
    N = int(round(oversampling * n))
    field = np.zeros((N, N), dtype=complex)
    ph = np.zeros_like(pupil, dtype=float) if phase is None else phase
    field[:n, :n] = pupil * np.exp(1j * ph)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return psf / (np.abs(pupil).sum() ** 2)
