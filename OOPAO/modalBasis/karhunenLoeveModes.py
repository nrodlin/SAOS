import numpy as np
import torch

from scipy.special import gamma

from OOPAO.modalBasis.zernikeModes import ZernikeNaive, get_zernikes

# Code reused from the Phase Diversity repository of the EST project.
# An important point regarding this implementation is that the modes amplitude is not corrected by the atmosphere r0 nor the telescope diameter D. 
# If done, the ampltiude of the modes will match the stadistical variance of the modes, but cthe purpose of this function is to provide a modal basis 
# imited between -1 and 1 to be used with DMs and in the control loop. Hence, if the range of the modes will be cut afetrwards, there is not need to 
# correct the original amplitude by the atmosphere conditions.

def generate_kl_modes(dm, nModes=None, useTorch=False):
    # Read the main parameters of the DM: nActs and pupil mask
    pupil_mask = dm.validAct_2D
    nActs = dm.nValidAct

    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = nActs

    kl_modes = np.zeros((pupil_mask.shape[0], pupil_mask.shape[1], nModes))

    # Implementation based on Anzuloa and Gladysz (2017), original work from Roddier (1990)

    Noffset = 2 # +1 due to Noll's index starting at 1, +1 due to the piston mode

    Z_covmat = generate_covariance_matrix(nModes, Noffset)

    _, _, U = np.linalg.svd(Z_covmat)

    zernikes = get_zernikes(pupil_mask, nModes, Noffset)

    kl_modes = np.zeros_like(zernikes)

    # Correct each Zernike to de-correlate them and generate the KL modes
    for i in range(nModes):
        for j in range(nModes):
            kl_modes[:, :, i] += U[j, i] * zernikes[:, :, j]
    
    # Normalize between -1 and 1
    max_values = np.max(np.abs(kl_modes), axis=(0,1), keepdims=True)
    max_values[max_values == 0] = 1
    kl_modes = kl_modes / max_values
    
    # Check torch option
    if useTorch:
        return torch.tensor(kl_modes, dtype=torch.float32)
    return kl_modes

def compute_covariance(j, j_prima, n, n_prima, m, m_prima):
    # From: Atmospheric wavefront simulation using Zernike polynomials, Roddier (1990)
    parity = (j % 2 == 0) == (j_prima % 2 == 0)
    sz = (m == m_prima) and (parity or (m == 0))  # OK,
    # notice that the parity function in the paper returns 1 if odd and then negates the output,
    # this implementation applies directly the negation inside the function

    if sz:
        kzz_coef = gamma(14 / 3) * np.power((24 / 5) * gamma(6 / 5), 5 / 6) * np.power(gamma(11 / 6), 2) / (
                2 * np.power(np.pi, 2))  # OK
        kzz = kzz_coef * np.power(-1, (n + n_prima - 2 * m) / 2) * np.sqrt((n + 1) * (n_prima + 1))

        numerador = kzz * sz * gamma((n + n_prima - 5 / 3) / 2)
        denominador = gamma((n - n_prima + 17 / 3) / 2) * gamma((n_prima - n + 17 / 3) / 2) * gamma(
            (n + n_prima + 23 / 3) / 2)

        E = (numerador / denominador)
        return E
    else:
        return 0

def generate_covariance_matrix(nModes, Noffset=1):

    Z_covmat = np.zeros((nModes, nModes))

    zernObj = ZernikeNaive(mask=[])

    for j in range(1, Z_covmat.shape[0] + 1):
        for j_prima in range(1, Z_covmat.shape[1] + 1):
            n, m = zernObj.zernIndex(j + Noffset)
            n_prima, m_prima = zernObj.zernIndex(j_prima + Noffset)
            Z_covmat[j - 1, j_prima - 1] = compute_covariance(j + Noffset, j_prima + Noffset, n,
                                                                    n_prima, m, m_prima)

    return Z_covmat

