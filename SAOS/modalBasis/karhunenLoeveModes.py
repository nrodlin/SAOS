import numpy as np
import torch

from scipy.special import gamma

from SAOS.modalBasis.zernikeModes import ZernikeNaive, get_zernikes

# Code reused from the Phase Diversity repository of the EST project.
# An important point regarding this implementation is that the modes amplitude is not corrected by the atmosphere r0 nor the telescope diameter D. 
# If done, the ampltiude of the modes will match the stadistical variance of the modes, but cthe purpose of this function is to provide a modal basis 
# imited between -1 and 1 to be used with DMs and in the control loop. Hence, if the range of the modes will be cut afetrwards, there is not need to 
# correct the original amplitude by the atmosphere conditions.

def generate_kl_modes(dm, nModes=None, useTorch=False):

    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = dm.nValidAct

    kl_modes = np.zeros((dm.coordinates.shape[0], nModes))

    # Implementation based on Anzuloa and Gladysz (2017), original work from Roddier (1990) + Double Diagonalisation in Verinaud and Correia (2023)
    # Based on the python implementation of SPECULA: https://github.com/ArcetriAdaptiveOptics/SPECULA/blob/955a4840635b2d68ce766a0cc98ab706b88299e8/specula/lib/modal_base_generator.py

    # Get metric on how far from orthogonal are the Zernikes in the DM:
    Noffset = 2 # +1 due to Noll's index starting at 1, +1 due to the piston mode
    zernikes = get_zernikes(dm.coordinates, dm.validAct, nModes, Noffset)

    zernikes_qr, R =  np.linalg.qr(zernikes[dm.validAct,:])

    zernikes_ortho = np.zeros_like(zernikes)

    zernikes_ortho[dm.validAct,:] = zernikes_qr
    zernikes_ortho[~dm.validAct,:] = 0.0

    # Compute atmosphere Covariance Matrix

    Z_covmat = generate_covariance_matrix(nModes, Noffset-1)

    # Transform the CovMat to the Orthogonal space

    Rinv = np.linalg.inv(R)
    Z_covmat_ortho = R @ Z_covmat @ R.T

    # Now, do the second diagonalisation --> orthogonality w.r.t atmosphere

    evals, U = np.linalg.eigh(Z_covmat_ortho)

    # Sort by descending value
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    U = U[:, idx]

    # Transform eigenvector to the original space

    kl_modes = zernikes_ortho @ U

    kl_modes[~dm.validAct,:] = 0.0

       
    # Normalize between -1 and 1
    max_values = np.max(np.abs(kl_modes), axis=0, keepdims=True)
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

    # Precompute indices
    j_offset = np.empty(nModes, dtype=np.int32)
    n_arr    = np.empty(nModes, dtype=np.int32)
    m_arr    = np.empty(nModes, dtype=np.int32)

    for i in range(nModes):
        j = i+1
        j_offset[i] = j + Noffset
        n_arr[i], m_arr[i] = zernObj.zernIndex(j_offset[i])

    # Compute covariance using Symmetry:

    for i in range(nModes):
        ji = j_offset[i]
        ni = n_arr[i]
        mi = m_arr[i]

        Z_covmat[i, i] = compute_covariance(ji, ji, ni, ni, mi, mi)

        for k in range(i+1, nModes):
            jk = j_offset[k]
            nk = n_arr[k]
            mk = m_arr[k]

            val  = compute_covariance(ji, jk, ni, nk, mi, mk)
            Z_covmat[i, k] = val
            Z_covmat[k, i] = val

    return Z_covmat

