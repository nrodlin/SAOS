import numpy as np
import torch

# Entry point to the library, so as to simplify the usage and match the interface of the other modal bases

def generate_zernike_modes(dm, nModes=None, useTorch=False, include_piston=False):

    # If the modes are not specified, use every degree of freedom: nValidAct, minus piston when it is excluded
    if nModes is None:
        nModes = dm.nValidAct if include_piston else dm.nValidAct - 1

    if include_piston:
        Noffset = 1
    else:
        Noffset = 2
    
    # Get the Zernike modes
    
    zern_modes = get_zernikes(dm.coordinates, dm.validAct, nModes, Noffset)
    
    # Normalize between -1 and 1
    max_values = np.max(np.abs(zern_modes), axis=0, keepdims=True)
    max_values[max_values == 0] = 1
    zern_modes = zern_modes / max_values
    
    # Check torch option
    if useTorch:
        return torch.tensor(zern_modes, dtype=torch.float32)
    return zern_modes

# Generate zernikes, this process is split form the main call function to provide a better interface for the KL modal base
# that requires the generation of the full set of Zernikes, not normalized and without the pupil occlusion and spider.

def piston_free_orthonormal(Q):
    """
    Orthonormal, piston-free version of the orthonormal columns Q [nValidAct, nModes] (nModes <= nValidAct - 1): the piston
    is projected out and the columns are re-orthonormalised in order (Gram-Schmidt keeps the columns that were already
    piston-free, with their sign). Needed when the sampled modes are rank-deficient on the actuators: the QR then completes
    the basis with arbitrary directions that may contain piston.
    """
    Qp = Q - Q.mean(axis=0, keepdims=True)
    Q2, R2 = np.linalg.qr(Qp)
    return Q2 * np.where(np.diag(R2) < 0, -1.0, 1.0)


def get_zernikes(coordinates, validActs, nModes, Noffset, return_energy=False):
    """
    Noll-normalised Zernike modes j = Noffset ... Noffset + nModes - 1 sampled at the actuators, piston removed and rescaled to unit rms
    over the valid actuators (zern_modes[:, j] = Z_j / energy_j). With return_energy=True the rms energy_j of every mode before the
    rescaling is returned too: the coefficients on these columns are the Noll coefficients times energy_j.
    """

    zern_modes = np.zeros((coordinates.shape[0], nModes))
    energies = np.ones(nModes)

    Z_machine = ZernikeNaive(mask=[])

    rho = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
    # Normalize to unitary circle
    if np.any(validActs):
        rho /= np.max(rho[validActs])
    theta = np.arctan2(coordinates[:,1], coordinates[:,0])

    for j in range(nModes):
        # Jacobi form of the radial polynomial: the explicit factorial sum ('Standard') loses all precision above n ~ 40
        Z, norm_coeff = Z_machine.Z_nm(j + Noffset, rho, theta, True, 'Jacobi')
        Z = Z * norm_coeff                      # Noll normalisation (Z_nm returns the coefficient without applying it)
        Z[~validActs] = 0.0
        
        # Remove piston
        Z[validActs] -= np.mean(Z[validActs])
        
        # Normalize energy to 1
        energy = np.sqrt(np.mean(Z[validActs]**2))
        if energy > 0:
            Z /= energy
            energies[j] = energy
        
        # Apply the pupil mask
        zern_modes[:, j] = Z

    if return_energy:
        return zern_modes, energies
    return zern_modes

### --------------------------------------- ###
# -#                   ZERN                  #-#
### --------------------------------------- ###

"""
Python package for the evaluation of Zernike polynomials

Date: Jan 2018
Author: Alvaro Menduina Fernandez - University of Oxford
Email: alvaro.menduinafernandez@physics.ox.ac.uk
Version: 0.1
Description: evaluation of Zernike polynomials with the relation between Jacobi and Zernike radial polynomials and
their recurrence (stable for any order). The naive factorial sum, the Chong-Kintner recurrence, the series evaluator and
ZernikeSmart of the original package were removed from SAOS on 2026-10-27 (unused).
"""

class ZernikeNaive(object):
    def __init__(self, mask):
        """
        Evaluates single Zernike polynomials (Noll index) with the Jacobi form of the radial polynomial.
        """
        self.mask = mask

    def zernIndex(self, j):
        """
        Find the [n,m] list giving the radial order n and azimuthal order
        of the Zernike polynomial of Noll index j.
        Parameters:
            j (int): The Noll index for Zernike polynomials
        Returns:
            list: n, m values
        """
        n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
        p = (j - (n * (n + 1)) / 2.)
        k = n % 2
        m = int((p + k) / 2.) * 2 - k

        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s

        return [n, m]

    def R_nm_Jacobi(self, n, m, rho):
        """
        Computes the Radial Zernike polynomial of order 'n', 'm' R_nm
        but this version uses a method which is faster than the Naive R_nm.

        It exploits the relation between the Radial Zernike polynomial and Jacobi polynomials
            R_nm(rho) = (-1)^[(n-m)/2] * rho^|m| * J_{[(n-m)/2]}^{|m|, 0} (1 - 2*rho^2)

        In simpler terms, the R_nm polynomial evaluated at rho, is related to the J_{k}^{alfa, beta},
        the k-th Jacobi polynomial of orders {alfa, beta} evaluated at 1 - 2 rho^2,
        with k = (n-m)/2, alfa = |m|, beta = 0

        To calculate each Jacobi polynomial, it takes advantage of recurrence formulas
        """
        n, m = np.abs(n), np.abs(m)
        m_m = (n - m) / 2
        x = 1. - 2 * rho ** 2
        R = (-1) ** (m_m) * rho ** m * self.Jacobi(x, n=m_m, alfa=m, beta=0)
        return R

    def Jacobi(self, x, n, alfa, beta):
        """
        Returns the Jacobi polynomial J_{n}^{alfa, beta} (x)
        For the sake of efficiency and numerical stability it relies on a 3-term recurrence formula
        """
        J0 = np.ones_like(x)
        J1 = 0.5 * ((alfa - beta) + (alfa + beta + 2) * x)
        if n == 0:
            return J0
        if n == 1:
            return J1
        if n >= 2:
            J2 = None
            n_n = 2
            # Recurrence Relationship
            # a1n' * J_{n'+1} (x) = (a2n' + a3n' * x) * J_{n'} (x) - a4n' * J_{n'-1} (x)
            alfa_beta = alfa + beta
            while n_n <= n:
                # Update recurrence coefficients
                n2_alfa_beta = 2 * n_n + alfa_beta
                a1n = 2 * n_n * (n_n + alfa_beta) * (n2_alfa_beta - 2)
                a2n = (n2_alfa_beta - 1) * (x * n2_alfa_beta * (n2_alfa_beta - 2) + alfa ** 2 - beta ** 2)
                a3n = 2 * (n_n + alfa - 1) * (n_n + beta - 1) * n2_alfa_beta

                J2 = (a2n * J1 - a3n * J0) / a1n
                J0 = J1  # Update polynomials
                J1 = J2
                n_n += 1

            return J2

    def Z_nm(self, j, rho, theta, normalize_noll, mode):
        """
        Main function to evaluate a single Zernike polynomial of order 'n', 'm'

        The Noll normalisation coefficient is returned, not applied.

        :param rho: radial coordinate (ideally it should come normalized to 1)
        :param theta: azimuth coordinate
        :param normalize_noll: True {Applies Noll coefficient}, False {Does nothing}
        :param mode: 'Jacobi' (Jacobi-based recurrence, stable for any n), the only radial evaluator kept: the naive factorial
                sum ('Standard', unusable above n ~ 40) and 'ChongKintner' were removed on 2026-10-27
        """
        n, m = self.zernIndex(j)

        is_even = (j % 2 == 0)  # True if even, False otherwise

        if mode != 'Jacobi':
            raise ValueError(f"ZernikeNaive.Z_nm - unsupported radial mode '{mode}': only 'Jacobi' is available.")
        R = self.R_nm_Jacobi(n, m, rho)

        if m == 0:
            if n == 0:
                Z = np.ones_like(rho)
            else:
                Z = R
        elif is_even:
            Z = R * np.cos(m * theta)
        else:
            Z = R * np.sin(abs(m) * theta)        # Noll: odd j -> sin(|m| theta) (m < 0 only labels the sine term)
        if normalize_noll:
            if m == 0:
                norm_coeff = np.sqrt(n + 1)
            else:
                norm_coeff = np.sqrt(2) * np.sqrt(n + 1)
        else:
            norm_coeff = 1

        return Z, norm_coeff
