import torch
import numpy as np
import scipy.special as sp

# Returns a 2D matrix of size: (nValidActs x Nmodes) whose range is between -1 and 1.
# Based on: Milton & Lloyd-Hart, Disk Harmonic functions for adaptive optics simulations, Optics Society of America, 2005.

def generate_dh_modes(dm, nModes=None, useTorch=False, include_piston=False):

    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = dm.nValidAct

    # Compute radius and theta arrays for all points
    r = np.sqrt(dm.coordinates[:,0]**2 + dm.coordinates[:,1]**2)
    if np.any(dm.validAct):
        r /= np.max(r[dm.validAct])
    else:
        r /= np.max(r)
    # Clip to 1.0 to avoid ValueErrors outside the pupil
    r = np.clip(r, 0, 1.0)
    theta = np.arctan2(dm.coordinates[:,1], dm.coordinates[:,0])

    # Allocate output array
    dh_modes = np.zeros((dm.coordinates.shape[0], nModes))

    # Compute modes in a vectorized way
    for i in range(nModes):
        n, m = get_index(i+1) if include_piston else get_index(i+2)
        dh_modes[:, i] = disk_harmonic_mode(m, n, r, theta)  # Apply mask

        # Remove piston from modes other than the piston mode itself
        if not include_piston or i > 0:
            dh_modes[dm.validAct,i] -= np.mean(dh_modes[dm.validAct,i])

        # Normalize energy to 1 --> Necessary before Gram-Schmidt
        energy = np.sqrt(np.mean(dh_modes[dm.validAct, i]**2))
        if energy > 0:
            dh_modes[:,i] /= energy
    
    # Gram-Schmidt orthogonalization
    dh_modes_qr, _ =  np.linalg.qr(dh_modes[dm.validAct,:])

    dh_modes[dm.validAct, :]  = dh_modes_qr
    dh_modes[~dm.validAct, :] = 0.0

    # Normalize the mode between -1 and 1

    max_values = np.max(np.abs(dh_modes), axis=0, keepdims=True)
    max_values[max_values == 0] = 1
    dh_modes = dh_modes / max_values

    # Check torch option

    if useTorch:
        return torch.tensor(dh_modes, dtype=torch.float32)

    return dh_modes
        
# Get the radial and azimuthal order from the Noll index (>= 1)
def get_index(noll_index):
        """
        Return the radial order n and the azimuthal order m from the Noll index provided.
        """
        n = int((-1. + np.sqrt(8 * (noll_index - 1) + 1)) / 2.)
        p = (noll_index - (n * (n + 1)) / 2.)
        k = n % 2
        m = int((p + k) / 2.) * 2 - k

        if m != 0:
            if noll_index % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s

        return n, m

def disk_harmonic_mode(m ,n ,r, theta):
    """
    Computes the disk harmonic mode dnm(m, n, r, theta) using Bessel functions.
    
    Parameters:
        m (int): Azimuthal index (can be negative)
        n (int): Radial index (non-negative integer)
        r (numpy array): Radial coordinate (0 <= r <= 1)
        theta (numpy array): Azimuthal coordinate (same shape as r)
    
    Returns:
        numpy array: Disk harmonic mode evaluated at (r, theta)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if np.any(r < 0) or np.any(r > 1):
        raise ValueError("r must be within [0,1]")
    
    if n == 0 and m == 0:
        return np.ones_like(r)
    elif n == 0:
        raise ValueError("Only m=0 allowed when n=0")
    
    # Compute the radial eigenvalue knm (root of Bessel function derivative)
    mu = abs(m)
    q = (n-mu) // 2 if mu == 0 else (n-mu) //2 + 1 
    knm = sp.jnp_zeros(mu, q)[-1]  # Root of Bessel function derivative
    
    # Compute normalization factor anm (Milton & Lloyd-Hart, 2005)
    anm_den = np.sqrt(0.5 * (1 - (m/knm)**2) * (sp.jv(m, knm)**2))
    if anm_den == 0:
        anm = 1
    else:
        anm = 1 / anm_den
    
    # Compute radial function Rnm(r)
    Rnm = anm * sp.jv(m, knm * r)
    
    # Compute the Disk Harmonic function
    if m == 0:
        return Rnm
    elif m > 0:
        return np.sqrt(2) * Rnm * np.cos(m * theta)
    else:
        return np.sqrt(2) * Rnm * np.sin(mu * theta)

    