import torch
import numpy as np
import scipy.special as sp

# Returns a 2D matrix of size: (nValidActs x Nmodes) whose range is between -1 and 1.
# Based on: Milton & Lloyd-Hart, Disk Harmonic functions for adaptive optics simulations, Optics Society of America, 2005.

def generate_dh_modes(dm, nModes=None, useTorch=False, include_piston=False):
    # Read the main parameters of the DM: nActs and pupil mask
    pupil_mask = dm.validAct_2D
    nActs = dm.nValidAct
    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = nActs

    dh_modes = np.zeros((pupil_mask.shape[0], pupil_mask.shape[1], nActs))

    for i in range(nModes):
        print('Mode ', i)
        # Get the radial and azimuthal order
        if include_piston:
            n, m = get_index(i+1)
        else:
            n, m = get_index(i+2)
        # Begin the evaluation of the mode in the pupil --> the coordinates must be polar and defined in the unit circle
        coords = np.linspace(0, pupil_mask.shape[0]-1, pupil_mask.shape[0])
        x, y = np.meshgrid(coords, coords)
        r_max = np.max(np.sqrt((x-pupil_mask.shape[1]/2)**2 + (y-pupil_mask.shape[0]/2)**2)*pupil_mask)
        for j in range(pupil_mask.shape[0]): # Y-axis
            for k in range(pupil_mask.shape[1]): # X-axis
                if pupil_mask[j, k]:
                    # Move origin from top-left corner to the center of the image
                    origin_x = pupil_mask.shape[0] / 2
                    origin_y = pupil_mask.shape[1] / 2
                    # Compute the radius
                    r = np.sqrt((j-origin_y)**2+(k-origin_x)**2) / r_max # The radius is normalized to the unit circle
                    thetha = np.atan2(j-origin_y, k-origin_x)
                    dh_modes[j, k, i] = disk_harmonic_mode(m, n, r, thetha)
    # Normalize the mode between -1 and 1

    max_values = np.max(np.abs(dh_modes), axis=(0,1), keepdims=True)
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
    knm = sp.jn_zeros(mu, n)[-1]  # Approximate root using Bessel function zeros
    
    # Compute normalization factor anm
    anm = 1 / np.sqrt((1 - (m/knm)**2) * (sp.jv(m, knm)**2))
    
    # Compute radial function Rnm(r)
    Rnm = anm * sp.jv(m, knm * r)
    
    # Compute the Disk Harmonic function
    if m == 0:
        return Rnm
    elif m > 0:
        return np.sqrt(2) * Rnm * np.cos(m * theta)
    else:
        return np.sqrt(2) * Rnm * np.sin(mu * theta)

    