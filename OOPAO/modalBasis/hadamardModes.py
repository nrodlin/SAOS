import torch
import numpy as np
import time
import cv2

# Returns a 2D matrix of size: (nValidActs x Nmodes) whose range is between -1 and 1.
def generate_hadamard_modes(dm, nModes=None, useTorch=False, include_piston=False):
    # Read the main parameters of the DM: nActs and pupil mask
    pupil_mask = dm.validAct_2D
    nActs = dm.nValidAct
    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = nActs
    # Hadamard modes are defined over a surface that is a power of 2 ==> we must compute the nearest power of 2 
    # Each row contains the exact informatiion of its equivalent column ==> row 0 == col 0
    # We are considering that: each row is a mode, and each column is an actuator.
    # If piston is discarded, we must assure the computation of an additional mode
    if include_piston:
        nHadamard = 2**np.ceil(np.log2(nActs)).astype(int)
    else:
        nHadamard = 2**np.ceil(np.log2(nActs+1)).astype(int)
    # Compute the modes
    t = time.time()
    H = hadamard(nHadamard)

    # Generate the 3D matrices, each 2D surface contains one Hadamard mode
    H_modes = np.zeros((pupil_mask.shape[0], pupil_mask.shape[1], nActs))

    indices = np.where(pupil_mask)  # Obtiene los índices donde el pupil_mask es True

    for i in range(nActs):
        if include_piston:
            H_modes[indices[0], indices[1], i] = H[i, :len(indices[0])]
        else:
            H_modes[indices[0], indices[1], i] = H[i+1, :len(indices[0])]

    if useTorch:
        modes_torch_output = torch.tensor(H_modes, dtype=torch.float32)
        return modes_torch_output
    else:
        return H_modes
    
def hadamard(n):
    """Efficient Hadamard matrix generation using Numba for parallel computation."""
    if (n & (n - 1)) != 0:  # Ensure n is a power of 2
        raise ValueError("N must be a power of 2")
    

    H = hadamard_recursive(n)

    return H.astype(np.float32)  # Convert to final type

def hadamard_recursive(n):
    if n==1:
        return np.array([[1]])
    else:
        H = hadamard_recursive(n//2)
        H = np.block([[H, H], [H, -H]])
        return H

