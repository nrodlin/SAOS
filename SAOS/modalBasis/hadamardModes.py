import torch
import numpy as np

# Returns a 3D matrix of size: (nActs x nActs x Nmodes) whose range is between -1 and 1.
def generate_hadamard_modes(dm, nModes=None, useTorch=False, include_piston=False):
    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = dm.nValidAct
    # Hadamard modes are defined over a surface that is a power of 2 ==> we must compute the nearest power of 2 
    # Each row contains the exact informatiion of its equivalent column ==> row 0 == col 0
    # We are considering that: each row is a mode, and each column is an actuator.
    # If piston is discarded, we must assure the computation of an additional mode
    if include_piston:
        nHadamard = 2**np.ceil(np.log2(dm.nValidAct)).astype(int)
    else:
        nHadamard = 2**np.ceil(np.log2(dm.nValidAct+1)).astype(int)
    # Compute the modes

    H = hadamard(nHadamard)

    # Generate the 3D matrices, each 2D surface contains one Hadamard mode
    H_modes = np.zeros((dm.coordinates.shape[0], nModes))

    # Hadamard modes must be generated using all the DoF, then, we can crop the number of modes
    for i in range(nModes):
        if include_piston:
            H_modes[dm.validAct, i] = H[i,:dm.nValidAct]
        else:
            H_modes[dm.validAct, i] = H[i+1,:dm.nValidAct]
            # Remove residual piston
            H_modes[dm.validAct, i] -= np.mean(H_modes[dm.validAct, i])

    # Normalize between -1 and 1
    max_values = np.max(np.abs(H_modes), axis=0, keepdims=True)
    max_values[max_values == 0] = 1
    H_modes = H_modes / max_values

    if useTorch:
        return torch.tensor(H_modes, dtype=torch.float32)
   
    return H_modes
    
def hadamard(n):
    # Recursive Hadamard matrix generation using Sylvester's Construction.
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
