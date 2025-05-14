import numpy as np
import torch

def generate_zonal_modes(dm, nModes=None, useTorch=False):
    # Read the main parameters of the DM: nActs and pupil mask
    pupil_mask = dm.validAct_2D
    nActs = dm.nValidAct

    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = nActs

    zonal_modes = np.zeros((pupil_mask.shape[0], pupil_mask.shape[1], nModes))

    # Get the actuators positions
    
    rows, cols = np.where(pupil_mask == 1)
    
    # Assign each actuator to a layer of the 3D zonal modes matriz
    
    for i, (r, c) in enumerate(zip(rows, cols)):
        if i == nModes: # Rows and Cols dimension equal the number of actuators, although nModes can be smaller
            break
        zonal_modes[r, c, i] = 1    

    # Check torch option
    if useTorch:
        return torch.tensor(zonal_modes, dtype=torch.float32)
    return zonal_modes