import numpy as np
import torch
import h5py

from scipy.special import gamma

import logging
import logging.handlers
from queue import Queue

import time

import matplotlib.pyplot as plt

class LearnEstimator:
    """
    Estimate atmospheric parameters from WFS telemetry.
    """

    def __init__(self, 
                 altitudes, 
                 zenith,
                 measLPs,
                 targetLPs,
                 logger=None):

        # shared logging setup
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        self.tag       = 'learnEstimator'

        self.altitudes = np.array(altitudes)
        self.nLayers   = len(self.altitudes)

        self.zenith    = zenith

        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

        self.measLPs = measLPs
        self.targetLPs = targetLPs

        # Correct altitudes by zenith
        self.altitudes = self.altitudes / np.cos((self.zenith/180)*np.pi)       
        
        # Create the WFS structure
        
        self.measWFSparams   = self.initializeWFS(measLPs)
        self.targetWFSparams = self.initializeWFS(targetLPs)

        # Generate midpoints

        self.measWFSmidpoint_X, self.measWFSmidpoint_Y     = self.generate_midpoints_coords(self.measWFSparams)
        self.targetWFSmidpoint_X, self.targetWFSmidpoint_Y = self.generate_midpoints_coords(self.targetWFSparams)

        self.logger.info('LearnEstimator::__init__ - Initialization completed.')

    def initializeWFS(self, lps):
        # Scan each LP to check that there is a WFS
        # If there is a WFS, then store its properties -> valid map, physical coordinates

        wfs_params = []

        for i in range(len(lps)):
            # Input check
            if lps[i].wfs is None:
                raise ValueError(f'LearnEstimator::initializeMeasureWFS - LP {i} does not have a WFS.')

            # Save params of each WFS
            wfs_dict = {}

            wfs_dict['subap_size']  = lps[i].wfs.subaperture_size
            wfs_dict['plate_scale'] = lps[i].wfs.plate_scale
            wfs_dict['wavelength']  = lps[i].src.wavelength
            
             # Compute the coordinates per layer
            x = np.linspace(-lps[i].tel.D / 2 + lps[i].wfs.subaperture_size / 2, lps[i].tel.D / 2 - lps[i].wfs.subaperture_size / 2, lps[i].wfs.nSubap)
            y = np.linspace(-lps[i].tel.D / 2 + lps[i].wfs.subaperture_size / 2, lps[i].tel.D / 2 - lps[i].wfs.subaperture_size / 2, lps[i].wfs.nSubap)

            xx, yy = np.meshgrid(x, y)

            # Create list to store the coordinates of the subaps center per layer
            wfs_coordinates = [] 

            for j in range(self.nLayers):
                # Compute grid origin at the layer
                origin_x_arcsec = lps[i].src.coordinates[0] * np.cos(lps[i].src.coordinates[1] * (np.pi/180))
                origin_y_arcsec = lps[i].src.coordinates[0] * np.sin(lps[i].src.coordinates[1] * (np.pi/180))

                origin_x = self.altitudes[j] * (origin_x_arcsec / 206265.) # in meters
                origin_y = self.altitudes[j] * (origin_y_arcsec / 206265.) # in meters

                # Move grid
                layer_x_map = origin_x + xx
                layer_y_map = origin_y + yy

                # Select valid subaps
                layer_x_map_valid = layer_x_map[lps[i].wfs.valid_subapertures]
                layer_y_map_valid = layer_y_map[lps[i].wfs.valid_subapertures]

                # Concat arrays
                wfs_coordinates.append(np.column_stack((layer_x_map_valid, layer_y_map_valid)))
            
            # Store the coordinates in the WFS params dictionary
            wfs_dict['coordinates_per_layer'] =  np.array(wfs_coordinates)

            # Append dictionary to list
            wfs_params.append(wfs_dict.copy())

        # Return the parameters as a list of size nWFS
        return wfs_params

    # CGenerate midpoints to compute separations

    def generate_midpoints_coords(self, wfs_params):

        # Gradient X
        wfs_midpoints_gradientX = []

        wfs_midpoints_dict = {}

        for i in range(len(wfs_params)):
            wfs_midpoints_dict['midPointA']  = wfs_params[i]['coordinates_per_layer'].copy()
            wfs_midpoints_dict['midPointa']  = wfs_params[i]['coordinates_per_layer'].copy()

            wfs_midpoints_dict['midPointA'][:,:,0] += wfs_params[i]['subap_size']/2
            wfs_midpoints_dict['midPointa'][:,:,0] -= wfs_params[i]['subap_size']/2

            wfs_midpoints_gradientX.append(wfs_midpoints_dict.copy())

        # Gradient Y
        wfs_midpoints_gradientY = []

        wfs_midpoints_dict = {}

        for i in range(len(wfs_params)):
            wfs_midpoints_dict['midPointA']  = wfs_params[i]['coordinates_per_layer'].copy()
            wfs_midpoints_dict['midPointa']  = wfs_params[i]['coordinates_per_layer'].copy()

            wfs_midpoints_dict['midPointA'][:,:,1] += wfs_params[i]['subap_size']/2
            wfs_midpoints_dict['midPointa'][:,:,1] -= wfs_params[i]['subap_size']/2

            wfs_midpoints_gradientY.append(wfs_midpoints_dict.copy())
        
        return wfs_midpoints_gradientX, wfs_midpoints_gradientY
    
    # Structure function

    def prepare_vk_constants_torch(self, dtype):
        return {
            "k1": torch.tensor(gamma(11/6) * (2**(1/6)) * (np.pi**(-8/3)) *(((24/5) * gamma(6/5))**(5/6)), dtype=dtype, device=self.device),
            "a0": torch.tensor(gamma(-5/6) / (2**(1/6)), dtype=dtype, device=self.device),
            "b0": torch.tensor(gamma(5/6) / (2**(1/6)), dtype=dtype, device=self.device),
        }    

    def vk_structure_torch(self, r0, L0, separation, constants, nMax=10):
        Dphi = torch.zeros_like(separation)

        valid = separation > 0
        if not torch.any(valid):
            return Dphi

        r = separation[valid]

        x = torch.pi * (r / L0)
        y = x ** (5.0 / 3.0)

        k1 = constants["k1"]
        a0 = constants["a0"]
        b0 = constants["b0"]

        X1 = x ** 2

        recursive_term = torch.zeros_like(r)
        aprev = a0
        bprev = b0
        Xprev = torch.ones_like(r)

        for i in range(nMax):
            n = i + 1
            an = aprev / (n * (n + 5.0 / 6.0))
            bn = bprev / (n * (n - 5.0 / 6.0))
            Xn = Xprev * X1

            recursive_term = recursive_term + (an * y + bn) * Xn

            aprev = an
            bprev = bn
            Xprev = Xn

        Dphi[valid] = -k1 * (r0 ** (-5.0 / 3.0)) * (L0 ** (5.0 / 3.0)) * (
            a0 * y + recursive_term
        )

        return Dphi
    
    # Compute separations

    def separations_to_torch(self, AB, Ab, aB, ab, dtype=torch.float64):
        sep_stack = np.stack((AB, Ab, aB, ab), axis=0)
        return torch.as_tensor(sep_stack, dtype=dtype, device=self.device)

    def compute_coordinate_differences(self, A_row, a_row, B_col, b_col):
        # A_row, a_row: shape (nLayers, N_row, 2)
        # B_col, b_col: shape (nLayers, N_col, 2)
        diff_AB = A_row[:, :, None, :] - B_col[:, None, :, :]
        diff_Ab = A_row[:, :, None, :] - b_col[:, None, :, :]
        diff_aB = a_row[:, :, None, :] - B_col[:, None, :, :]
        diff_ab = a_row[:, :, None, :] - b_col[:, None, :, :]
        return np.stack((diff_AB, diff_Ab, diff_aB, diff_ab), axis=0)

    def compute_separations_vectorized(self, A_row, a_row, B_col, b_col):
        # A_row, a_row: shape (nLayers, N_row, 2)
        # B_col, b_col: shape (nLayers, N_col, 2)
        AB = np.linalg.norm(A_row[:, :, None, :] - B_col[:, None, :, :], axis=-1)
        Ab = np.linalg.norm(A_row[:, :, None, :] - b_col[:, None, :, :], axis=-1)
        aB = np.linalg.norm(a_row[:, :, None, :] - B_col[:, None, :, :], axis=-1)
        ab = np.linalg.norm(a_row[:, :, None, :] - b_col[:, None, :, :], axis=-1)
        return AB, Ab, aB, ab

    def compute_covariance_matrix_torch(self, diff_stack_torch, r0, L0, fractionalCn2, scale, constants, noise_var=None, row_sizes_tensor=None, diag_idx=None, Vx=None, Vy=None, dt=None):
        # diff_stack_torch: shape (4, nLayers, N_rows, N_cols, 2)
        # scale: shape (N_rows, N_cols)
        # constants: dict containing k1, a0, b0
        
        if Vx is not None and Vy is not None and dt is not None and dt != 0:
            disp_x = Vx[:, None, None] * dt
            disp_y = Vy[:, None, None] * dt
            disp = torch.stack((disp_x, disp_y), dim=-1) # (nLayers, 1, 1, 2)
            s_diff = diff_stack_torch + disp
            separations_torch = torch.norm(s_diff, p=2, dim=-1)
        else:
            separations_torch = torch.norm(diff_stack_torch, p=2, dim=-1)
        
        # Compute Von Karman structure function
        D = self.vk_structure_torch(r0, L0, separations_torch, constants)
        
        # Compute covariance block
        # block shape: (nLayers, N_rows, N_cols)
        block = scale * (-D[0] + D[1] + D[2] - D[3])
        
        # Sum over layers weighted by fractionalCn2
        # fractionalCn2 shape: (nLayers,)
        cov_matrix = torch.sum(fractionalCn2[:, None, None] * block, dim=0)
        
        # Add noise variance if applicable
        if noise_var is not None and row_sizes_tensor is not None and diag_idx is not None:
            nWFS = row_sizes_tensor.shape[0] // 2
            if noise_var.ndim == 0:
                noise_var_full = noise_var.expand(nWFS)
            elif noise_var.shape[0] == 1:
                noise_var_full = noise_var.expand(nWFS)
            else:
                noise_var_full = noise_var
            noise_block_values = torch.cat((noise_var_full, noise_var_full))
            noise_diag = torch.repeat_interleave(noise_block_values, row_sizes_tensor)
            cov_matrix[diag_idx, diag_idx] = cov_matrix[diag_idx, diag_idx] + noise_diag
            
        return cov_matrix
    
    def compute_covariance_matrix_chunked_torch(self, A_row, a_row, B_col, b_col, r0, L0, fractionalCn2, scale, constants, noise_var=None, row_sizes_tensor=None, diag_idx=None, Vx=None, Vy=None, dt=None, chunk_size=100, device=None, label="Covariance"):
        # A_row, a_row: shape (nLayers, N_rows, 2)
        # B_col, b_col: shape (nLayers, N_cols, 2)
        
        dtype = torch.float64
        calc_device = device if device is not None else self.device
        
        A_row_torch = torch.as_tensor(A_row, dtype=dtype, device=calc_device)
        a_row_torch = torch.as_tensor(a_row, dtype=dtype, device=calc_device)
        B_col_torch = torch.as_tensor(B_col, dtype=dtype, device=calc_device)
        b_col_torch = torch.as_tensor(b_col, dtype=dtype, device=calc_device)
        
        r0_dev = r0.to(calc_device)
        L0_dev = L0.to(calc_device)
        fractionalCn2_dev = fractionalCn2.to(calc_device)
        constants_dev = {k: v.to(calc_device) for k, v in constants.items()}
        
        N_rows = A_row_torch.shape[1]
        N_cols = B_col_torch.shape[1]
        
        cov_matrix = torch.zeros((N_rows, N_cols), dtype=dtype, device=calc_device)
        
        # Precompute displacement if wind is present
        if Vx is not None and Vy is not None and dt is not None and dt != 0:
            Vx_dev = Vx.to(calc_device)
            Vy_dev = Vy.to(calc_device)
            disp_x = Vx_dev[:, None, None] * dt
            disp_y = Vy_dev[:, None, None] * dt
            disp = torch.stack((disp_x, disp_y), dim=-1) # (nLayers, 1, 1, 2)
        else:
            disp = None
            
        n_chunks = (N_rows + chunk_size - 1) // chunk_size
        
        for idx_chunk, i in enumerate(range(0, N_rows, chunk_size)):
            if idx_chunk % 5 == 0 or idx_chunk == n_chunks - 1:
                self.logger.info(f"[{label}] Processing chunk {idx_chunk + 1}/{n_chunks} (rows {i} to {min(i + chunk_size, N_rows)})...")
                
            end_idx = min(i + chunk_size, N_rows)
            chunk_rows = end_idx - i
            chunk_A = A_row_torch[:, i:end_idx, :]
            chunk_a = a_row_torch[:, i:end_idx, :]
            
            # Pre-allocate differences stack to avoid duplicating memory allocation
            chunk_diff_stack = torch.zeros((4, A_row_torch.shape[0], chunk_rows, N_cols, 2), dtype=dtype, device=calc_device)
            
            # Compute coordinate differences directly into the pre-allocated stack
            torch.sub(chunk_A[:, :, None, :], B_col_torch[:, None, :, :], out=chunk_diff_stack[0])
            torch.sub(chunk_A[:, :, None, :], b_col_torch[:, None, :, :], out=chunk_diff_stack[1])
            torch.sub(chunk_a[:, :, None, :], B_col_torch[:, None, :, :], out=chunk_diff_stack[2])
            torch.sub(chunk_a[:, :, None, :], b_col_torch[:, None, :, :], out=chunk_diff_stack[3])
            
            if disp is not None:
                s_diff = chunk_diff_stack + disp
                separations_torch = torch.norm(s_diff, p=2, dim=-1)
            else:
                separations_torch = torch.norm(chunk_diff_stack, p=2, dim=-1)
            
            # Clean up intermediate difference tensors to free memory immediately
            del chunk_diff_stack
            
            # Compute Von Karman structure function for this chunk
            D = self.vk_structure_torch(r0_dev, L0_dev, separations_torch, constants_dev)
            
            # Compute covariance block for the chunk
            # slice scale first (which could be on CPU), then move chunk_scale to calc_device
            chunk_scale = scale[i:end_idx, :].to(calc_device)
            block = chunk_scale * (-D[0] + D[1] + D[2] - D[3])
            
            # Sum over layers weighted by fractionalCn2
            chunk_cov = torch.sum(fractionalCn2_dev[:, None, None] * block, dim=0)
            
            cov_matrix[i:end_idx, :] = chunk_cov
            
            # Clean up loop variables to release memory
            del separations_torch, D, block, chunk_cov
            
        # Add noise variance to diagonal if noise_var is provided
        if noise_var is not None and row_sizes_tensor is not None and diag_idx is not None:
            noise_var_dev = noise_var.to(calc_device)
            row_sizes_tensor_dev = row_sizes_tensor.to(calc_device)
            diag_idx_dev = diag_idx.to(calc_device)
            
            nWFS = row_sizes_tensor_dev.shape[0] // 2
            if noise_var_dev.ndim == 0:
                noise_var_full = noise_var_dev.expand(nWFS)
            elif noise_var_dev.shape[0] == 1:
                noise_var_full = noise_var_dev.expand(nWFS)
            else:
                noise_var_full = noise_var_dev
            noise_block_values = torch.cat((noise_var_full, noise_var_full))
            noise_diag = torch.repeat_interleave(noise_block_values, row_sizes_tensor_dev)
            cov_matrix[diag_idx_dev, diag_idx_dev] = cov_matrix[diag_idx_dev, diag_idx_dev] + noise_diag
            
        return cov_matrix

    
    def loadMeasurements(self, data_path, nWFS, subapsSel, selSamples, lp_indices=None, delay_frames=0):
        # Open data set
        with h5py.File(data_path, 'r') as f:
            wfs_slopes_t = []
            wfs_slopes_tdelay = []
            # Read number of samples
            nSamples = f['/LightPath_0/slopes_1D/data'].shape[0]
            n_valid_samples = nSamples - delay_frames
            # Generate random selection
            nSel = max(1, int(selSamples * n_valid_samples))

            idx_t = np.sort(np.random.choice(n_valid_samples, size=nSel, replace=False))
            idx_tdelay = idx_t + delay_frames
            # Read data
            for iWFS in range(nWFS):
                lp_idx = lp_indices[iWFS] if lp_indices is not None else iWFS
                data = f[f'/LightPath_{lp_idx}/slopes_1D/data']

                nSlopes = data.shape[1]
                nSubaps = nSlopes // 2
                sel = np.sort(np.concatenate([subapsSel[iWFS], subapsSel[iWFS] + nSubaps]))
                wfs_slopes_t.append(data[idx_t][:, sel])
                if delay_frames > 0:
                    wfs_slopes_tdelay.append(data[idx_tdelay][:, sel])
                else:
                    wfs_slopes_tdelay.append(data[idx_t][:, sel])

        return wfs_slopes_t, wfs_slopes_tdelay
    
    def compute_measurement_covariance(self, wfs_slopes_t, wfs_slopes_tdelay, gain, dtype):
        nWFS = len(wfs_slopes_t)
        nSubaps = wfs_slopes_t[0].shape[1] // 2
        nSamples = wfs_slopes_t[0].shape[0]

        cov_mat = np.zeros((2 * nWFS * nSubaps, 2 * nWFS * nSubaps))

        for iWFS in range(nWFS):
            slopes_x_i = wfs_slopes_t[iWFS][:, :nSubaps]
            slopes_y_i = wfs_slopes_t[iWFS][:, nSubaps:]

            slopes_x_i = slopes_x_i - slopes_x_i.mean(axis=0, keepdims=True)
            slopes_y_i = slopes_y_i - slopes_y_i.mean(axis=0, keepdims=True)

            for jWFS in range(nWFS):
                slopes_x_j = wfs_slopes_tdelay[jWFS][:, :nSubaps]
                slopes_y_j = wfs_slopes_tdelay[jWFS][:, nSubaps:]

                slopes_x_j = slopes_x_j - slopes_x_j.mean(axis=0, keepdims=True)
                slopes_y_j = slopes_y_j - slopes_y_j.mean(axis=0, keepdims=True)

                Cxx = slopes_x_i.T @ slopes_x_j / (nSamples - 1)
                Cxy = slopes_x_i.T @ slopes_y_j / (nSamples - 1)
                Cyx = slopes_y_i.T @ slopes_x_j / (nSamples - 1)
                Cyy = slopes_y_i.T @ slopes_y_j / (nSamples - 1)

                rxi = iWFS * nSubaps
                rxj = jWFS * nSubaps
                ryi = nWFS * nSubaps + iWFS * nSubaps
                ryj = nWFS * nSubaps + jWFS * nSubaps

                cov_mat[rxi:rxi+nSubaps, rxj:rxj+nSubaps] = Cxx * (gain**2)
                cov_mat[rxi:rxi+nSubaps, ryj:ryj+nSubaps] = Cxy * (gain**2)
                cov_mat[ryi:ryi+nSubaps, rxj:rxj+nSubaps] = Cyx * (gain**2)
                cov_mat[ryi:ryi+nSubaps, ryj:ryj+nSubaps] = Cyy * (gain**2)

        return torch.as_tensor(cov_mat, dtype=dtype, device=self.device)


    def learn(self, atm_guess, data_path, output_path, selSubaps=0.1, selSamples=0.1,
              lr=1e-2, max_iters=150, patience=50, min_delta=1e-9,
              lr_patience=20, lr_factor=0.5, optimize_noise=True, initial_noise=1e-2,
              lp_indices=None, wind_delay_frames=10, max_iters_wind=150, lr_wind=1e-1,
              patience_wind=50):
        self.logger.info('LearnEstimator::learn - loading initial params.')
        # Extract initial params
        initial_r0 = atm_guess['r0']
        initial_L0 = atm_guess['L0']
        
        initial_fractionalCn2 = np.array(atm_guess['fractionalCn2'])
        initial_windSpeedX = np.array(atm_guess.get('windSpeedX', np.zeros(self.nLayers)))
        initial_windSpeedY = np.array(atm_guess.get('windSpeedY', np.zeros(self.nLayers)))   

        # Check input parameters
        if initial_fractionalCn2.shape[0] != self.nLayers:
            raise ValueError('LearnEstimator::__init__ - Fractional r0 dimensions does not correspond the number of layers')
        if initial_windSpeedX.shape[0] != self.nLayers:
            raise ValueError('LearnEstimator::__init__ - Vx dimensions does not correspond the number of layers')
        if initial_windSpeedY.shape[0] != self.nLayers:
            raise ValueError('LearnEstimator::__init__ - Vy dimensions does not correspond the number of layers')              

        # Compute random subaps for each WFS
        self.logger.info('LearnEstimator::learn - randomly select subaps.')
        subapsSel = []
        for i in range(len(self.measWFSparams)):
            nSubaps = self.measWFSparams[i]['coordinates_per_layer'].shape[1]

            nSel = max(1, int(selSubaps * nSubaps))

            idx = np.sort(np.random.choice(nSubaps, size=nSel, replace=False))
            subapsSel.append(idx)

        dtype = torch.float64
        # Compute coordinate differences for the subapertures selected
        self.logger.info('LearnEstimator::learn - computing coordinate differences for randomly selected subaps.')
        t0 = time.time()
        
        # Vectorized coordinate computation for Z and S (which are both measWFSparams with subapsSel)
        A_row = np.concatenate([self.measWFSmidpoint_X[j]['midPointA'][:, subapsSel[j], :] for j in range(len(self.measWFSparams))] + 
                               [self.measWFSmidpoint_Y[j]['midPointA'][:, subapsSel[j], :] for j in range(len(self.measWFSparams))], axis=1)
        a_row = np.concatenate([self.measWFSmidpoint_X[j]['midPointa'][:, subapsSel[j], :] for j in range(len(self.measWFSparams))] + 
                               [self.measWFSmidpoint_Y[j]['midPointa'][:, subapsSel[j], :] for j in range(len(self.measWFSparams))], axis=1)
        
        diff_stack = self.compute_coordinate_differences(A_row, a_row, A_row, a_row)
        diff_stack_torch = torch.as_tensor(diff_stack, dtype=dtype, device=self.device)
        
        # Compute theoretical covariance matrix
        t1 = time.time()        
        self.logger.info('LearnEstimator::learn - Compute theoretical covariance matrix.')
        constants = self.prepare_vk_constants_torch(dtype)
        
        # Precompute sizes, scale and tensors for noise
        sizes_Z = []
        for j, wfs in enumerate(self.measWFSparams):
            sizes_Z.extend([wfs["subap_size"]] * len(subapsSel[j]))
        sizes_Z = sizes_Z + sizes_Z
        dZ = torch.tensor(sizes_Z, dtype=dtype, device=self.device)
        
        scale = 1.0 / (2.0 * dZ[:, None] * dZ[None, :])
        
        row_sizes = [len(subapsSel[j]) for j in range(len(self.measWFSparams))] * 2
        row_sizes_tensor = torch.tensor(row_sizes, device=self.device)
        cov_matrix_shape_0 = sum(row_sizes)
        diag_idx = torch.arange(cov_matrix_shape_0, device=self.device)
        
        # Define parameters to be optimized
        initial_r0_torch = torch.as_tensor(initial_r0, dtype=dtype, device=self.device)
        initial_L0_torch = torch.as_tensor(initial_L0, dtype=dtype, device=self.device)
        initial_fractionalCn2_torch = torch.as_tensor(initial_fractionalCn2, dtype=dtype, device=self.device)        
        
        # Initial call to compute_covariance_matrix_torch (no noise optimization parameters here yet)
        cov_theo = self.compute_covariance_matrix_torch(
            diff_stack_torch, initial_r0_torch, initial_L0_torch, initial_fractionalCn2_torch,
            scale, constants, noise_var=None
        )
        t2 = time.time()
        # Experimental covariance matrix (instantaneous, delay = 0)
        self.logger.info('LearnEstimator::learn - Load slopes measurements for Phase 1.')
        meas_slopes_t, meas_slopes_tdelay = self.loadMeasurements(data_path, len(self.measWFSparams), subapsSel, selSamples, lp_indices=lp_indices, delay_frames=0)
        self.logger.info('LearnEstimator::learn - Compute experimental covariance for Phase 1.')
        gain = (self.measWFSparams[0]['subap_size']*2*np.pi*2)/(206265.0*self.measWFSparams[0]['plate_scale']*self.measWFSparams[0]['wavelength'])
        cov_meas = self.compute_measurement_covariance(meas_slopes_t, meas_slopes_tdelay, gain, dtype)
        t3 = time.time()
        self.logger.info(f'Separation {t1-t0}, CovMat (1it):{t2-t1}, Experimental: {t3-t2}')

        # Phase 1: Atmosphere & Noise Estimation
        self.logger.info("LearnEstimator::learn - Starting Phase 1: Atmosphere & Noise Estimation...")
        raw_r0 = torch.nn.Parameter(torch.log(torch.expm1(initial_r0_torch)))
        raw_L0 = torch.nn.Parameter(torch.log(torch.expm1(initial_L0_torch)))
        raw_cn2 = torch.nn.Parameter(torch.log(initial_fractionalCn2_torch + 1e-12))

        # Setup noise optimization if requested
        if optimize_noise:
            initial_noise_torch = torch.as_tensor(initial_noise, dtype=dtype, device=self.device)
            raw_noise = torch.nn.Parameter(torch.log(torch.expm1(initial_noise_torch)))
            optimizer = torch.optim.Adam([raw_r0, raw_L0, raw_cn2, raw_noise], lr=lr)
        else:
            raw_noise = None
            optimizer = torch.optim.Adam([raw_r0, raw_L0, raw_cn2], lr=lr)

        # Setup learning rate scheduler for plateau detection
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience, threshold=1e-6
        )

        # Variables to track early stopping state
        best_loss = float('inf')
        best_params = None
        patience_counter = 0
        last_lr = lr

        for it in range(max_iters):
            optimizer.zero_grad()

            r0 = torch.nn.functional.softplus(raw_r0) + 1e-6
            L0 = torch.nn.functional.softplus(raw_L0) + 1e-6
            fractionalCn2 = torch.nn.functional.softmax(raw_cn2, dim=0)
            
            if optimize_noise:
                noise_var = torch.nn.functional.softplus(raw_noise) + 1e-8
            else:
                noise_var = None

            cov_theo = self.compute_covariance_matrix_torch(
                diff_stack_torch, r0, L0, fractionalCn2, scale, constants,
                noise_var=noise_var, row_sizes_tensor=row_sizes_tensor, diag_idx=diag_idx
            )

            loss = torch.mean((cov_theo - cov_meas) ** 2)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            # Update scheduler for plateau detection
            scheduler.step(loss_val)

            # Monitor learning rate reduction for logging plateau detection
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < last_lr:
                self.logger.info(f"it={it} - [Plateau Detected] Reducing learning rate from {last_lr:.2e} to {current_lr:.2e}")
                last_lr = current_lr
            if loss_val < best_loss - min_delta:
                best_loss = loss_val
                if optimize_noise:
                    best_params = (raw_r0.detach().clone(), raw_L0.detach().clone(), raw_cn2.detach().clone(), raw_noise.detach().clone())
                else:
                    best_params = (raw_r0.detach().clone(), raw_L0.detach().clone(), raw_cn2.detach().clone(), None)
                patience_counter = 0
            else:
                patience_counter += 1

            if it % 25 == 0:
                noise_str = f", noise={noise_var.detach().cpu().numpy()}" if optimize_noise else ""
                self.logger.info(f"it={it}, loss={loss_val:.6e}, r0={r0.item():.4f}, L0={L0.item():.4f}, Cn2={fractionalCn2.detach().cpu().numpy()}{noise_str}")

            if patience_counter >= patience:
                self.logger.info(f"it={it} - [Early Stopping] Triggered. No improvement in loss for {patience} iterations.")
                break

        # Restore the best parameters
        if best_params is not None:
            with torch.no_grad():
                raw_r0.copy_(best_params[0])
                raw_L0.copy_(best_params[1])
                raw_cn2.copy_(best_params[2])
                if optimize_noise and best_params[3] is not None:
                    raw_noise.copy_(best_params[3])

        # Compute and assign final optimized parameters
        with torch.no_grad():
            r0_opt = torch.nn.functional.softplus(raw_r0) + 1e-6
            L0_opt = torch.nn.functional.softplus(raw_L0) + 1e-6
            fractionalCn2_opt = torch.nn.functional.softmax(raw_cn2, dim=0)
            if optimize_noise:
                noise_opt = torch.nn.functional.softplus(raw_noise) + 1e-8
            else:
                noise_opt = None

        self.r0 = r0_opt.item()
        self.L0 = L0_opt.item()
        self.fractionalCn2 = fractionalCn2_opt.detach().cpu().numpy()
        if optimize_noise:
            self.noise_var = noise_opt.detach().cpu().numpy()
        else:
            self.noise_var = None

        self.logger.info(f"Phase 1 completed. Best Loss: {best_loss:.6e}")
        self.logger.info(f"Optimized parameters (Phase 1): r0={self.r0:.4f}, L0={self.L0:.4f}, Cn2={self.fractionalCn2}")
        if optimize_noise:
            self.logger.info(f"Optimized noise variance per WFS (Phase 1): {self.noise_var}")

        # Phase 2: Wind Velocity Estimation (Vx, Vy)
        self.logger.info("LearnEstimator::learn - Starting Phase 2: Wind Speed Estimation...")
        
        # Freeze atmospheric parameters
        r0_fixed = torch.as_tensor(self.r0, dtype=dtype, device=self.device)
        L0_fixed = torch.as_tensor(self.L0, dtype=dtype, device=self.device)
        fractionalCn2_fixed = torch.as_tensor(self.fractionalCn2, dtype=dtype, device=self.device)
        noise_var_fixed = torch.as_tensor(self.noise_var, dtype=dtype, device=self.device) if self.noise_var is not None else None

        if isinstance(wind_delay_frames, int):
            wind_delays = [wind_delay_frames]
        else:
            wind_delays = list(wind_delay_frames)

        cov_meas_wind_list = []
        dt_list = []
        sampling_time = self.measLPs[0].tel.samplingTime if len(self.measLPs) > 0 else 0.0005

        for delay in wind_delays:
            # Load measurement slopes with delay
            self.logger.info(f'LearnEstimator::learn - Load slopes measurements for Phase 2 (delay={delay} frames).')
            meas_slopes_t_wind, meas_slopes_tdelay_wind = self.loadMeasurements(
                data_path, len(self.measWFSparams), subapsSel, selSamples, lp_indices=lp_indices, delay_frames=delay
            )
            self.logger.info(f'LearnEstimator::learn - Compute experimental cross-covariance for Phase 2 (delay={delay} frames).')
            cov_meas_wind = self.compute_measurement_covariance(meas_slopes_t_wind, meas_slopes_tdelay_wind, gain, dtype)
            cov_meas_wind_list.append(cov_meas_wind)
            dt_list.append(delay * sampling_time)

        # Define wind parameters to be optimized in PyTorch
        raw_Vx = torch.nn.Parameter(torch.as_tensor(initial_windSpeedX, dtype=dtype, device=self.device))
        raw_Vy = torch.nn.Parameter(torch.as_tensor(initial_windSpeedY, dtype=dtype, device=self.device))

        optimizer_wind = torch.optim.Adam([raw_Vx, raw_Vy], lr=lr_wind)
        scheduler_wind = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_wind, mode='min', factor=lr_factor, patience=lr_patience, threshold=1e-6
        )

        best_loss_wind = float('inf')
        best_params_wind = None
        patience_counter_wind = 0
        last_lr_wind = lr_wind

        for it in range(max_iters_wind):
            optimizer_wind.zero_grad()

            loss_wind = 0.0
            for cov_meas_wind, dt in zip(cov_meas_wind_list, dt_list):
                cov_theo_wind = self.compute_covariance_matrix_torch(
                    diff_stack_torch, r0_fixed, L0_fixed, fractionalCn2_fixed, scale, constants,
                    noise_var=noise_var_fixed, row_sizes_tensor=row_sizes_tensor, diag_idx=diag_idx,
                    Vx=raw_Vx, Vy=raw_Vy, dt=dt
                )
                loss_wind = loss_wind + torch.mean((cov_theo_wind - cov_meas_wind) ** 2)
            
            # Average the loss across all delays
            loss_wind = loss_wind / len(wind_delays)
            
            loss_wind.backward()
            optimizer_wind.step()

            loss_val_wind = loss_wind.item()
            scheduler_wind.step(loss_val_wind)

            current_lr_wind = optimizer_wind.param_groups[0]['lr']
            if current_lr_wind < last_lr_wind:
                self.logger.info(f"wind_it={it} - [Plateau Detected] Reducing wind learning rate from {last_lr_wind:.2e} to {current_lr_wind:.2e}")
                last_lr_wind = current_lr_wind

            if loss_val_wind < best_loss_wind - min_delta:
                best_loss_wind = loss_val_wind
                best_params_wind = (raw_Vx.detach().clone(), raw_Vy.detach().clone())
                patience_counter_wind = 0
            else:
                patience_counter_wind += 1

            if it % 25 == 0:
                self.logger.info(f"wind_it={it}, loss={loss_val_wind:.6e}, Vx={raw_Vx.detach().cpu().numpy()}, Vy={raw_Vy.detach().cpu().numpy()}")

            if patience_counter_wind >= patience_wind:
                self.logger.info(f"wind_it={it} - [Early Stopping] Triggered. No improvement in wind loss for {patience_wind} iterations.")
                break

        # Restore best wind parameters
        if best_params_wind is not None:
            with torch.no_grad():
                raw_Vx.copy_(best_params_wind[0])
                raw_Vy.copy_(best_params_wind[1])

        self.windSpeedX = raw_Vx.detach().cpu().numpy()
        self.windSpeedY = raw_Vy.detach().cpu().numpy()

        self.logger.info(f"Phase 2 completed. Best Loss: {best_loss_wind:.6e}")
        self.logger.info(f"Optimized parameters: r0={self.r0:.4f}, L0={self.L0:.4f}, Cn2={self.fractionalCn2}, Vx={self.windSpeedX}, Vy={self.windSpeedY}")

        # Clean up local references and release CUDA memory cache
        if torch.cuda.is_available():
            del raw_r0, raw_L0, raw_cn2, raw_Vx, raw_Vy, optimizer, scheduler, optimizer_wind, scheduler_wind
            del cov_theo, cov_meas, diff_stack_torch, best_params, best_params_wind
            if 'cov_theo_wind' in locals():
                del cov_theo_wind
            del cov_meas_wind_list
            if optimize_noise:
                del raw_noise
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return {
            'r0': self.r0,
            'L0': self.L0,
            'fractionalCn2': self.fractionalCn2,
            'noise_var': self.noise_var,
            'windSpeedX': self.windSpeedX,
            'windSpeedY': self.windSpeedY
        }


    def build_reconstructor(self, r0=None, L0=None, fractionalCn2=None, noise_var=None, regularization=1e-9, windSpeedX=None, windSpeedY=None, dt=None, chunk_size=None, device='cpu', permute_for_controller=True):
        """
        Build the tomographic reconstructor R_tomo = C_ts (C_ss + C_nn)^-1
        
        Parameters
        ----------
        r0: float, optional
            Fried parameter. If None, uses self.r0.
        L0: float, optional
            Outer scale. If None, uses self.L0.
        fractionalCn2: array-like, optional
            Fractional Cn2 profile. If None, uses self.fractionalCn2.
        noise_var: array-like, optional
            Noise variance per sensed WFS. If None, uses self.noise_var.
            If self.noise_var is None, uses zero noise.
        regularization: float, optional
            Diagonal regularization added to C_ss.
        windSpeedX: array-like, optional
            Wind speed along X for delay compensation. If None, uses self.windSpeedX.
        windSpeedY: array-like, optional
            Wind speed along Y for delay compensation. If None, uses self.windSpeedY.
        dt: float, optional
            Time delay in seconds. If None, computes from target LP delay.
        chunk_size: int, optional
            Chunk size along the rows for memory-efficient computation.
        device: str or torch.device, optional
            The device to perform computations on. Default 'cpu'.
        """
        if r0 is None:
            r0 = self.r0
        if L0 is None:
            L0 = self.L0
        if fractionalCn2 is None:
            fractionalCn2 = self.fractionalCn2
        if noise_var is None:
            noise_var = self.noise_var
        if windSpeedX is None:
            windSpeedX = getattr(self, 'windSpeedX', None)
        if windSpeedY is None:
            windSpeedY = getattr(self, 'windSpeedY', None)

        if r0 is None or L0 is None or fractionalCn2 is None:
            raise ValueError("r0, L0, and fractionalCn2 must be estimated or provided.")

        if dt is None:
            # Check target LPs delay
            delay_frames = self.targetLPs[0].delay if hasattr(self, 'targetLPs') and len(self.targetLPs) > 0 else 0
            sampling_time = self.targetLPs[0].tel.samplingTime if hasattr(self, 'targetLPs') and len(self.targetLPs) > 0 else 0.001
            dt = delay_frames * sampling_time

        dtype = torch.float64
        calc_device = device if device is not None else self.device
        if chunk_size is None:
            chunk_size = 100

        try:
            constants = self.prepare_vk_constants_torch(dtype)

            r0_torch = torch.as_tensor(r0, dtype=dtype, device=calc_device)
            L0_torch = torch.as_tensor(L0, dtype=dtype, device=calc_device)
            fractionalCn2_torch = torch.as_tensor(fractionalCn2, dtype=dtype, device=calc_device)

            if windSpeedX is not None and windSpeedY is not None and dt != 0:
                windSpeedX_torch = torch.as_tensor(windSpeedX, dtype=dtype, device=calc_device)
                windSpeedY_torch = torch.as_tensor(windSpeedY, dtype=dtype, device=calc_device)
            else:
                windSpeedX_torch = None
                windSpeedY_torch = None

            # 1. Build C_ss (sensed-sensed covariance matrix) using all subapertures
            subaps_sensed = [np.arange(wfs['coordinates_per_layer'].shape[1]) for wfs in self.measWFSparams]
            
            self.logger.info("LearnEstimator::build_reconstructor - Computing sensed-sensed C_ss covariance matrix...")
            A_row_s = np.concatenate([self.measWFSmidpoint_X[j]['midPointA'][:, subaps_sensed[j], :] for j in range(len(self.measWFSparams))] + 
                                     [self.measWFSmidpoint_Y[j]['midPointA'][:, subaps_sensed[j], :] for j in range(len(self.measWFSparams))], axis=1)
            a_row_s = np.concatenate([self.measWFSmidpoint_X[j]['midPointa'][:, subaps_sensed[j], :] for j in range(len(self.measWFSparams))] + 
                                     [self.measWFSmidpoint_Y[j]['midPointa'][:, subaps_sensed[j], :] for j in range(len(self.measWFSparams))], axis=1)
            
            if noise_var is not None:
                noise_var_torch = torch.as_tensor(noise_var, dtype=dtype, device=calc_device)
            else:
                noise_var_torch = None

            sizes_meas = []
            for j, wfs in enumerate(self.measWFSparams):
                sizes_meas.extend([wfs["subap_size"]] * len(subaps_sensed[j]))
            sizes_meas = sizes_meas + sizes_meas
            d_meas = torch.tensor(sizes_meas, dtype=dtype, device='cpu')
            
            scale_ss = 1.0 / (2.0 * d_meas[:, None] * d_meas[None, :])
            
            row_sizes_meas = [len(subaps_sensed[j]) for j in range(len(self.measWFSparams))] * 2
            row_sizes_meas_tensor = torch.tensor(row_sizes_meas, device=calc_device)
            diag_idx_ss = torch.arange(sum(row_sizes_meas), device=calc_device)

            Css = self.compute_covariance_matrix_chunked_torch(
                A_row_s, a_row_s, A_row_s, a_row_s, r0_torch, L0_torch, fractionalCn2_torch,
                scale_ss, constants, noise_var=noise_var_torch,
                row_sizes_tensor=row_sizes_meas_tensor, diag_idx=diag_idx_ss, chunk_size=chunk_size, device=calc_device, label="Css"
            )

            # Apply diagonal regularization
            if regularization > 0:
                diag_idx = torch.arange(Css.shape[0], device=calc_device)
                Css[diag_idx, diag_idx] = Css[diag_idx, diag_idx] + regularization

            # 2. Build C_ts (target-sensed covariance matrix) using all subapertures
            subaps_target = [np.arange(wfs['coordinates_per_layer'].shape[1]) for wfs in self.targetWFSparams]

            self.logger.info("LearnEstimator::build_reconstructor - Computing target-sensed C_ts covariance matrix...")
            A_row_t = np.concatenate([self.targetWFSmidpoint_X[j]['midPointA'][:, subaps_target[j], :] for j in range(len(self.targetWFSparams))] + 
                                     [self.targetWFSmidpoint_Y[j]['midPointA'][:, subaps_target[j], :] for j in range(len(self.targetWFSparams))], axis=1)
            a_row_t = np.concatenate([self.targetWFSmidpoint_X[j]['midPointa'][:, subaps_target[j], :] for j in range(len(self.targetWFSparams))] + 
                                     [self.targetWFSmidpoint_Y[j]['midPointa'][:, subaps_target[j], :] for j in range(len(self.targetWFSparams))], axis=1)
            
            sizes_target = []
            for j, wfs in enumerate(self.targetWFSparams):
                sizes_target.extend([wfs["subap_size"]] * len(subaps_target[j]))
            sizes_target = sizes_target + sizes_target
            d_target = torch.tensor(sizes_target, dtype=dtype, device='cpu')
            
            scale_ts = 1.0 / (2.0 * d_target[:, None] * d_meas[None, :])

            Cts = self.compute_covariance_matrix_chunked_torch(
                A_row_t, a_row_t, A_row_s, a_row_s, r0_torch, L0_torch, fractionalCn2_torch,
                scale_ts, constants, noise_var=None,
                Vx=windSpeedX_torch, Vy=windSpeedY_torch, dt=dt, chunk_size=chunk_size, device=calc_device, label="Cts"
            )

            # 3. Solve R_tomo = C_ts @ Css^-1 using a stable solver
            self.logger.info("LearnEstimator::build_reconstructor - Solving reconstructor R_tomo...")
            try:
                L_factor = torch.linalg.cholesky(Css)
                Rtomo_T = torch.cholesky_solve(Cts.T.contiguous(), L_factor, upper=False)
                Rtomo = Rtomo_T.T
            except RuntimeError as e:
                self.logger.warning(f"Cholesky solve failed: {e}. Falling back to torch.linalg.solve.")
                Rtomo_T = torch.linalg.solve(Css, Cts.T.contiguous())
                Rtomo = Rtomo_T.T

            # Clean up CUDA memory cache
            del Css, Cts
            if noise_var_torch is not None:
                del noise_var_torch
            if windSpeedX_torch is not None:
                del windSpeedX_torch
            if windSpeedY_torch is not None:
                del windSpeedY_torch
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            if permute_for_controller:
                self.logger.info("LearnEstimator::build_reconstructor - Permuting Rtomo to [WFS0_X, WFS0_Y, WFS1_X, WFS1_Y, ...] controller format.")
                # Sensed WFS permutation
                n_subs_s = [wfs['coordinates_per_layer'].shape[1] for wfs in self.measWFSparams]
                total_sub_s = sum(n_subs_s)
                x_starts_s = [0] + list(np.cumsum(n_subs_s)[:-1])
                y_starts_s = [total_sub_s + start for start in x_starts_s]
                perm_s = []
                for i in range(len(n_subs_s)):
                    perm_s.extend(list(range(x_starts_s[i], x_starts_s[i] + n_subs_s[i])) + 
                                  list(range(y_starts_s[i], y_starts_s[i] + n_subs_s[i])))

                # Target WFS permutation
                n_subs_t = [wfs['coordinates_per_layer'].shape[1] for wfs in self.targetWFSparams]
                total_sub_t = sum(n_subs_t)
                x_starts_t = [0] + list(np.cumsum(n_subs_t)[:-1])
                y_starts_t = [total_sub_t + start for start in x_starts_t]
                perm_t = []
                for i in range(len(n_subs_t)):
                    perm_t.extend(list(range(x_starts_t[i], x_starts_t[i] + n_subs_t[i])) + 
                                  list(range(y_starts_t[i], y_starts_t[i] + n_subs_t[i])))

                Rtomo = Rtomo[perm_t, :][:, perm_s]

            return Rtomo.cpu().numpy()

        except (torch.OutOfMemoryError, RuntimeError) as e:
            if calc_device != 'cpu' and torch.cuda.is_available():
                self.logger.warning(f"CUDA Out Of Memory or error during build_reconstructor on device {calc_device}: {e}. Falling back to CPU...")
                # Clear GPU cache to free up VRAM for other tasks
                torch.cuda.empty_cache()
                # Run on CPU with a larger chunk_size (1000) for speed
                return self.build_reconstructor(
                    r0=r0, L0=L0, fractionalCn2=fractionalCn2, noise_var=noise_var,
                    regularization=regularization, windSpeedX=windSpeedX, windSpeedY=windSpeedY,
                    dt=dt, chunk_size=1000, device='cpu', permute_for_controller=permute_for_controller
                )
            else:
                raise e


    def setup_logging(self, logging_level=logging.INFO):
        #  Setup of logging at the main process using QueueHandler
        log_queue = Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging_level)  # Minimum log level

        # Setup of the formatting
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # Output to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Qeue handler captures the messages from the different logs and serialize them
        queue_listener = logging.handlers.QueueListener(log_queue, console_handler)
        root_logger.addHandler(queue_handler)
        queue_listener.start()

        return queue_listener
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()