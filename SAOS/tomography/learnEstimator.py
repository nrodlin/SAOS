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

    def separations_to_torch(self, separations, dtype=torch.float64):
        sep_torch = np.empty(separations.shape, dtype=object)

        for idx in np.ndindex(separations.shape):
            AB, Ab, aB, ab = separations[idx]
            sep_torch[idx] = (
                torch.as_tensor(AB, dtype=dtype, device=self.device),
                torch.as_tensor(Ab, dtype=dtype, device=self.device),
                torch.as_tensor(aB, dtype=dtype, device=self.device),
                torch.as_tensor(ab, dtype=dtype, device=self.device),
            )

        return sep_torch        
    
    def compute_separations(self, A, a, B, b):
        # A, a: shape (nSubapA, 2)
        # B, b: shape (nSubapB, 2)

        AB = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        Ab = np.linalg.norm(A[:, None, :] - b[None, :, :], axis=2)
        aB = np.linalg.norm(a[:, None, :] - B[None, :, :], axis=2)
        ab = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

        return AB, Ab, aB, ab    
    
    def compute_separations_array(self, subapsSelZ, subapsSelS, listWFS_Z_X, listWFS_Z_Y, listWFS_S_X, listWFS_S_Y):
        
        separations = {'XX': [], 'XY': [], 'YX': [], 'YY': []}

        for iLayer in range(self.nLayers):

            layer_sep  = {'XX': [], 'XY': [], 'YX': [], 'YY': []}

            for jWFS in range(len(listWFS_Z_X)):
                for kWFS in range(len(listWFS_S_X)):
                    sel_j = subapsSelZ[jWFS]
                    sel_k = subapsSelS[kWFS]

                    # XX
                    A = listWFS_Z_X[jWFS]['midPointA'][iLayer, sel_j, :]
                    a = listWFS_Z_X[jWFS]['midPointa'][iLayer, sel_j, :]
                    B = listWFS_S_X[kWFS]['midPointA'][iLayer, sel_k, :]
                    b = listWFS_S_X[kWFS]['midPointa'][iLayer, sel_k, :]

                    layer_sep['XX'].append(
                        self.compute_separations(A, a, B, b)
                    )

                    # XY
                    A = listWFS_Z_X[jWFS]['midPointA'][iLayer, sel_j, :]
                    a = listWFS_Z_X[jWFS]['midPointa'][iLayer, sel_j, :]
                    B = listWFS_S_Y[kWFS]['midPointA'][iLayer, sel_k, :]
                    b = listWFS_S_Y[kWFS]['midPointa'][iLayer, sel_k, :]

                    layer_sep['XY'].append(
                        self.compute_separations(A, a, B, b)
                    )

                    # YX
                    A = listWFS_Z_Y[jWFS]['midPointA'][iLayer, sel_j, :]
                    a = listWFS_Z_Y[jWFS]['midPointa'][iLayer, sel_j, :]
                    B = listWFS_S_X[kWFS]['midPointA'][iLayer, sel_k, :]
                    b = listWFS_S_X[kWFS]['midPointa'][iLayer, sel_k, :]

                    layer_sep['YX'].append(
                        self.compute_separations(A, a, B, b)
                    )

                    # YY
                    A = listWFS_Z_Y[jWFS]['midPointA'][iLayer, sel_j, :]
                    a = listWFS_Z_Y[jWFS]['midPointa'][iLayer, sel_j, :]
                    B = listWFS_S_Y[kWFS]['midPointA'][iLayer, sel_k, :]
                    b = listWFS_S_Y[kWFS]['midPointa'][iLayer, sel_k, :]

                    layer_sep['YY'].append(
                        self.compute_separations(A, a, B, b)
                    )
            for key in separations:
                separations[key].append(layer_sep[key])

        # Separations per layer to array
        nWFS_Z = len(listWFS_Z_X)
        nWFS_S = len(listWFS_S_X)

        separations_array = []

        for iLayer in range(self.nLayers):

            XX = np.empty((nWFS_Z, nWFS_S), dtype=object)
            XY = np.empty((nWFS_Z, nWFS_S), dtype=object)
            YX = np.empty((nWFS_Z, nWFS_S), dtype=object)
            YY = np.empty((nWFS_Z, nWFS_S), dtype=object)

            for jWFS in range(nWFS_Z):
                for kWFS in range(nWFS_S):
                    pair_id = jWFS * nWFS_S + kWFS

                    XX[jWFS, kWFS] = separations['XX'][iLayer][pair_id]
                    XY[jWFS, kWFS] = separations['XY'][iLayer][pair_id]
                    YX[jWFS, kWFS] = separations['YX'][iLayer][pair_id]
                    YY[jWFS, kWFS] = separations['YY'][iLayer][pair_id]

            sep = np.block([
                [XX, XY],
                [YX, YY]
            ])

            separations_array.append(sep)
        # Each cell of the array contains a tuple (AB, Ab, aB, ab)
        separations_array = np.array(separations_array, dtype=object)

        return separations_array
    
    def covariance_from_separations_torch(self, r0, L0, AB, Ab, aB, ab, sizeZ, sizeS, constants):
        sep_stack = torch.stack((AB, Ab, aB, ab), dim=0)
        D = self.vk_structure_torch(r0, L0, sep_stack, constants)

        return (1.0 / (2.0 * sizeZ * sizeS)) * (-D[0] + D[1] + D[2] - D[3])
    
    def compute_covariance_matrix_torch(self, separations, r0, L0, fractR0, listWFS_Z_params, listWFS_S_params, constants, dtype=torch.float64):

        nWFS_Z = len(listWFS_Z_params)
        nWFS_S = len(listWFS_S_params)

        row_sizes = [separations[0, row, 0][0].shape[0] for row in range(2 * nWFS_Z)]
        col_sizes = [separations[0, 0, col][0].shape[1] for col in range(2 * nWFS_S)]

        row_offsets = np.cumsum([0] + row_sizes)
        col_offsets = np.cumsum([0] + col_sizes)

        cov_matrix = torch.zeros((row_offsets[-1], col_offsets[-1]), dtype=dtype, device=self.device)

        for iLayer in range(self.nLayers):
            for row in range(2 * nWFS_Z):
                jWFS = row if row < nWFS_Z else row - nWFS_Z
                r0a, r1a = row_offsets[row], row_offsets[row + 1]

                for col in range(2 * nWFS_S):
                    kWFS = col if col < nWFS_S else col - nWFS_S
                    c0, c1 = col_offsets[col], col_offsets[col + 1]

                    AB, Ab, aB, ab = separations[iLayer, row, col]

                    dZ = listWFS_Z_params[jWFS]["subap_size"]
                    dS = listWFS_S_params[kWFS]["subap_size"]

                    block = self.covariance_from_separations_torch(
                        r0, L0, AB, Ab, aB, ab, dZ, dS, constants
                    )

                    cov_matrix[r0a:r1a, c0:c1] = (cov_matrix[r0a:r1a, c0:c1] + fractR0[iLayer] * block)

        return cov_matrix
    
    def loadMeasurements(self, data_path, nWFS, subapsSel, selSamples):
        # Open data set
        with h5py.File(data_path, 'r') as f:
            wfs_slopes = []
            # Read number of samples
            nSamples = f['/LightPath_0/slopes_1D/data'].shape[0]
            # Generate random selection
            nSel = max(1, int(selSamples * nSamples))

            idx = np.sort(np.random.choice(nSamples, size=nSel, replace=False))
            # Read data
            for iWFS in range(nWFS):
                data = f[f'/LightPath_{iWFS}/slopes_1D/data']

                nSlopes = data.shape[1]
                nSubaps = nSlopes // 2
                sel = np.sort(np.concatenate([subapsSel[iWFS], subapsSel[iWFS] + nSubaps]))
                wfs_slopes.append(data[idx][:, sel])                

        return wfs_slopes
    
    def compute_measurement_covariance(self, wfs_slopes, gain, dtype):
        nWFS = len(wfs_slopes)
        nSubaps = wfs_slopes[0].shape[1] // 2
        nSamples = wfs_slopes[0].shape[0]

        cov_mat = np.zeros((2 * nWFS * nSubaps, 2 * nWFS * nSubaps))

        for iWFS in range(nWFS):
            slopes_x_i = wfs_slopes[iWFS][:, :nSubaps]
            slopes_y_i = wfs_slopes[iWFS][:, nSubaps:]

            slopes_x_i = slopes_x_i - slopes_x_i.mean(axis=0, keepdims=True)
            slopes_y_i = slopes_y_i - slopes_y_i.mean(axis=0, keepdims=True)

            for jWFS in range(nWFS):
                slopes_x_j = wfs_slopes[jWFS][:, :nSubaps]
                slopes_y_j = wfs_slopes[jWFS][:, nSubaps:]

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


    def learn(self, atm_guess, data_path, output_path, selSubaps=0.1, selSamples=0.1):
        self.logger.info('LearnEstimator::learn - loading initial params.')
        # Extract initial params
        initial_r0 = atm_guess['r0']
        initial_L0 = atm_guess['L0']
        
        initial_fractionalR0 = np.array(atm_guess['fractionalR0'])
        initial_windSpeedX = np.array(atm_guess['windSpeedX'])
        initial_windSpeedY = np.array(atm_guess['windSpeedY'])   

        # Check input parameters
        if initial_fractionalR0.shape[0] != self.nLayers:
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
        # Compute separations for the subapertures selected
        self.logger.info('LearnEstimator::learn - computing separations for randomly selected subaps.')
        t0 = time.time()
        separations_theo_meas = self.compute_separations_array(subapsSel, subapsSel, self.measWFSmidpoint_X, self.measWFSmidpoint_Y, 
                                                               self.measWFSmidpoint_X, self.measWFSmidpoint_Y)
        separations_torch = self.separations_to_torch(separations_theo_meas, dtype)
        # Compute theoretical covariance matrix
        t1 = time.time()        
        self.logger.info('LearnEstimator::learn - Compute theoretical covariance matrix.')
        constants = self.prepare_vk_constants_torch(dtype)
        cov_theo = self.compute_covariance_matrix_torch(separations_torch, initial_r0, initial_L0, initial_fractionalR0, self.measWFSparams, self.measWFSparams, constants, dtype=dtype)
        t2 = time.time()
        # Experimental covariance matrix 
        self.logger.info('LearnEstimator::learn - Load slopes measurements.')
        meas_slopes = self.loadMeasurements(data_path, len(self.measWFSparams), subapsSel, selSamples)
        self.logger.info('LearnEstimator::learn - Compute exprimental covariance.')
        gain = (self.measWFSparams[0]['subap_size']*2*np.pi*2)/(206265.0*self.measWFSparams[0]['plate_scale']*self.measWFSparams[0]['wavelength'])
        cov_meas    = self.compute_measurement_covariance(meas_slopes, gain, dtype)
        t3 = time.time()
        self.logger.info(f'Separation {t1-t0}, CovMat (1it):{t2-t1}, Experimental: {t3-t2}')

        # Define parameters to be optimized
        initial_r0_torch = torch.as_tensor(initial_r0, dtype=dtype, device=self.device)
        initial_L0_torch = torch.as_tensor(initial_L0, dtype=dtype, device=self.device)
        initial_fractionalR0_torch = torch.as_tensor(initial_fractionalR0, dtype=dtype, device=self.device)        
        
        raw_r0 = torch.nn.Parameter(torch.log(torch.expm1(initial_r0_torch)))
        raw_L0 = torch.nn.Parameter(torch.log(torch.expm1(initial_L0_torch)))
        raw_cn2 = torch.nn.Parameter(torch.log(initial_fractionalR0_torch + 1e-12))

        # Setup the optimizer
        optimizer = torch.optim.Adam([raw_r0, raw_L0, raw_cn2], lr=1e-2)

        for it in range(500):
            optimizer.zero_grad()

            r0 = torch.nn.functional.softplus(raw_r0) + 1e-6
            L0 = torch.nn.functional.softplus(raw_L0) + 1e-6
            fractR0 = torch.nn.functional.softmax(raw_cn2, dim=0)

            cov_theo = self.compute_covariance_matrix_torch(separations_torch, r0, L0, fractR0, self.measWFSparams, self.measWFSparams, constants, dtype=dtype)

            loss = torch.mean((cov_theo - cov_meas) ** 2)

            loss.backward()
            optimizer.step()

            if it % 25 == 0:
                self.logger.info(f"it={it}, loss={loss.item():.6e}, r0={r0.item():.4f}, L0={L0.item():.4f}, Cn2={fractR0.detach().cpu().numpy()}")

        return True


    def setup_logging(self, logging_level=logging.WARNING):
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