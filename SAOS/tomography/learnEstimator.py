import numpy as np
import torch

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

    def vk_structure(self, r0, L0, separation, nMax=10):

        # Force numpy

        separation = np.asarray(separation, dtype=float)

        Dphi = np.zeros_like(separation)

        # If all separation are 0, then we return inmediately 0s
        valid = separation > 0

        if not np.any(valid):
            return Dphi      

        # Select separations that are valid (>0)
        r = separation[valid]

        x  = np.pi * (r / L0)
        y = x**(5/3)

        k1 = gamma(11/6) * (2**(1/6)) * (np.pi**(-8/3)) * (((24/5) * gamma(6/5))**(5/6))

        X1 = x**2
        a0 = gamma(-5/6) / (2**(1/6))
        b0 = gamma(5/6)  / (2**(1/6))

        recursive_term = np.zeros_like(r)
        aprev = a0
        bprev = b0
        Xprev = np.ones_like(r)

        for i in range(nMax):
            n = i+1
            an = aprev / (n*(n+5/6))
            bn = bprev / (n*(n-5/6))
            Xn = Xprev * X1

            recursive_term += (an*y + bn)*Xn

            # Update for next it
            aprev = an
            bprev = bn
            Xprev = Xn
        
        # compute structure function
        Dphi[valid] = -k1 * (r0**(-5/3)) * (L0 **(5/3)) * (a0*y + recursive_term)

        return Dphi
    
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
    
    def covariance_from_separations(self, r0, L0, AB, Ab, aB, ab, sizeZ, sizeS):
        # Stack the separations to vectorize the call to Von Karman Structure function
        sep_stack = np.stack((AB, Ab, aB, ab), axis=0)
        D = self.vk_structure(r0, L0, sep_stack)
        
        covariance = (1/(2*sizeZ*sizeS)) * (-D[0] + D[1] + D[2] - D[3])

        return covariance
    
    def compute_covariance_matrix(self, separations, r0, L0, fractR0, listWFS_Z_params, listWFS_S_params):
        
        nWFS_Z = len(listWFS_Z_params)
        nWFS_S = len(listWFS_S_params)

        row_sizes = []
        for row in range(2 * nWFS_Z):
            AB, Ab, aB, ab = separations[0, row, 0]
            row_sizes.append(AB.shape[0])

        col_sizes = []
        for col in range(2 * nWFS_S):
            AB, Ab, aB, ab = separations[0, 0, col]
            col_sizes.append(AB.shape[1])

        row_offsets = np.cumsum([0] + row_sizes)
        col_offsets = np.cumsum([0] + col_sizes)

        cov_matrix = np.zeros((row_offsets[-1], col_offsets[-1]), dtype=float)

        for iLayer in range(self.nLayers):
            layer_weight = fractR0[iLayer]

            for row in range(2 * nWFS_Z):
                jWFS = row if row < nWFS_Z else row - nWFS_Z
                r0a, r1a = row_offsets[row], row_offsets[row + 1]

                for col in range(2 * nWFS_S):
                    kWFS = col if col < nWFS_S else col - nWFS_S
                    c0, c1 = col_offsets[col], col_offsets[col + 1]

                    AB, Ab, aB, ab = separations[iLayer, row, col]

                    dZ = listWFS_Z_params[jWFS]['subap_size']
                    dS = listWFS_S_params[kWFS]['subap_size']

                    block = self.covariance_from_separations(r0, L0, AB, Ab, aB, ab, dZ, dS)

                    cov_matrix[r0a:r1a, c0:c1] += layer_weight * block

        return cov_matrix

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

            idx = np.random.choice(nSubaps, size=nSel, replace=False)
            subapsSel.append(idx)

        # Compute separations for the subapertures selected
        self.logger.info('LearnEstimator::learn - computing separations for randomly selected subaps.')
        t0 = time.time()
        separations_theo_meas = self.compute_separations_array(subapsSel, subapsSel, self.measWFSmidpoint_X, self.measWFSmidpoint_Y, 
                                                               self.measWFSmidpoint_X, self.measWFSmidpoint_Y)
    
        # Compute covariance matrix
        t1 = time.time()
        cov_mat_theo = self.compute_covariance_matrix(separations_theo_meas, initial_r0, initial_L0, initial_fractionalR0, self.measWFSparams, self.measWFSparams)
        t2 = time.time()
        self.logger.info(f'Separation {t1-t0}, CovMat (1it):{t2-t1}')

        # Load data 

        cov_mat_meas = np.copy(cov_mat_theo)

        # Estimate the parameters

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