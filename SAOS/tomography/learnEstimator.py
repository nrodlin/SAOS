import numpy as np
import torch

import logging
import logging.handlers
from queue import Queue

import matplotlib.pyplot as plt

class LearnEstimator:
    """
    Estimate atmospheric parameters from WFS telemetry.
    """

    def __init__(self, 
                 dataset_filename, 
                 output_path, 
                 altitudes, 
                 zenith, 
                 atm_guess,
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

        self.tag                            = 'learnEstimator'

        self.dataset_filename = dataset_filename
        self.output_path = output_path

        self.initial_r0 = atm_guess['r0']
        self.initial_L0 = atm_guess['L0']

        self.altitudes = np.array(altitudes)
        self.nLayers   = len(self.altitudes)

        self.zenith    = zenith

        self.initial_fractionalR0 = np.array(atm_guess['fractionalR0'])
        self.initial_windSpeedX = np.array(atm_guess['windSpeedX'])
        self.initial_windSpeedY = np.array(atm_guess['windSpeedY'])

        # Correct altitudes by zenith
        self.altitudes = self.altitudes / np.cos((self.zenith/180)*np.pi)

        # Check input parameters

        if self.initial_fractionalR0.shape[0] != self.nLayers:
            raise ValueError('LearnEstimator::__init__ - Fractional r0 dimensions does not correspond the number of layers')
        if self.initial_windSpeedX.shape[0] != self.nLayers:
            raise ValueError('LearnEstimator::__init__ - Vx dimensions does not correspond the number of layers')
        if self.initial_windSpeedY.shape[0] != self.nLayers:
            raise ValueError('LearnEstimator::__init__ - Vy dimensions does not correspond the number of layers')        
        
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