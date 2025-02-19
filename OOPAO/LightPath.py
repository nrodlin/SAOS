import time

import numpy as np
import torch
from joblib import Parallel, delayed

import logging
import logging.handlers
from queue import Queue

class LightPath:
    def __init__(self, logger):
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.logger = logger
        # Process variables: updated per iteration
        # Optical Path Difference: is the difference in optical path length (OPL) between two rays of light [m]
        # IMPORTANT: DM is considered transmissive, instead of reflective, so there is no need to multiply by 2
        # the physical displacemente of the actuators to take into account the go-and-return path.
        self.atmosphere_opd = None # in [m]
        self.atmosphere_phase = None # in [rad]
        
        self.dms_opd = None # in [m]
        self.dms_phase = None # in [rad]

        self.ncpa_opd = None # in [m]
        self.ncpa_phase = None # in [rad]

        self.telescope_opd = None # in [m]
        self.telescope_phase = None # in [rad]

        # WFS variables
        self.slopes_1D = None # [px] or [rad] depending on the WFS configuration
        self.slopes_2D = None # [px] or [rad] depending on the WFS configuration
        self.wfs_frame = None
        # DMs variables
        self.dm_coeffs = None
        # Science variables
        self.sci_frame = None
        self.quality_metrics = None

    # An optical path is defiend, at least, by the source object emitting the light, the atmosphere and the telescope.
    # Optionally, the telescope can have deformable mirror(s), a wavefront sensor, ncpa and a science camera
    def initialize_path(self, src, atm, tel, dm=None, wfs=None, ncpa=None, sci=None):
        self.logger.debug('LightPath::initialize_path')
        # Assign the objects to class attributes
        # The objects cannot be affected by paralell processing, their inner set of parameters must be modified externally at the main thread
        self.src = src
        self.atm = atm
        self.tel = tel

        # Now, handle the optional objects
        self.dm = dm
        self.wfs = wfs
        self.ncpa = ncpa
        self.sci = ncpa        
        self.logger.info('LightPath::initialize_path - Path initialized')
        return True
    
    # This method propagates the light through the optical path, updating the variables contained within this class
    # The main process will call this method during a simualtion to update the metrics, its execution is thread-safe
    def propagate(self):
        self.logger.debug('LightPath::propagate')
        return True
    
    def setup_logging(self, logging_level=logging.WARNING):
        #
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
    
    def __del__(self):
        self.queue_listerner.stop()

       
