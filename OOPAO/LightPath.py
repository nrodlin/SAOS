import numpy as np

from joblib import Parallel, delayed

import logging
import logging.handlers
from queue import Queue

"""
LightPath Module
=================

This module contains the `LightPath` class, used for modeling different line of sights in adaptive optics simulations.
"""

class LightPath:
    def __init__(self, logger=None):
        """
        Initialize the LightPath object, which encapsulates the complete optical train.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for diagnostics.
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger
        # Process variables: updated per iteration
        # Optical Path Difference: is the difference in optical path length (OPL) between two rays of light [m]
        # IMPORTANT: DM is considered transmissive, instead of reflective, so there is no need to multiply by 2
        # the physical displacemente of the actuators to take into account the go-and-return path.
        self.atmosphere_opd = None # in [m]
        self.atmosphere_phase = None # in [rad]
        
        self.dm_opd = None # in [m]
        self.dm_phase = None # in [rad]

        # The OPD and phase that reaches the WFS: atm + dm
        self.wfs_opd = None # in [m]
        self.wfs_phase = None # in [rad]

        self.ncpa_opd = None # in [m]
        self.ncpa_phase = None # in [rad]

        # The OPD and phase that reaches the science camera: atm + dm + ncpa
        self.sci_opd = None # in [m]
        self.sci_phase = None # in [rad]

        # WFS variables
        self.slopes_1D = None # [px] or [rad] depending on the WFS configuration
        self.slopes_2D = None # [px] or [rad] depending on the WFS configuration
        self.wfs_frame = None
        
        # Science variables
        self.sci_frame = None

    # An optical path is defiend, at least, by the source object emitting the light, the atmosphere and the telescope.
    # Optionally, the telescope can have deformable mirror(s), a wavefront sensor, ncpa and a science camera
    def initialize_path(self, src, atm, tel, dm=None, wfs=None, ncpa=None, sci=None):
        """
        Define and configure the optical path with all components.

        Parameters
        ----------
        src : Source
            Light source (NGS, LGS, or Sun).
        atm : Atmosphere
            Atmospheric model.
        tel : Telescope
            Telescope instance.
        dm : DeformableMirror or list, optional
            Deformable mirrors in the path.
        wfs : ShackHartmann, optional
            Wavefront sensor.
        ncpa : object, optional
            Non-common path aberration object.
        sci : object, optional
            Science detector.

        Returns
        -------
        bool
            True if initialization succeeds.
        """
        self.logger.debug('LightPath::initialize_path')
        # Assign the objects to class attributes
        # The objects cannot be affected by paralell processing, their inner set of parameters must be modified externally at the main thread
        self.src = src
        self.atm = atm
        self.tel = tel

        # Now, the optional objects
        if dm is None:
            self.dm = None
        else:
            if isinstance(dm, list): # There might be several DMs, hence the LightPath expects a list
                self.dm = dm
            else:
                self.dm = [dm]
        
        self.wfs = wfs
        self.ncpa = ncpa
        self.sci = sci

        self.logger.debug('LightPath::initialize_path - Path initialized')
        return True
    
    # This method propagates the light through the optical path, updating the variables contained within this class
    # The main process will call this method during a simualtion to update the metrics, its execution is thread-safe
    # Parameters:
    # parallel_atm: if True, the atmosphere method getOPD is executed in parallel with the each DMs getOPD
    # parallel_dms: if True, each DM getOPD is executed in parallel
    # interaction_matrix: if True, the Atmosphere is not added to the DM OPD during the IM measurement
    def propagate(self, parallel_atm=False, parallel_dms=False, interaction_matrix=False):
        """
        Simulate light propagation through the configured optical path.

        Parameters
        ----------
        parallel_atm : bool, optional
            If True, compute atmosphere OPD in parallel.
        parallel_dms : bool, optional
            If True, compute DMs in parallel.
        interaction_matrix : bool, optional
            If True, disable atmosphere during propagation (used for IM calibration).

        Returns
        -------
        bool
            True if propagation was successful.
        """
        self.logger.debug('LightPath::propagate')

        ## The first two tasks consist of getting the effect of the atmosphere and DMs on the light
        tasks  = []

        # Prepare the parallel processing
        if parallel_atm and (not interaction_matrix):
            parallel_dms = True
            tasks.append(delayed(self.atm.getOPD)(self.src))
        
        elif (not parallel_atm) and (not interaction_matrix):
            self.atmosphere_opd = self.atm.getOPD(self.src)
            self.atmosphere_phase = self.atmosphere_opd * (2 * np.pi /self.src.wavelength)
        else: # Avoid interacting with the atmosphere while the IM is being measured
            self.atmosphere_opd = 0
        
        if self.dm is not None:
            nthreads = 1

            if parallel_dms == True:
                nthreads = len(self.dm)
            
            for i in range(len(self.dm)):
                tasks.append(delayed(self.dm[i].get_dm_opd)(self.src))

            # Execute the tasks
            opd_results = Parallel(n_jobs=nthreads, prefer="threads")(tasks)

            # Unpack the results
            self.dm_opd = []
            self.dm_phase = []

            for i in range(len(opd_results)):
                if i == 0 and parallel_atm and (not interaction_matrix):
                    self.atmosphere_opd = opd_results[i]
                    self.atmosphere_phase = self.atmosphere_opd * (2 * np.pi /self.src.wavelength)
                else:
                    self.dm_opd.append(opd_results[i][0])
                    self.dm_phase.append(opd_results[i][1])
        else:
            self.dm_opd = np.zeros_like(self.atmosphere_opd)
            self.dm_phase = np.copy(self.dm_opd)

        # Combine the OPD before reaching the WFS

        self.wfs_opd = self.atmosphere_opd + np.sum(self.dm_opd, axis=0) # Note that for the IM measuring, atmosphere_opd is 0

        self.wfs_phase = self.wfs_opd * (2 * np.pi /self.src.wavelength)

        # Then, measure the slopes at the WFS - if defined
        if self.wfs is not None:
            self.slopes_1D, self.slopes_2D, self.wfs_frame = self.wfs.wfs_measure(self.wfs_phase, self.src)

        # If there are NCPA, add them to the OPD
        # TODO: NCPA class must be upgrade to match the new scheme. It shall return a phase in the pupil plane using the projection of the src as the DMs
        if self.ncpa is not None:
            self.sci_opd = self.wfs_opd + self.ncpa_opd
        else:
            self.sci_opd = np.copy(self.wfs_opd)
        
        self.sci_phase = self.sci_opd * (2 * np.pi /self.src.wavelength)

        # Generate the Science frame, if defined
        if self.sci is not None:
            self.sci_frame = self.sci.get_frame(self.src, self.sci_phase)
        
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
        if not self.external_logger_flag:
            self.queue_listerner.stop()
