"""
Created on March 24 2025
@author: nrodlin
"""

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

        self.tag = 'lightpath'

    # Initialize the buffers with the appropiate values during the initialization of the Light Pat
    def initialize_parameters(self):
        """
        Initialize the internal buffers used during light propagation.

        Returns
        -------
        None
        """
        # Process variables: updated per iteration
        # Optical Path Difference: is the difference in optical path length (OPL) between two rays of light [m]
        # IMPORTANT: DM is considered transmissive, instead of reflective, so there is no need to multiply by 2
        # the physical displacement of the actuators to take into account the go-and-return path.

        # Null buffer
        null_buffer = np.zeros_like(self.tel.pupil, dtype=np.float64)

        self.vibration_opd   = null_buffer.copy() # in [m]
        self.vibration_phase = null_buffer.copy() # in [rad]

        self.localSeeing_opd   = null_buffer.copy() # in [m]
        self.localSeeing_phase = null_buffer.copy() # in [rad]        
        
        self.atmosphere_opd   = null_buffer.copy() # in [m]
        self.atmosphere_phase = null_buffer.copy() # in [rad]
        
        self.dm_opd   = null_buffer.copy() # in [m]
        self.dm_phase = null_buffer.copy() # in [rad]

        # The OPD and phase that reaches the WFS: atm + vibration + localSeeing + dm
        self.wfs_opd   = null_buffer.copy() # in [m]
        self.wfs_phase = null_buffer.copy() # in [rad]

        if self.ncpa:
            self.ncpa_opd    = self.ncpa.getPhase() # in [m]
            self.ncpa_phase  = self.ncpa_opd * (2*np.pi/self.src.wavelength) # in [rad]
        else:
            self.ncpa_opd   = null_buffer.copy() # in [m]
            self.ncpa_phase = null_buffer.copy() # in [rad]

        # The OPD and phase that reaches the science camera: atm + vibration + localSeeing + dm + ncpa
        self.sci_opd   = null_buffer.copy() # in [m]
        self.sci_phase = null_buffer.copy() # in [rad]

        # WFS variables
        if self.wfs:
            if self.src.tag == 'sun': # We need to generate the pseudo refence
                self.pseudo_ref, self.reference_slopes = self.wfs.initialize_wfs(self.tel, self.src)
            self.slopes_1D = np.zeros(self.wfs.nSignal) # [px] or [rad] depending on the WFS configuration
            self.slopes_2D = np.zeros((2*self.wfs.nSubap, self.wfs.nSubap)) # [px] or [rad] depending on the WFS configuration
            self.wfs_frame = None

            # Error buffer --> the controller will access this variable, that contains a buffer of the last N samples to provide the error inputs when delays are simulated
            self.error_measurement = np.zeros((self.slopes_1D.shape[0], self.delay+1)) # [px] or [rad] depending on the WFS configuration 
        
        # Science variables
        if self.sci:
            self.sci_frame = None # Frame, noise free and of exposure equivalent to 1 sampling cycle. Normalize so that sum of energy is 1.
            self.long_exposure_frame = None # Frame of exposure setup by the user. Can be noisy (user-config). The frame is scaled by the number of photons.
            self.long_exp_cumulative = []
            self.decimation_counter = 0


    # An optical path is defiend, at least, by the source object emitting the light and the telescope.
    # Optionally, we can simulate vibrations, local seeing, atmosphere, deformable mirror(s), a wavefront sensor, ncpa and a science camera. 
    def initialize_path(self, src, tel, atm=None, dm=None, wfs=None, ncpa=None, sci=None, vibration=None, delay=0, localSeeing=None):
        """
        Define and configure the optical path with all components.

        Parameters
        ----------
        src : Source
            Light source (NGS, LGS, or Sun).
        tel : Telescope
            Telescope instance.
        atm : Atmosphere, optional
            Atmospheric model.        
        dm : DeformableMirror or list, optional
            Deformable mirrors in the path.
        wfs : ShackHartmann, optional
            Wavefront sensor.
        ncpa : NCPA, optional
            Non-common path aberration object.
        sci : Science camera, optional
            Science detector.
        vibration : Vibration source, optional
            Vibration object.
        delay : int, optional
            Light Path delay in samples.
        localSeeing : Local Seeing, optional
            Local Seeing object

        Returns
        -------
        bool
            True if initialization succeeds.
        """
        self.logger.debug('LightPath::initialize_path')
        # Assign the objects to class attributes
        # The objects cannot be affected by paralell processing, their inner set of parameters must be modified externally at the main thread
        # Mandatory objects: source and telescope
        self.src = src
        self.tel = tel

        # Atmosphere
        self.atm = atm

        # Mirror object(s)
        if dm is None:
            self.dm = None
        else:
            if isinstance(dm, list): # There might be several DMs, hence the LightPath expects a list
                self.dm = dm
            else:
                self.dm = [dm]
        
        # Sensor objects

        self.wfs = wfs
        self.sci = sci
        # Special disturbance object

        self.vibration   = vibration
        self.localSeeing = localSeeing
        self.ncpa        = ncpa

        # Temporal management variables
        self.delay = int(np.round(delay))
        self.iteration = 0

        # Initialize the buffers
        self.initialize_parameters()
        self.logger.debug('LightPath::initialize_path - Path initialized')
        return True
    
    # This method propagates the light through the optical path, updating the variables contained within this class
    # The main process will call this method during a simualtion to update the metrics, its execution is thread-safe
    # Parameters:
    # temporal_tick: if True, advances the time in the simulation
    # parallel_dms: if True, each DM getOPD is executed in parallel
    # interaction_matrix: if True, the Atmosphere is not added to the DM OPD during the IM measurement
    def propagate(self, temporal_tick, parallel_dms=False, interaction_matrix=False):
        """
        Simulate light propagation through the configured optical path.

        Parameters
        ----------
        temporal_tick : bool
            If True, the simulation time advances 1 sample.
        interaction_matrix : bool, optional
            If True, disable atmosphere during propagation (used for IM calibration).            
        parallel_dms : bool, optional
            If True, compute DMs in parallel.
        Returns
        -------
        bool
            True if propagation was successful.
        """
        self.logger.debug('LightPath::propagate')
        ## Vibrations
        if (not interaction_matrix) and (self.vibration is not None):
            self.vibration_opd = self.vibration.getCurrentVibrations(self.iteration)
            self.vibration_phase = self.vibration_opd * (2*np.pi/self.src.wavelength)
        else:
            self.vibration_opd   *= 0
            self.vibration_phase *= 0
        
        ## Local seeing
        if (not interaction_matrix) and (self.localSeeing is not None):
            self.localSeeing_opd = self.localSeeing.getCurrentOPD(self.iteration)
            self.localSeeing_phase = self.localSeeing_opd * (2*np.pi/self.src.wavelength)
        else:
            self.localSeeing_opd   *= 0
            self.localSeeing_phase *= 0

        ## Project the atmosphere
     
        if (not interaction_matrix) and (self.atm is not None):
            self.atmosphere_opd = self.atm.getOPD(self.src)
            self.atmosphere_phase = self.atmosphere_opd * (2 * np.pi /self.src.wavelength)
        else: # Avoid interacting with the atmosphere while the IM is being measured
            self.atmosphere_opd   *= 0
            self.atmosphere_phase *= 0

        # Project the DM
        if self.dm is not None:
            tasks = []
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
                self.dm_opd.append(opd_results[i][0].copy())
                self.dm_phase.append(opd_results[i][1].copy())
        else:
            # Set DM OPD and phase to zero
            self.dm_opd   *= 0
            self.dm_phase *= 0

        # Combine the OPD before reaching the WFS
        # Note that for the IM measuring, atmosphere_opd and the vibration_opd are 0

        self.wfs_opd = self.vibration_opd + self.localSeeing_opd + self.atmosphere_opd + np.sum(self.dm_opd, axis=0)
        self.wfs_opd *= self.tel.pupil # apply pupil mask to the OPD

        self.wfs_phase = self.wfs_opd * (2 * np.pi /self.src.wavelength)

        # Then, measure the slopes at the WFS - if defined
        if self.wfs is not None:
            if self.src.tag == 'sun':
                self.slopes_1D, self.slopes_2D, self.wfs_frame, _ = self.wfs.wfs_measure(self.wfs_phase, self.src, self.pseudo_ref, self.reference_slopes)
            else:
                self.slopes_1D, self.slopes_2D, self.wfs_frame = self.wfs.wfs_measure(self.wfs_phase, self.src)

            self.error_measurement[:, (self.iteration+1)%self.error_measurement.shape[1]] = self.slopes_1D.copy()

        # If there are NCPA, add them to the OPD
        self.sci_opd = self.wfs_opd + self.ncpa_opd
        self.sci_opd *= self.tel.pupil # apply pupil mask to the OPD
        
        self.sci_phase = self.sci_opd * (2 * np.pi /self.src.wavelength)

        # Generate the Science frame, if the time advances
        if (self.sci is not None) and (not interaction_matrix) and (temporal_tick):
            get_frame = False
            # Check Science cam decimation
            if self.sci.decimation > 0:
                # Then, we have to check the decimation
                if (self.decimation_counter % self.sci.decimation) == 0:
                    get_frame = True
            else:
                get_frame = True
            # If we have to get the frame, do it, if not empty the sci img buffer for the publishing modules
            if get_frame:
                # Get short exp frame (noise-free)
                noise_free_frame = self.sci.get_frame(self.src, self.sci_phase) # Noise free frame
                self.sci_frame = noise_free_frame.copy()
                # Append current short exp frame to the cumulative list
                self.long_exp_cumulative.append(self.sci_frame)
                # Now, manage the long exposure frame --> The number of frames accumulated let us know the time exposed
                # Incorporate the decimation factor since we only acquire a frame every decimation steps
                decimation_factor = max(1, self.sci.decimation) if self.sci.decimation > 0 else 1
                exposured_time = len(self.long_exp_cumulative) * decimation_factor * self.tel.samplingTime
                # Check if we have exposed the required time
                if exposured_time >= self.sci.integrationTime:
                    # Add the frames accumulated
                    longExp = np.sum(self.long_exp_cumulative, 0)
                    # Normalize energy to 1
                    total_energy = np.sum(longExp)
                    if total_energy > 0:
                        longExp /= total_energy
                    # Check if the user wants to add noise
                    if self.sci.cam.noiseFlag:
                        longExp = self.sci.apply_noise(longExp, self.src.nPhoton * self.sci.integrationTime)
                    else:
                        longExp = longExp * self.src.nPhoton * self.sci.integrationTime * self.sci.lightRatio
                    # Save the frame 
                    self.long_exposure_frame = np.squeeze(longExp).copy()
                    # Reset the cumulative frame
                    self.long_exp_cumulative = []
                else:
                    self.long_exposure_frame = None
            else:
                self.sci_frame = None
                self.long_exposure_frame = None
        
        # Advance the simulation time, if required
        if temporal_tick:
            self.iteration += 1
            if self.sci:
                self.decimation_counter += 1
        
        return True
    
    def get_wavefront_error(self):
        """
        Return the wavefront error measurement, accounting for latency delay.

        Returns
        -------
        np.ndarray or bool
            Delayed measurement array, zeros if not enough iterations have elapsed,
            or False if no WFS is attached to this light path.
        """

        if self.wfs:
            if (self.iteration-self.delay) >= 0:
                index = (self.iteration - self.delay) % self.error_measurement.shape[1]
                return self.error_measurement[:, index]
            else:
                return np.zeros_like(self.slopes_1D)
        else:
            return False

    
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
