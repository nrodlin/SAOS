# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:18:03 2024

@authors: astriffl & cheritie

Update on March 26 2025
@author: nrodlin

Major implementation belong to OOPAO authors, the module now is just adjusted to fit the code style of SAOS.
"""

import numpy as np



import logging
import logging.handlers
from queue import Queue


"""
Detector sensor Module
=================

This module contains the `Detector` class, used for modeling a camera in adaptive optics simulations.
"""

class Detector:
    def __init__(self,
                 nPix:int,
                 samplingTime:float,
                 fullWellCapacity:int=np.inf,
                 nBits:int=12,                 
                 quantumEfficiency:float=1,
                 shotNoise:bool=0,
                 darkCurrent:float=0,
                 readoutNoise:float=0,
                 gain:float=1,
                 quantization_conversion:float=0,
                 sensorType:str='CCD',
                 darkCalibration:bool=True,
                 noiseFlag:bool=True,
                 logger=None,
                 **kwargs):
        '''
        Initialize a Detector object to simulate real detector effects like noise, saturation, and quantization.

        Parameters
        ----------
        nPix : int
            Resolution of the detector [px].
        samplingTime : float
            Minimal sampling time for the camera [s].
        fullWellCapacity : int, optional
            Full Well Capacity of pixels [e-]. Default, np.inf.
        nBits : int, optional
            Bit depth for quantization. Default is 12, shall be >= 8
        quantumEfficiency : float, optional
            Quantum efficiency (0-1). Default is 1.
        shotNoise : bool, optional
            Shot noise flag. Default disabled, 0.
        darkCurrent : float, optional
            Dark current [e-]. Default disabled, 0.
        readoutNoise : float, optional
            Readout noise [e-]. Default disabled, 0.
        gain : float, optional
            Gain of the detector. Default is 1.
        quantization_conversion : float, optional.
            Conversion gain to discretize the measurement [e-/px]. Default disabled, 0.
        sensorType : str, optional
            Sensor type ('CCD', 'CMOS', 'EMCCD'). Default is 'CCD'.
        darkCalibration : int, optional
            Number of frames to calibrate the dark. By default disabled, 0.
        noiseFlag : bool, optional
            If True, the detector adds noise using the params/default config. By default, True.            
        logger : logging.Logger, optional
            Logger instance for diagnostics.
        **kwargs : dict, optional
            Additional keyword arguments.

            randomState : int, optional
                Seed for the random number generator. Default is None.
            integrationTime : float, optional
                Integration time for the detector [s]. Default is None.
        '''
        
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Map parameters to the 

        self.nPix                    = nPix
        self.samplingTime            = samplingTime


        self.fullWellCapacity        = fullWellCapacity
        self.nBits                   = nBits
        self.quantumEfficiency       = quantumEfficiency

        self.shotNoise               = shotNoise
        self.darkCurrent             = darkCurrent
        self.readoutNoise            = readoutNoise

        self.gain                    = gain
        self.quantization_conversion = quantization_conversion
        self.sensorType              = sensorType

        self.noiseFlag          = noiseFlag

        # Check consistency of parameters
        if self.nBits < 8:
            self.logger.warning('Detector::__init__ - Number of bits is below the acceptable threshold, defaulting to 8.')
            self.nBits = 8
        if self.gain == 0:
            self.logger.warning('Detector::__init__ - The gain is zero, the output will be zero.')
        elif self.gain < 0:
            self.logger.warning('Detector::__init__ - Negative gain, defaulting to 1.')
            self.gain = 1
        if not sensorType in ['CCD','CMOS','EMCCD']:
            self.logger.warning(f"Unknown sensor type '{sensorType}' specified. Defaulting to 'CCD'.")
            self.sensorType = 'CCD'

        self.darkCalibration   = darkCalibration
        
        
        # Map kwargs if any:
        self.integrationTime    = kwargs.get('integrationTime', self.samplingTime)
        self.random_state       = kwargs.get('randomState', None)

        self.tag                = 'detector'  

        #### Initialize the class

        self.frame                  = np.zeros((self.nPix,self.nPix)) # stores the result frame
        self.dark_calibration_frame = np.zeros_like(self.frame)  # stores the dark calibration frame
        
        self.randomGenerator    = np.random.default_rng(seed=self.random_state)

        if self.nBits == 8:
            self.dataType = np.uint8   
        elif self.nBits <= 16:
            self.dataType = np.uint16
        elif self.nBits <= 32:
            self.dataType = np.uint32
        else:
            self.dataType = np.uint64

        # Define noise types
        self.peak_signal        = 0
        self.photon_noise_sigma = 0
        self.dark_noise_sigma   = 0
        self.quantizationNoise  = 0

        # Calibrate dark if specified
        self.dark_calibration_frame = np.zeros_like(self.frame)

        if self.noiseFlag:
            self.dark_calibration_frame.astype(self.dataType)

        if self.darkCalibration > 0:
            self.shotNoise = 0 # Disable shot noise to avoid crash in the Poisson of lam=0
            self.calibrating  = True
            for _ in range(self.darkCalibration):
                self.dark_calibration_frame += self.integrate(np.zeros_like(self.frame, dtype=float), 0)
            self.dark_calibration_frame = self.dark_calibration_frame // self.darkCalibration
            self.shotNoise = shotNoise
            self.calibrating = False
        else:
            self.dark_calibration_frame = np.zeros_like(self.frame)


    def integrate(self, input_frame, input_photons):
        """
        Integrate the noise free frame, adding the corresponding noise.
        
        Parameters
        ----------
        input_frame : np.ndarray
            The noise-free frame to integrate.
        input_photons : int
            The number of photons received in the current sampling time.

        Returns
        -------
        np.ndarray or None
            The integrated frame with noise added, when the integration time is completed. 
            If None, the integration is ongoing.
        """
        
        # Integrate
        if self.noiseFlag:
            # Get the noisy frame
            frame = self.readout(input_frame, input_photons)
        else:
            frame = input_frame

        return frame
        
    def readout(self, input_frame, photons):
        """
        Simulate the readout process of the detector including noises and quantification.

        Parameters
        ----------
        input_frame : np.ndarray
            The noise-free input frame.
        photons : int or float
            Number of incident photons.

        Returns
        -------
        np.ndarray
            The simulated quantized and saturated frame.
        """

        # 1: Normalize energy in the frame to 1
        energy = np.sum(input_frame)
        norm_frame = input_frame.copy()
        if energy > 0:
            norm_frame /= np.sum(input_frame)

        # 2: Scale by the number of photons
        photons_frame = np.round(norm_frame * photons) # [photons]

        # 3: Photon noise
        if self.shotNoise:
            photon_noisy_frame = self.randomGenerator.poisson(photons_frame) # [photons]
        else:
            photon_noisy_frame = photons_frame
        
        # 4: Convert from photons to electrons

        electron_noisy_frame = self.quantumEfficiency * photon_noisy_frame # [e-]

        # 5: Dark current noise
        dark_current_map = np.ones_like(input_frame) * self.darkCurrent * self.integrationTime
        electron_noisy_frame += self.randomGenerator.poisson(dark_current_map) # [e-]

        # 6: Saturate

        if electron_noisy_frame.max() > self.fullWellCapacity:
            self.logger.warning('Detector::readout - The sensor is saturating.')
        
        electron_noisy_frame = np.clip(electron_noisy_frame, a_min=0, a_max=self.fullWellCapacity) # [e-]

        # 7: EMCCD gain
        if self.sensorType == 'EMCCD':
            electron_noisy_frame *= self.gain # [e-]
        
        # 8: Readout noise

        electron_noisy_frame += self.randomGenerator.normal(loc=0.0, scale=self.readoutNoise, size=electron_noisy_frame.shape) # [e-]

        # 9: CCD/CMOS gain

        if self.sensorType == 'CCD' or self.sensorType == 'CMOS':
            electron_noisy_frame *= self.gain # [e-]
        
        # 10: Quantification
        if self.quantization_conversion == 0:
            if self.fullWellCapacity is np.inf:
                self.quantizationNoise = 0 # [e-]
                quantized_frame = (electron_noisy_frame / electron_noisy_frame.max()) * 2**(self.nBits) # [counts]
            else:
                self.quantizationNoise = self.fullWellCapacity / (np.sqrt(12) * 2**(self.nBits)) # [e-]
                quantized_frame = (electron_noisy_frame / self.fullWellCapacity) * 2**(self.nBits)  # [counts]
        else:
            self.quantizationNoise = self.quantization_conversion / np.sqrt(12)  # [e-]
            quantized_frame = electron_noisy_frame / self.quantization_conversion  # [counts]
        
        quantized_saturated_frame = np.clip(quantized_frame, a_min=0, a_max=(2**self.nBits) - 1) # [counts]
        self.saturation_level = 100 * (quantized_saturated_frame.max() / 2**self.nBits)  # [%]

       
        # Apply dark calibration

        if self.darkCalibration and not self.calibrating:
            quantized_saturated_frame -= self.dark_calibration_frame

        # 11: Set precision 
            
        return quantized_saturated_frame.astype(self.dataType)       
    
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