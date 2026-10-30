# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:18:03 2024

@authors: astriffl & cheritie

Update on March 26 2025
@author: nrodlin

Major implementation belong to OOPAO authors, the module now is just adjusted to fit the code style of SAOS.
"""

import threading

import numpy as np



import logging
from .LoggingHelper import get_logger


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
        
        self.logger = logger if logger is not None else get_logger()

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
        # Noise streams: every LightPath connected to this detector gets its own generator (register_stream), so that the noise of a
        # path does not depend on the order in which paths sharing the detector are scheduled in threads
        self._stream_entropy    = np.random.SeedSequence(self.random_state).entropy
        self._streams           = {}
        self._n_streams         = 0
        self._stream_lock       = threading.Lock()

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
            self.dark_calibration_frame = self.dark_calibration_frame / self.darkCalibration     # mean dark [DN]
            self.shotNoise = shotNoise
            self.calibrating = False
        else:
            self.dark_calibration_frame = np.zeros_like(self.frame)


    def register_stream(self):
        """
        Assign a noise stream to a new user of the detector (LightPath.initialize_path): streams are numbered 0, 1, 2... in the
        order in which the light paths are connected to this detector.

        Returns
        -------
        int
            Stream index, to be passed to integrate(stream=...).
        """
        with self._stream_lock:
            stream = self._n_streams
            self._n_streams += 1
        return stream

    def _generator(self, stream):
        """Random generator of a noise stream (None: the detector's own generator). Each stream is seeded from the detector seed
        and its index (SeedSequence spawn key), so its noise does not depend on the other streams."""
        if stream is None:
            return self.randomGenerator
        with self._stream_lock:
            rng = self._streams.get(stream)
            if rng is None:
                rng = np.random.default_rng(np.random.SeedSequence(self._stream_entropy, spawn_key=(int(stream),)))
                self._streams[stream] = rng
        return rng

    def integrate(self, input_frame, input_photons, stream=None):
        """
        Integrate the noise free frame, adding the corresponding noise.
        
        Parameters
        ----------
        input_frame : np.ndarray
            The noise-free frame to integrate.
        input_photons : int
            The number of photons received in the current sampling time.
        stream : int, optional
            Noise stream (register_stream) of the light path; None uses the detector's own generator.

        Returns
        -------
        np.ndarray or None
            The integrated frame with noise added, when the integration time is completed. 
            If None, the integration is ongoing.
        """
        
        # Integrate
        if self.noiseFlag:
            # Get the noisy frame
            frame = self.readout(input_frame, input_photons, stream)
        else:
            frame = input_frame

        return frame
        
    def readout(self, input_frame, photons, stream=None):
        """
        Simulate the readout process of the detector including noises and quantification.

        Parameters
        ----------
        input_frame : np.ndarray
            The noise-free input frame.
        photons : int or float
            Number of incident photons.
        stream : int, optional
            Noise stream (register_stream); None uses the detector's own generator.

        Returns
        -------
        np.ndarray
            The simulated quantized and saturated frame.
        """
        rng = self._generator(stream)

        # 1: Normalize energy in the frame to 1
        energy = np.sum(input_frame)
        norm_frame = input_frame.copy()
        if energy > 0:
            norm_frame /= np.sum(input_frame)

        # 2: Scale by the number of photons
        photons_frame = np.round(norm_frame * photons) # [photons]

        # 3-4: Photon to electron conversion and shot noise. The detected electrons are a binomial thinning (QE) of the
        # Poisson photon stream, i.e. Poisson(QE * photons). Applying QE after the Poisson draw would underestimate the
        # shot noise by sqrt(QE)
        if self.shotNoise:
            electron_noisy_frame = rng.poisson(self.quantumEfficiency * photons_frame).astype(np.float64) # [e-]
        else:
            electron_noisy_frame = self.quantumEfficiency * photons_frame # [e-]

        # 5: Dark current noise
        dark_current_map = np.ones_like(input_frame) * self.darkCurrent * self.integrationTime
        electron_noisy_frame += rng.poisson(dark_current_map) # [e-]

        # 6: Saturate

        if electron_noisy_frame.max() > self.fullWellCapacity:
            self.logger.warning('Detector::readout - The sensor is saturating.')
        
        electron_noisy_frame = np.clip(electron_noisy_frame, a_min=0, a_max=self.fullWellCapacity) # [e-]

        # 7: EMCCD gain
        if self.sensorType == 'EMCCD':
            electron_noisy_frame *= self.gain # [e-]
        
        # 8: Readout noise

        electron_noisy_frame += rng.normal(loc=0.0, scale=self.readoutNoise, size=electron_noisy_frame.shape) # [e-]

        # 9: CCD/CMOS gain

        if self.sensorType == 'CCD' or self.sensorType == 'CMOS':
            electron_noisy_frame *= self.gain # [e-]
        
        # 10: Quantification
        if self.quantization_conversion == 0:
            if np.isinf(self.fullWellCapacity):
                # Automatic conversion with an infinite full well: 1 e-/DN, so that the absolute signal scale is kept
                self.quantizationNoise = 1 / np.sqrt(12) # [e-]
                quantized_frame = electron_noisy_frame # [counts]
            else:
                self.quantizationNoise = self.fullWellCapacity / (np.sqrt(12) * 2**(self.nBits)) # [e-]
                quantized_frame = (electron_noisy_frame / self.fullWellCapacity) * 2**(self.nBits)  # [counts]
        else:
            self.quantizationNoise = self.quantization_conversion / np.sqrt(12)  # [e-]
            quantized_frame = electron_noisy_frame / self.quantization_conversion  # [counts]
        
        # Rounding to the nearest count (truncation would bias the output by -0.5 DN)
        quantized_saturated_frame = np.clip(np.round(quantized_frame), a_min=0, a_max=(2**self.nBits) - 1) # [counts]
        self.saturation_level = 100 * (quantized_saturated_frame.max() / 2**self.nBits)  # [%]

       
        # Apply dark calibration: the calibrated frame is clipped at 0 (no bias pedestal: unsigned output, negative residuals
        # of the dark subtraction would otherwise wrap to ~2^bits)
        if self.darkCalibration and not self.calibrating:
            quantized_saturated_frame = np.clip(np.round(quantized_saturated_frame - self.dark_calibration_frame), a_min=0, a_max=None)

        # 11: Set precision 
            
        return quantized_saturated_frame.astype(self.dataType)
