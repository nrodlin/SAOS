# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:18:03 2024

@authors: astriffl & cheritie

Update on March 26 2025
@author: nrodlin
"""

import numpy as np
import time
from OOPAO.tools.tools import set_binning

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
                 nRes:int,
                 samplingTime:float,
                 integrationTime:float=None,
                 bits:int=None,
                 output_precision:int =None,
                 FWC:int=None,
                 gain:int=1,
                 sensor:str='CCD',
                 QE:float=1,
                 binning:int=1,
                 darkCurrent:float=0,
                 readoutNoise:float=0,
                 photonNoise:bool=False,
                 backgroundNoise:bool=False,
                 backgroundFlux:float=None,
                 backgroundMap:float = None,
                 logger=None):
        '''
        Initialize a Detector object to simulate real detector effects like noise, saturation, and quantization.

        Parameters
        ----------
        nRes : int
            Resolution in pixels of the detector.
        samplingTime : float
            Minimal sampling time for the camera [s].
        integrationTime : float, optional
            Integration time of the detector in [s]. Default is sampling time.
        bits : int, optional
            Bit depth for quantization. Default is None.
        output_precision : int, optional
            Output precision in bits. Default is None.
        FWC : int, optional
            Full Well Capacity of pixels in electrons. Default is None.
        gain : int, optional
            Gain of the detector. Default is 1.
        sensor : str, optional
            Sensor type ('CCD', 'CMOS', 'EMCCD'). Default is 'CCD'.
        QE : float, optional
            Quantum efficiency (0-1). Default is 1.
        binning : int, optional
            Binning factor. Default is 1.
        darkCurrent : float, optional
            Dark current in e-/pixel/s. Default is 0.
        readoutNoise : float, optional
            Readout noise in e-/pixel. Default is 0.
        photonNoise : bool, optional
            Enable photon noise. Default is False.
        backgroundNoise : bool, optional
            Enable background noise. Default is False.
        backgroundFlux : float, optional
            Background flux level. Default is None.
        backgroundMap : float, optional
            Background noise map. Default is None.
        logger : logging.Logger, optional
            Logger instance for diagnostics.
        '''
        
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        self.resolution         = nRes
        self.samplingTime       = samplingTime

        if integrationTime is None:
            self.integrationTime = samplingTime
        else:
            self.integrationTime = integrationTime

        self.bits               = bits
        self.output_precision   = output_precision
        self.FWC                = FWC
        self.gain               = gain
        self.sensor             = sensor
        if self.sensor not in ['EMCCD','CCD','CMOS']:
            raise ValueError("Sensor must be 'EMCCD', 'CCD', or 'CMOS'")
        self.QE                 = QE
        self.binning            = binning
        self.darkCurrent        = darkCurrent
        self.readoutNoise       = readoutNoise
        self.photonNoise        = photonNoise        
        self.backgroundNoise    = backgroundNoise   
        self.backgroundFlux     = backgroundFlux
        self.backgroundMap      = backgroundMap
        
        self.frame              = np.zeros([self.resolution,self.resolution])          
                
        self.saturation         = 0
        self.tag                = 'detector'   
        self._integrated_time   = 0

        # Precision
        if self.output_precision is not None:
            self.set_output_precision(self.output_precision)
        else:
            self.set_output_precision(self.bits)

        # Noise 
        if self.FWC is not None:
            self.SNR_max = self.FWC / np.sqrt(self.FWC)
        else:
            self.SNR_max = np.nan

        if self.FWC:
            self.quantification_noise = self.FWC * 2**(-self.bits) / np.sqrt(12)
        else:
            self.quantification_noise = 0

        self.dark_shot_noise = np.sqrt(self.darkCurrent * self.integrationTime)
        
        # random state to create random values for the noise
        self.random_state_photon_noise      = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_readout_noise     = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_background_noise  = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_dark_shot_noise   = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise       


    def rebin(self,arr, new_shape):
        '''
        Rebin an array to a new shape by averaging values.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        new_shape : tuple
            New shape of the array.

        Returns
        -------
        np.ndarray
            Rebinned array.
        '''            
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                    new_shape[1], arr.shape[1] // new_shape[1])        
        out = (arr.reshape(shape).mean(-1).mean(1)) * (arr.shape[0] // new_shape[0]) * (arr.shape[1] // new_shape[1])        
        return out
        
        
    def set_binning(self, array, binning_factor,mode='sum'):
        '''
        Apply binning to an array.

        Parameters
        ----------
        array : np.ndarray
            Input array.
        binning_factor : int
            Binning factor.
        mode : str, optional
            Mode of binning ('sum' or 'mean'). Default is 'sum'.

        Returns
        -------
        np.ndarray
            Binned array.
        '''        
        frame = set_binning(array, binning_factor,mode)
        return frame

   
    def set_output_precision(self, value):
        '''
        Set the output precision in bits.

        Parameters
        ----------
        value : int
            Bit depth for output precision.
        '''        
            
        if value ==8:
            self.output_precision = np.uint8
            
        elif value ==16:
            self.output_precision = np.uint16

        elif value ==32:
            self.output_precision = np.uint32
        
        elif value ==64:
            self.output_precision = np.uint64
        else:
           self.output_precision = int     
           
        return 
    
    def conv_photon_electron(self,frame):
        '''
        Convert photon counts to electrons using the detector's quantum efficiency.

        Parameters
        ----------
        frame : np.ndarray
            Input frame with photon counts.

        Returns
        -------
        np.ndarray
            Frame converted to electrons.
        '''        
        frame = (frame * self.QE)
        return frame
        
    
    def set_saturation(self,frame):
        '''
        Apply saturation limits based on full well capacity (FWC).

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Clipped frame respecting FWC.
        '''        
        
        saturation = (100*frame.max()/self.FWC)
        
        if frame.max() > self.FWC:
            self.logger.warning('Detector::set_saturation - The detector is saturating, %.1f %%' % saturation)
        
        return np.clip(frame, a_min = 0, a_max = self.FWC)
    
    
    def digitalization(self, frame):
        '''
        Apply digital quantization to a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Quantized frame.
        '''
        if self.FWC is None:
            return (frame / frame.max() * 2**self.bits).astype(self.output_precision)
        else:
            saturation           = (100*frame.max()/self.FWC)

            if frame.max() > self.FWC:
                self.logger.warning('Detector::digitalization the ADC is saturating (gain applyed %i), %.1f %%' % (self.gain, saturation))
            
            frame = (frame / self.FWC * (2**self.bits-1)).astype(self.output_precision) 
            return np.clip(frame, a_min=frame.min(), a_max=2**self.bits-1)

    
    def set_photon_noise(self,frame):
        '''
        Apply Poisson-distributed photon noise.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Frame with photon noise.
        '''        
        return self.random_state_photon_noise.poisson(frame)


    def set_background_noise(self,frame):
        '''
        Apply background noise to the detector frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Frame with background noise applied.
        '''        
        if hasattr(self,'backgroundFlux') is False or self.backgroundFlux is None:
            raise ValueError('The background map backgroundFlux is not properly set. A map of shape '+str(frame.shape)+' is expected')
        else:
            backgroundNoiseAdded = self.random_state_background.poisson(self.backgroundFlux)
            frame += backgroundNoiseAdded
            return frame

    def set_readout_noise(self,frame):
        '''
        Apply readout noise to a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Frame with readout noise.
        '''        
        noise = (np.round(self.random_state_readout_noise.randn(frame.shape[0],frame.shape[1])*self.readoutNoise)).astype(int)  #before np.int64(...)
        frame += noise
        return frame
    
    
    def set_dark_shot_noise(self,frame):
        '''
        Apply dark shot noise to the detector frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Frame with dark shot noise applied.
        '''        
        dark_current_map = np.ones(frame.shape) * (self.darkCurrent * self.integrationTime)
        dark_shot_noise_map = self.random_state_dark_shot_noise.poisson(dark_current_map)
        frame += dark_shot_noise_map
        return frame 
    
    def remove_background(self,frame):
        '''
        Remove background from the detector frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Frame with background removed.
        '''        
        try:
            frame -= self.backgroundMap        
            return frame
        except:
            raise AttributeError('Detector::remove_background - The shape of the background map does not match the frame shape')
    
    
    def readout(self, frame):
        '''
        Simulate detector readout including noise and quantization.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Processed detector frame.
        '''            
        self.logger.debug('Detector::readout')

        if frame.shape[0] != self.resolution:
            raise ValueError('Detector::readout - The frame shape does not match the resolution of the detector')      

        if self.darkCurrent!=0:
            frame = self.set_dark_shot_noise(frame)
        
        # Simulate the saturation of the detector (without blooming and smearing)
        if self.FWC is not None:
            frame = self.set_saturation(frame)    
        
        # If the sensor is EMCCD the applyed gain is before the analog-to-digital conversion
        if self.sensor == 'EMCCD': 
            frame *= self.gain

        # Simulate hardware binning of the detector
        if self.binning != 1:
            frame = set_binning(frame,self.binning)
                    
        # Apply readout noise
        if self.readoutNoise!=0:    
            frame = self.set_readout_noise(frame)    

        # Apply the CCD/CMOS gain
        if self.sensor == 'CCD' or self.sensor == 'CMOS':
            frame *= self.gain
            
        # Apply the digital quantification of the detector
        if self.bits is not None:
            frame = self.digitalization(frame)
    
        # Remove the dark fromthe detector
        if self.backgroundMap is not None:
            frame = self.remove_bakground(frame)
        # Save the integrated frame and buffer
        self.frame  = frame.copy()

        # add the integrated time
        self._integrated_time += self.samplingTime

        return self.frame
             
    
    def integrate(self,frame):
        '''
        Integrate multiple exposures and apply detector effects.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        Returns
        -------
        np.ndarray
            Integrated frame.
        '''        
        # Apply photon noise 
        if self.photonNoise!=0:
            frame = self.set_photon_noise(frame)

        # Apply background noise
        if self.backgroundNoise is True:    
            frame = self.set_background_noise(frame)
            
        # Simulate the quantum efficiency of the detector (photons to electrons)
        frame = self.conv_photon_electron(frame)
        
        if self.integrationTime == self.samplingTime:
            return self.readout(frame) # Noisy frame
        elif self.integrationTime > self.samplingTime:
            self.logger.warning('Detector::integrate - Integration during a window larger than the sampling is not currently supported, \
                                 returning the instant noisy frame.')
            return self.readout(frame) # Noisy frame
        else:
            raise ValueError('Detector::integrate - The integration time is smaller than the sampling time.')

    def computeSNR(self, perfect_frame):
        '''
        Compute the Signal-to-Noise Ratio (SNR) for the detector.

        Parameters
        ----------
        perfect_frame : np.ndarray
            Ideal noise-free frame.

        Returns
        -------
        float
            Computed SNR value.
        '''
        flux_max_px = perfect_frame.max() 
        signal = self.QE * flux_max_px
        
        photon_noise = np.sqrt(signal)

        SNR = signal / np.sqrt(self.quantification_noise**2 + photon_noise**2 + self.readoutNoise**2 + self.dark_shot_noise**2) 

        self.logger.info('Theoretical maximum SNR: %.2f'%self.SNR_max)
        self.logger.info('Current SNR: %.2f'%SNR)

        return SNR
    
    
    def displayNoiseError(self):
        '''
        Display a summary of noise effects in the detector.

        Returns
        -------
        None
        '''        
        self.logger.debug('Detector::displayNoiseError')
        self.logger.info('------------ Noise error ------------')
        if self.bits is not None:
            self.logger.info('{:^25s}|{:^9.4f}'.format('Quantization noise [e-]',self.quantification_noise))
        if self.darkCurrent!=0:    
            self.logger.info('{:^25s}|{:^9.4f}'.format('Dark shot noise [e-]',self.dark_shot_noise))
        if self.readoutNoise!=0:
            self.logger.info('{:^25s}|{:^9.1f}'.format('Readout noise [e-]',self.readoutNoise))
        pass
                     
    def print_properties(self):
        self.logger.debug('Detector::print_properties')
        self.logger.info('------------ Detector ------------')
        self.logger.info('{:^25s}|{:^9s}'.format('Sensor type',self.sensor))
        if self.resolution is not None:
            self.logger.info('{:^25s}|{:^9d}'.format('Resolution [px]',self.resolution//self.binning))
        if self.integrationTime is not None:
            self.logger.info('{:^25s}|{:^9.4f}'.format('Exposure time [s]',self.integrationTime))
        if self.bits is not None:
            self.logger.info('{:^25s}|{:^9d}'.format('Quantization [bits]',self.bits))
        if self.FWC is not None:
            self.logger.info('{:^25s}|{:^9d}'.format('Full well capacity [e-]',self.FWC))
        self.logger.info('{:^25s}|{:^9d}'.format('Gain',self.gain))
        self.logger.info('{:^25s}|{:^9d}'.format('Quantum efficiency [%]',int(self.QE*100)))
        self.logger.info('{:^25s}|{:^9s}'.format('Binning',str(self.binning)+'x'+str(self.binning)))
        self.logger.info('{:^25s}|{:^9d}'.format('Dark current [e-/pixel/s]',self.darkCurrent))
        self.logger.info('{:^25s}|{:^9s}'.format('Photon noise',str(self.photonNoise)))
        self.logger.info('{:^25s}|{:^9s}'.format('Bkg noise [e-]',str(self.backgroundNoise)))
        self.logger.info('{:^25s}|{:^9.1f}'.format('Readout noise [e-/pixel]',self.readoutNoise))
        self.logger.info('----------------------------------')
    
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