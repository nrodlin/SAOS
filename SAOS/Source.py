# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:32:15 2020

@author: cheritie

Update on March 24 2025
@author: nrodlin
"""

import logging
import logging.handlers
from queue import Queue

import numpy as np


"""
Source Module
=================

This module contains the `Source` class, used for modeling a natural star in adaptive optics simulations.
"""

class Source:    
    def __init__(self,
                 optBand:str,
                 magnitude:float,
                 coordinates:list = [0,0],
                 altitude:float = np.inf, 
                 laser_coordinates:list = [0,0],
                 Na_profile:float = None,
                 FWHM_spot_up:float = None,
                 chromatic_shift:list = None,
                 logger = None):
        """
        Initialize a Source object.

        Parameters
        ----------
        optBand : str
            Optical band identifier (e.g., 'V', 'H').
        magnitude : float
            Apparent magnitude of the star.
        coordinates : list, optional
            Sky coordinates [zenith, azimuth] in [arcsec, degrees], by default [0, 0].
        altitude : float, optional
            Altitude of the source in meters. Defaults to infinity (NGS).
        laser_coordinates : list, optional
            Launch coordinates for a laser source [x, y] in meters.
        Na_profile : float, optional
            Sodium layer profile [altitudes, values]. Required for LGS.
        FWHM_spot_up : float, optional
            FWHM of the LGS spot in arcsec.
        chromatic_shift : list, optional
            Shift per atmospheric layer due to chromatic dispersion, in arcsec.
        logger : logging.Logger, optional
            Logger instance for logging.
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger
        
        self.is_initialized = False
        tmp             = self.photometry(optBand)              # get the photometry properties
        self.optBand    = optBand                               # optical band
        self.wavelength = tmp[0]                                # wavelength in m
        self.bandwidth  = tmp[1]                                # optical bandwidth
        self.zeroPoint  = tmp[2]/368                            # zero point
        self.magnitude  = magnitude                             # magnitude
        self.phase      = []                                    # phase of the source 
        self.phase_no_pupil      = []                           # phase of the source (no pupil)
        self.fluxMap    = []                                    # 2D flux map of the source
        self.nPhoton    = self.zeroPoint*10**(-0.4*magnitude)   # number of photon per m2 per s
        self.tag        = 'source'                              # tag of the object
        self.altitude = altitude                                # altitude of the source object in m    
        self.coordinates = coordinates                          # polar coordinates [r,theta] 
        self.laser_coordinates = laser_coordinates              # Laser Launch Telescope coordinates in [m] 
        self.chromatic_shift = chromatic_shift                             # shift in arcsec to be applied to the atmospheric phase screens (one value for each layer) to simulate a chromatic effect
        if Na_profile is not None and FWHM_spot_up is not None:
            self.Na_profile = Na_profile
            self.FWHM_spot_up = FWHM_spot_up
            
            # consider the altitude weigthed by Na profile
            self.altitude = np.sum(Na_profile[0,:]*Na_profile[1,:])
            self.type     = 'LGS'
        else:
            
            self.type     = 'NGS'
            
        self.is_initialized = True   

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE PHOTOMETRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def photometry(self,arg):
        """
        Returns photometric properties of the selected band.

        Parameters
        ----------
        arg : str
            Name of the photometric band (e.g., 'V', 'H', 'Na').

        Returns
        -------
        list or int
            List of [wavelength, bandwidth, zero-point flux] or -1 if invalid.
        """
        self.logger.debug('Source::photometry')
        # photometry object [wavelength, bandwidth, zeroPoint]
        class phot:
            pass
        
        ## New entries with corrected zero point flux values 
        phot.U      = [ 0.360e-6 , 0.070e-6 , 1.96e12 ]
        phot.B      = [ 0.440e-6 , 0.100e-6 , 5.38e12 ]
        phot.V0     = [ 0.500e-6 , 0.090e-6 , 3.64e12 ]
        phot.V      = [ 0.550e-6 , 0.090e-6 , 3.31e12 ]
        phot.R      = [ 0.640e-6 , 0.150e-6 , 4.01e12 ]
        phot.R2     = [ 0.650e-6 , 0.300e-6 , 7.9e12  ] 
        phot.R3     = [ 0.600e-6 , 0.300e-6 , 8.56e12 ] 
        phot.R4     = [ 0.670e-6 , 0.300e-6 , 7.66e12 ] 
        phot.I      = [ 0.790e-6 , 0.150e-6 , 2.69e12 ]
        phot.I1     = [ 0.700e-6 , 0.033e-6 , 0.67e12 ] 
        phot.I2     = [ 0.750e-6 , 0.033e-6 , 0.62e12 ] 
        phot.I3     = [ 0.800e-6 , 0.033e-6 , 0.58e12 ] 
        phot.I4     = [ 0.700e-6 , 0.100e-6 , 2.02e12 ] 
        phot.I5     = [ 0.850e-6 , 0.100e-6 , 1.67e12 ] 
        phot.I6     = [ 1.000e-6 , 0.100e-6 , 1.42e12 ] 
        phot.I7     = [ 0.850e-6 , 0.300e-6 , 5.00e12 ] 
        phot.I8     = [ 0.750e-6 , 0.100e-6 , 1.89e12 ] 
        phot.I9     = [ 0.850e-6 , 0.300e-6 , 5.00e12 ] 
        phot.I10    = [ 0.900e-6 , 0.300e-6 , 4.72e12 ] 
        phot.J      = [ 1.215e-6 , 0.260e-6 , 1.90e12 ]
        phot.J2     = [ 1.550e-6 , 0.260e-6 , 1.49e12 ] 
        phot.H      = [ 1.654e-6 , 0.290e-6 , 1.05e12 ]
        phot.Kp     = [ 2.1245e-6 , 0.351e-6 , 0.62e12 ]
        phot.Ks     = [ 2.157e-6 , 0.320e-6 , 0.55e12 ]
        phot.K      = [ 2.179e-6 , 0.410e-6 , 0.70e12 ]
        phot.K0     = [ 2.000e-6 , 0.410e-6 , 0.76e12 ]
        phot.K1     = [ 2.400e-6 , 0.410e-6 , 0.64e12 ]

        phot.L      = [ 3.547e-6 , 0.570e-6 , 2.5e11 ]
        phot.M      = [ 4.769e-6 , 0.450e-6 , 8.4e10 ]
        phot.Na     = [ 0.589e-6 , 0        , 3.3e12 ] 
        phot.EOS    = [ 1.064e-6 , 0        , 3.3e12 ] 
        phot.IR1310 = [ 1.310e-6 , 0        , 2e12 ]  
        
        if isinstance(arg,str):
            if hasattr(phot,arg):
                return getattr(phot,arg)
            else:
                self.logger.error('Source::photometry - Wrong name for the photometry object.')
                raise ValueError('Wrong name for the photometry object.')
        else:
            self.logger.error('Source::photometry - The photometry object takes a scalar as an input.')
            raise ValueError('The photometry object takes a scalar as an input.')   
            
    def print_properties(self):
        """
        Print the main properties of the source.

        Returns
        -------
        None
        """
        self.logger.info('Source::print_properties')
        self.logger.info('{: ^8s}'.format('Source') +'{: ^10s}'.format('Wavelength')+ '{: ^8s}'.format('Zenith')+ '{: ^10s}'.format('Azimuth')+ '{: ^10s}'.format('Altitude')+ '{: ^10s}'.format('Magnitude') + '{: ^10s}'.format('Flux'))
        self.logger.info('{: ^8s}'.format('') +'{: ^10s}'.format('[m]')+ '{: ^8s}'.format('[arcsec]')+ '{: ^10s}'.format('[deg]')+ '{: ^10s}'.format('[m]')+ '{: ^10s}'.format('') + '{: ^10s}'.format('[phot/m2/s]') )

        self.logger.info('-------------------------------------------------------------------')        
        self.logger.info('{: ^8s}'.format(self.type) +'{: ^10s}'.format(str(self.wavelength))+ '{: ^8s}'.format(str(self.coordinates[0]))+ '{: ^10s}'.format(str(self.coordinates[1]))+'{: ^10s}'.format(str(np.round(self.altitude,2)))+ '{: ^10s}'.format(str(self.magnitude))+'{: ^10s}'.format(str(np.round(self.nPhoton,1))) )

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
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()    