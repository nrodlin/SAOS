import numpy as np

import h5py

import logging
from .LoggingHelper import get_logger

class Vibration:
    def __init__(self,
                 telescope,
                 source_file:str,
                 logger = None):
        """
        Initialize a vibration generator.

        Parameters
        ----------
        telescope : Telescope
            Telescope object to which the WFS is attached.
        source_file : str
            Vibration filename, the data is expected in arcsec on sky.
        """        
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        self.logger = logger if logger is not None else get_logger()

        # Define class attributes
        self.tag = 'vibrations'
        
        # Load vibrations sequence

        self.x_vibrations_arcsec, self.y_vibrations_arcsec, self.Ts = self.load(source_file)

        # Sampling time sanity check
        if self.Ts != telescope.samplingTime:
            self.logger.error(f'Vibrations sampling time ({self.Ts}s) does not match the Telescope\'s sampling time ({telescope.samplingTime}s)')
            raise ValueError('Vibrations sampling time does not match the telescope\'s.')
        
        # Generate base phase

        # Pixel centres of the pupil grid (i - (N-1)/2) D/N, normalised by D/2: the mode is x / (D/2) exactly (a linspace(-1, 1, N)
        # would put the edge pixels at +-D/2 and scale every tilt by N/(N-1))
        n = telescope.pupil.shape[0]
        x = (np.arange(n) - (n - 1) / 2) * 2 / n
        self.x_mode = np.tile(x, (x.shape[0],1)) * telescope.pupil
        self.y_mode = self.x_mode.T
        
        # Compute modal amplitude [m]: the modes span [-1, 1] over the pupil (D/2 in metres), so a tilt of theta [rad]
        # on sky is an OPD of theta * D/2 at the edge

        self.x_vibrations_stroke = telescope.D / 2 * (self.x_vibrations_arcsec / 206265)
        self.y_vibrations_stroke = telescope.D / 2 * (self.y_vibrations_arcsec / 206265)


    def load(self, filename): 
        """
        Load a vibration temporal sequence from a H5 file.

        Parameters
        ----------
        filename : str
            Path and base filename (with extension) of the H5 file to load.

        Returns
        -------
        np.ndarray
            X-axis vibrations
        np.ndarray
            Y-axis vibrations
        float
            Sampling time of the vibration sequence
            
        """
        self.logger.debug('Vibration::load')


        if not filename.endswith(".h5"):
            filename += ".h5"        

        with h5py.File(filename, 'r') as f:

            x_vibrations = f['x_axis']['data'][()]
            y_vibrations = f['y_axis']['data'][()]
            Ts = f['x_axis'].attrs['Ts']

            self.logger.info(f"Vibration::load - Loaded vibrations.")     

        return x_vibrations, y_vibrations, Ts
    
    def getCurrentVibrations(self, iteration):
        """
        Return the vibrations for the iteration asked.

        Parameters
        ----------
        iteration : int
            Iteration of the AO simulation. If iteration is larger than
            the temporal sequence of the vibrations, the method wraps-around.

        Returns
        -------
        np.ndarray
           Vibration combining X-Y vibrations for the given iteration
        """        
        if ((iteration+1) > len(self.x_vibrations_stroke)) or ((iteration+1) > len(self.y_vibrations_stroke)):
            self.logger.warning('Vibration::getCurrentVibrations - The length of the vibrations array is smaller than the simulation window. Wrapping-around.')
            iteration = iteration % np.minimum(len(self.x_vibrations_stroke), len(self.y_vibrations_stroke))
        
        opd_vibrations = self.x_vibrations_stroke[iteration]*self.x_mode + self.y_vibrations_stroke[iteration]*self.y_mode
        
        return opd_vibrations

    
