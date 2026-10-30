import numpy as np

import h5py
import cv2

import logging
from .LoggingHelper import get_logger

class LocalSeeing:
    def __init__(self,
                 telescope,
                 source_file:str,
                 logger = None):
        """
        Initialize a local seeing player.

        Parameters
        ----------
        telescope : Telescope
            Telescope object to which the WFS is attached.
        source_file : str
            Local seeing filename, the data is expected in [m] sampled with the telescope's pupil resolution
        """        
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        self.logger = logger if logger is not None else get_logger()

        # Define class attributes
        self.tag = 'localSeeing'
        
        # Load local seeing sequence

        temp_buffer, self.Ts = self.load(source_file)

        if temp_buffer.shape[0] != telescope.pupil.shape[0]:
            self.logger.warning('LocalSeeing - The array resolution is different from the telescope\'s. Interpolating, may introduce artifacs.')
            self.localSeeing_input = np.zeros((telescope.pupil.shape[0], telescope.pupil.shape[1], temp_buffer.shape[2]))
            for i in range(self.localSeeing_input.shape[2]):
                self.localSeeing_input[:,:,i] = cv2.resize(temp_buffer[:,:,i], (telescope.pupil.shape[0], telescope.pupil.shape[1]), 
                                                            interpolation=cv2.INTER_LINEAR) * telescope.pupil
        else:
            self.localSeeing_input = temp_buffer.copy() 
        # Sampling time sanity check
        if self.Ts != telescope.samplingTime:
            self.logger.error(f'Local Seeing sampling time ({self.Ts}s) does not match the Telescope\'s sampling time ({telescope.samplingTime}s)')
            raise ValueError('Local seeing sampling time does not match the telescope\'s.')
    

    def load(self, filename): 
        """
        Load a local seeing temporal sequence from a H5 file.

        Parameters
        ----------
        filename : str
            Path and base filename (with extension) of the H5 file to load.

        Returns
        -------
        np.ndarray
            OPD in [m]
        float
            Sampling time of the local seeing sequence
        """
        self.logger.debug('LocalSeeing::load')


        if not filename.endswith(".h5"):
            filename += ".h5"        

        with h5py.File(filename, 'r') as f:

            opd = f['opd']['data'][()]
            Ts = f['opd'].attrs['Ts']

            self.logger.info(f"LocalSeeing::load - Loaded OPD.")     

        return opd, Ts
    
    def getCurrentOPD(self, iteration):
        """
        Return the local seeing for the iteration asked.

        Parameters
        ----------
        iteration : int
            Iteration of the AO simulation. If iteration is larger than
            the temporal sequence of the local seeing, the method wraps-around.

        Returns
        -------
        np.ndarray
            Local seeing OPD for the given iteration
        """        
        if (iteration+1) > self.localSeeing_input.shape[2]:
            self.logger.warning('LocalSeeing::getCurrentOPD - The length of the local seeing array is smaller than the simulation window. Wrapping-around.')
            iteration = iteration % self.localSeeing_input.shape[2]
        
        return self.localSeeing_input[:,:,iteration]

    
