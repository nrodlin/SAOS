import numpy as np

import h5py
import cv2

import logging
from .LoggingHelper import get_logger

class NCPA:
    def __init__(self,
                 telescope,
                 logger = None):
        """
        Initialize a NCPA generator.

        Parameters
        ----------
        telescope : Telescope
            Telescope object to which the WFS is attached.
        """        
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        self.logger = logger if logger is not None else get_logger()

        # Define class attributes
        self.tag = 'ncpa'
        
        # Generate empty OPD

        self.ncpa_opd = np.zeros_like(telescope.pupil, dtype=np.float32)


    def load(self, filename): 
        """
        Load the NCPA OPD from a H5 file.

        Parameters
        ----------
        filename : str
            Path and base filename (with extension) of the H5 file to load.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        self.logger.debug('NCPA::load')


        if not filename.endswith(".h5"):
            filename += ".h5"        

        with h5py.File(filename, 'r') as f:
            temp_opd = f['opd']['data'][()]

            self.logger.info(f"NCPA::load - Loaded OPD.")     

        if (temp_opd.shape[0] != self.ncpa_opd.shape[0]) or (temp_opd.shape[1] != self.ncpa_opd.shape[1]):
            self.logger.warning('NCPA::Load - The dimensions of the NCPA do not match the telescope\'s pupil. Interpolating, may introduce artifacs.')
            # cv2.resize takes the output size as (width, height) = (columns, rows)
            self.ncpa_opd = cv2.resize(temp_opd, (self.ncpa_opd.shape[1], self.ncpa_opd.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            self.ncpa_opd = np.array(temp_opd, dtype=float)

        return True
    
    def setOPD(self, input_opd):
        """
        set the NCPA OPD directly from a variable

        Parameters
        ----------
        input_opd : np.ndarray
            Input OPD in [m]

        Returns
        -------
        bool
            True if loaded successfully,.
        """
        self.logger.debug('NCPA::load')

        if (input_opd.shape[0] != self.ncpa_opd.shape[0]) or (input_opd.shape[1] != self.ncpa_opd.shape[1]):
            self.logger.error('NCPA::setOPD - The dimensions of the NCPA do not match the telescope\'s pupil. Rejecting.')
            raise ValueError('The dimensions of the NCPA do not match the telescope\'s pupil.')

        self.ncpa_opd = input_opd.copy()
        
        return True
    
    def getPhase(self):
        """
        Return the current NCPA OPD.

        Returns
        -------
        np.ndarray
            NCPA OPD in meters [m].
        """
        self.logger.debug('NCPA::getPhase')
       
        return self.ncpa_opd
    

    
