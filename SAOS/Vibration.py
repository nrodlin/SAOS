import numpy as np

import h5py

import logging
import logging.handlers
from queue import Queue

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
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Define class attributes
        self.tag = 'vibrations'
        
        # Load vibrations sequence

        self.x_vibrations_arcsec, self.y_vibrations_arcsec, self.Ts = self.load(source_file)

        # Sampling time sanity check
        if self.Ts != telescope.samplingTime:
            self.logger.error(f'Vibrations sampling time ({self.Ts}s) does not match the Telescope\'s sampling time ({telescope.samplingTime}s)')
            raise ValueError('Vibrations sampling time does not match the telescope\'s.')
        
        # Generate base phase

        x = np.linspace(-1, 1, telescope.pupil.shape[0])
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
        if ((iteration+1) > len(self.x_vibrations_stroke)) or ((iteration+1) > len(self.x_vibrations_stroke)):
            self.logger.warning('Vibration::getCurrentVibrations - The length of the vibrations array is smaller than the simulation window. Wrapping-around.')
            iteration = iteration % np.minimum(len(self.x_vibrations_stroke), len(self.y_vibrations_stroke))
        
        opd_vibrations = self.x_vibrations_stroke[iteration]*self.x_mode + self.y_vibrations_stroke[iteration]*self.y_mode
        
        return opd_vibrations

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
