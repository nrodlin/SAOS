import numpy as np
import torch

import time

import logging
import logging.handlers
from queue import Queue

class predictiveLearnApply:
    def __init__(self,
                 window:int=1000, 
                 updateCycles:int=2000,            
                logger = None,
                **kwargs):
    
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Define class attributes
        self.tag = 'tomopLA'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

        # Initialize attributes
        self.slopes_buffer = None
        self.window = window
        self.updateCycles = updateCycles # number of cycles after the first matrix before updating the CovMat
        self.iteration = 0
        self.slopes_covmat = None

    def feed(self, combined_slopes):
        """
        Fill circular buffer with open-loop or pseudo-open-loop slopes and
        periodically estimate the slopes covariance matrix.
        """

        combined_slopes = np.asarray(combined_slopes).ravel()

        if self.slopes_buffer is None:
            self.slopes_buffer = np.zeros((self.window, combined_slopes.size))

        idx = self.iteration % self.window
        self.slopes_buffer[idx, :] = combined_slopes

        self.iteration += 1

        buffer_is_full = self.iteration >= self.window

        # Update if: first time the buffer is full or there is a CovMat and the number of samples to update has passed
        should_update = (self.iteration == self.window or (self.slopes_covmat is not None and (self.iteration - self.window) % self.updateCycles == 0))

        if buffer_is_full and should_update:
            X = self.slopes_buffer

            self.slopes_covmat = (X.T @ X) / (self.window - 1)

        return True
    
    def reconstruct(self, combined_slopes):
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
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()