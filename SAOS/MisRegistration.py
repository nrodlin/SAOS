# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:01:10 2020

@author: cheritie

Major update Oct 23 2025
@author: nrodlin
"""
import numpy as np

import logging
import logging.handlers
from queue import Queue

import scipy as sp
import cv2

"""
Misregistration Module
=================

This module contains the `Misresgistration` class, used for modeling the DM misregistration in adaptive optics simulations.
"""

class MisRegistration:
    def __init__(self,
                 shiftX:float,
                 shiftY:float,
                 rotation:float,
                 radialScaling:float,
                 telescope,
                 logger=None,
                 **kwargs):
        """
        Initialize the MisRegistration object, which defines misalignments of an optical element.

        Parameters
        ----------
        shiftX : float
            Displacement in X axis (left-right) w.r.t optical center [meters]
        shiftY : float
            Displacement in Y axis (top-bottom) w.r.t optical center [meters]
        rotation : float
            Rotation along optical axis [degres]
        radialScaling : float
            Pupil radial scaling. Larger than 1 is magnification. [adimensional]
        logger : logging.Logger, optional
            Logger instance for diagnostics.
        **kwargs : dict, optional
            Additional keyword arguments.
            vx : float, optional
                X axis displacement speed [m/s].
            vy : float, optional
                Y axis displacement speed [m/s].
            wz : float, optional
                Rotation speed along optical [degree/s]. 
        """        
        self.tag                  = 'misRegistration' 
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger        

        self.shiftX = shiftX
        self.shiftY = shiftY
        self.rotation = rotation
        self.radialScaling = radialScaling
        self.spatialSampling = telescope.pixelSize

        self.vx = kwargs.get('vx', 0.0)
        self.vy = kwargs.get('vy', 0.0)
        self.wz = kwargs.get('wz', 0.0)
        

    def update_params(self, elapsedTime):
        
        self.shiftX   += self.vx * elapsedTime
        self.shiftY   += self.vy * elapsedTime
        self.rotation += self.wz * elapsedTime

    def set_params(self, shiftX, shiftY, rotation, radialScaling, vx=None, vy=None, wz=None):
        self.shiftX = shiftX
        self.shiftY = shiftY
        self.rotation = rotation
        self.radialScaling = radialScaling

        if vx is not None:
            self.vx = vx
        if vy is not None:
            self.vy = vy
        if wz is not None:
            self.wz = wz

    
    def apply_misreg(self, input_buffer):

        # The DM OPD is not null at the border of the array (the pupil fills it), so the borders are extended with the
        # nearest value: a constant (zero) border distorts the pupil edge for any non-null misregistration, even 1e-9 deg
        temp_buffer = np.copy(input_buffer)
        if self.radialScaling != 1:
            # Apply magnification
            scalingMatrix = cv2.getRotationMatrix2D((input_buffer.shape[1]/2,input_buffer.shape[0]/2), 0, self.radialScaling)
            temp_buffer = cv2.warpAffine(input_buffer, scalingMatrix, (input_buffer.shape[1], input_buffer.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)

        if self.rotation != 0:
            # Apply rotation: cubic spline (order 3), sub-nm round-trip error on the DM OPD
            temp_buffer = sp.ndimage.rotate(temp_buffer, self.rotation, reshape=False, order=3, mode='nearest')

        # Apply shift
        if self.shiftX != 0 or self.shiftY != 0:
            translationMatrix = np.float32([[1, 0, self.shiftX/self.spatialSampling], [0, 1, self.shiftY/self.spatialSampling]])
            temp_buffer = cv2.warpAffine(temp_buffer, translationMatrix, (temp_buffer.shape[1], temp_buffer.shape[0]),
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_REPLICATE)

        return temp_buffer
    
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
     
