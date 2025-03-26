# -*- coding: utf-8 -*-
"""
Created on March 26 2025
@author: nrodlin
"""

import logging
import logging.handlers
from queue import Queue

import numpy as np
import torch
import cv2

from joblib import delayed, Parallel

from .Detector import Detector

"""
Science Camera Module
=================

This module contains the `ScienceCam` class, used for modeling a science camera in adaptive optics simulations.
"""

class ScienceCam:
    def __init__(self,
                 fieldOfView:float,
                 plate_scale:float,
                 samplingTime:float,
                 telescope,
                 fft_zero_padding:int=2,
                 integrationTime:float=None,
                 logger=None):
        
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger    

        self.fieldOfView      = fieldOfView
        self.plate_scale      = plate_scale
        self.fft_zero_padding = fft_zero_padding
        self.samplingTime     = samplingTime

        if integrationTime is None:
            self.integrationTime = self.samplingTime
        else:
            self.integrationTime = integrationTime
            if self.integrationTime > self.samplingTime:
                self.logger.warning('ScienceCam::init - Currently, integration a period longer than the sampling time is not supported due to parallelization conflicts.')
        
        self.nPix = int(np.round(self.fieldOfView / self.plate_scale))

        self.pupil = telescope.pupil.copy().astype(float)

        # TODO: Add noise during the initialization
        
        self.cam  = Detector(self.nPix, self.integrationTime, 
                             self.samplingTime, logger=self.logger)

    def get_frame(self, src, phase):
        if src.tag == 'source':
            coherence = self.compute_psf(phase, self.nPix)
            
            frame = self.cam.integrate(coherence.detach().numpy()) # The coherence is the PSF because the object is a point-source
        
        elif src.tag == 'sun':
            list_phase = [phase[i, :, :] for i in range(phase.shape[0])]
            
            npix = src.subDirs_sun.shape[0]

            # Compute in parallel the PSF for each subdir
            coherence = Parallel(n_jobs=1, prefer='threads')(delayed(self.compute_psf)(list_phase[i], npix) for i in range(len(list_phase)))

            # Convolute in prallel the PSF of each subDir with the sun patch of that subdir

            sun_patches = Parallel(n_jobs=1, prefer='threads')(delayed(self.compute_image)(src.subDirs_sun[:,:,i//src.nSubDirs, i%src.nSubDirs], coherence[i], npix) 
                                                           for i in range(len(list_phase)))
            
            # Combine the sun patches into a unique PSF

            sun_PSF_combined = torch.zeros(np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                           np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int))
            
            sun_psf_tmp_3D = torch.zeros((np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                          np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), src.nSubDirs*src.nSubDirs))

            for i in range(len(sun_patches)):
                dirX = i//src.nSubDirs
                dirY = i%src.nSubDirs
                
                gcorner_x = np.round(dirX*((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale))).astype(int)
                gcorner_y = np.round(dirY*((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale))).astype(int)

                gcorner_x_end = gcorner_x + np.round((src.subDirs_coordinates[2,0,0])/(self.plate_scale)).astype(int)
                gcorner_y_end = gcorner_y + np.round((src.subDirs_coordinates[2,0,0])/(self.plate_scale)).astype(int)

                start = npix//2 - src.filter_2D.shape[0]//2
                
                sun_psf_tmp_3D[gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end, i] = sun_patches[i][start:start+src.filter_2D.shape[0], 
                                                                                                     start:start+src.filter_2D.shape[0]] * src.filter_2D[:,:,i//src.nSubDirs, i%src.nSubDirs]
            
            # Crop the external region, which might be affected by the windowing effect.
            # The patch is larger so that the final crop matches the FoV of the object.
            sun_PSF_combined = torch.sum(sun_psf_tmp_3D,axis=2)
            
            offset = np.round(src.patch_padding/(2*self.plate_scale)).astype(int)

            frame = sun_PSF_combined[offset:np.round(offset+(src.fov/src.img_PS)).astype(int), 
                                                    offset:np.round(offset+(src.fov/self.plate_scale)).astype(int)].detach().numpy()
            
            # Add detector noise

            frame = self.cam.integrate(frame)

        else:
            raise ValueError(f'ScienceCam::get_frame - The source tag ({src.tag}) is not supported.')
        
        return frame
    
    def compute_psf(self, phase, npix):

        phase_torch = torch.from_numpy(cv2.resize(phase, (npix, npix), interpolation=cv2.INTER_LINEAR))

        pupil_torch = torch.from_numpy(cv2.resize(self.pupil, (npix, npix), interpolation=cv2.INTER_LINEAR))

        psf_zeropadded = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil_torch * torch.exp(1j * phase_torch), 
                                                 s=(npix*self.fft_zero_padding, npix*self.fft_zero_padding), norm='forward')))
        
        start = psf_zeropadded.shape[0]//2 - npix//2
        psf_nopad = psf_zeropadded[start:start+npix, start:start+npix]
        return psf_nopad / torch.max(psf_nopad)
    
    def compute_image(self, object, coherence, npix):
        
        object_torch = torch.from_numpy(object.copy())

        # Compute FFT2

        object_fft    = torch.fft.fft2(object_torch, s=(npix*self.fft_zero_padding, npix*self.fft_zero_padding), norm='backward')
        coherence_fft = torch.fft.fft2(coherence, s=(npix*self.fft_zero_padding, npix*self.fft_zero_padding), norm='backward')

        # Convolute

        image_zeropadded = torch.abs(torch.fft.ifft2(object_fft*coherence_fft, s=(npix*self.fft_zero_padding, npix*self.fft_zero_padding), norm='backward'))

        start = image_zeropadded.shape[0]//2 - npix//2

        image_nopad = image_zeropadded[start:start+npix, start:start+npix]


        return image_nopad / torch.max(image_nopad)
        
        

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