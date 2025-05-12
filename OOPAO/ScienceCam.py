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
                 decimation:int=50,
                 logger=None):
        """
        Initialize a ScienceCam object to simulate a science camera in adaptive optics simulations.

        Parameters
        ----------
        fieldOfView : float
            Field of view in arcseconds.
        plate_scale : float
            Plate scale in arcseconds per pixel.
        samplingTime : float
            Time interval between frames [s].
        telescope : Telescope
            Associated telescope object.
        fft_zero_padding : int, optional
            Zero-padding factor for FFT operations. Default is 2.
        integrationTime : float, optional
            Integration time in seconds. Defaults to samplingTime.
        decimation : int, optional
            Decimation factor for storing results. Default is 50.
        logger : logging.Logger, optional
            Logger instance for diagnostics.
        """        
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
        self.decimation       = decimation

        if integrationTime is None:
            self.integrationTime = self.samplingTime
        else:
            self.integrationTime = integrationTime
            if self.integrationTime > self.samplingTime:
                self.logger.warning('ScienceCam::init - Currently, integration a period longer than the sampling time is not supported due to parallelization conflicts.')
        
        self.nPix = int(np.round(self.fieldOfView / self.plate_scale))
        self.telescope_diameter = telescope.D
        self.pupil = telescope.pupil.copy().astype(float)

        # TODO: Add noise during the initialization
        
        self.cam  = Detector(self.nPix, self.integrationTime, 
                             self.samplingTime, logger=self.logger)

    def get_frame(self, src, phase):
        """
        Generate a science camera frame based on the input phase and source object.

        Parameters
        ----------
        src : Source
            Input source object, either a point source or extended source (e.g., Sun).
        phase : np.ndarray
            Optical phase at the science detector.

        Returns
        -------
        np.ndarray
            Final science frame with detector effects.
        """
        
        fwhm = (src.wavelength / self.telescope_diameter) * (206265 / self.plate_scale)

        if src.tag == 'source':
            psf = self.compute_psf(phase, fwhm)
            
            frame = self.cam.integrate(psf.detach().numpy()) # The coherence is the PSF because the object is a point-source
        
        elif src.tag == 'sun':
            if src.fov < self.fieldOfView:
                raise ValueError('ScienceCam::get_frame - The source FoV is smaller than the camera FoV.')
            
            # If the phase is not 3D, we need to repeat it for each subDir --> typically when the phase uses the DM only
            if np.ndim(phase) < 3:
                phase = np.repeat(phase[np.newaxis,:,:], src.nSubDirs**2, axis=0)
            
            list_phase = [phase[i, :, :] for i in range(phase.shape[0])]
            
            # Interpolate sun subDirs to the adequate size given the camera PS
            subDirs_torch = torch.from_numpy(src.subDirs_sun).contiguous()
            subDirs_torch = subDirs_torch.view(subDirs_torch.shape[0], subDirs_torch.shape[1], -1).permute(2, 0, 1)
            new_size = np.round((src.subDirs_coordinates[2,0,0]+src.subDir_margin)/self.plate_scale).astype(int)
            subDirs_torch = torch.nn.functional.interpolate(subDirs_torch.unsqueeze(0), size=(new_size, new_size), 
                                                            mode='bilinear', align_corners=True).squeeze(0).contiguous()  
                      
            # Compute in parallel the PSF for each subdir --> shape matches that of the source, using the camera plate scale
            psf = Parallel(n_jobs=1, prefer='threads')(delayed(self.compute_psf)(list_phase[i], fwhm, subDirs_torch.shape[2]) for i in range(len(list_phase)))
            
            # Convolute in parallel the PSF of each subDir with the sun patch of that subdir
            # This is faster than compute the FFT2 over 3D tensors and avoid looping, at least on CPU
            sun_patches = Parallel(n_jobs=1, prefer='threads')(delayed(self.compute_image)(subDirs_torch[i,:,:], psf[i]) 
                                                           for i in range(len(list_phase))) 
            
            # Resize the 2D filter
            filter_2D_torch = torch.from_numpy(src.filter_2D).contiguous()
            filter_2D_torch = filter_2D_torch.view(filter_2D_torch.shape[0], filter_2D_torch.shape[1], -1).permute(2, 0, 1)
            new_size = np.round(src.subDirs_coordinates[2,0,0]/self.plate_scale).astype(int)
            filter_2D_torch = torch.nn.functional.interpolate(filter_2D_torch.unsqueeze(0), size=(new_size, new_size), 
                                                            mode='bilinear', align_corners=True).squeeze(0).contiguous()
            # Combine the sun patches into a unique PSF

            sun_PSF_combined = torch.zeros(np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                           np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), dtype=torch.float32).contiguous()
            
            sun_psf_tmp_3D = torch.zeros((np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                          np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), src.nSubDirs*src.nSubDirs), dtype=torch.float32).contiguous()

            # The small gain corrector is used to normalize the filter after it was interpolated so that differences below 1-2% can be compensated
            small_gain_corrector = torch.zeros((np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                src.nSubDirs*src.nSubDirs), dtype=torch.float32).contiguous()
            
            for i in range(len(sun_patches)):
                dirX = i//src.nSubDirs
                dirY = i%src.nSubDirs
                
                gcorner_x = np.round(dirX*((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale))).astype(int)
                gcorner_y = np.round(dirY*((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale))).astype(int)

                gcorner_x_end = gcorner_x + np.round((src.subDirs_coordinates[2,0,0])/(self.plate_scale)).astype(int)
                gcorner_y_end = gcorner_y + np.round((src.subDirs_coordinates[2,0,0])/(self.plate_scale)).astype(int)

                start = sun_patches[i].shape[0]//2 - filter_2D_torch.shape[1]//2
                
                sun_psf_tmp_3D[gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end, i] = filter_2D_torch[i,:,:] * sun_patches[i][start:start+filter_2D_torch.shape[1], 
                                                                                                                              start:start+filter_2D_torch.shape[1]]
                
                small_gain_corrector[gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end, i] = filter_2D_torch[i,:,:]
            
            # Crop the region of interest
            sun_PSF_combined = torch.sum(sun_psf_tmp_3D,axis=2) / torch.sum(small_gain_corrector,axis=2)
            
            offset = sun_patches[0].shape[0]//2 - self.nPix//2

            frame = sun_PSF_combined[offset:offset+self.nPix, offset:offset+self.nPix].detach().numpy()
            
            # Add detector noise

            frame = self.cam.integrate(frame)

        else:
            raise ValueError(f'ScienceCam::get_frame - The source tag ({src.tag}) is not supported.')
        
        return frame
    
    def compute_psf(self, phase, fwhm, nPix=None):
        """
        Compute the PSF from a given phase map using FFT.

        Parameters
        ----------
        phase : np.ndarray
            Phase input in radians.
        fwhm : float
            Full width at half maximum in arcsec.
        nPix : int, optional
            Image size in pixels. If None, uses camera's resolution.

        Returns
        -------
        torch.Tensor
            Normalized PSF.
        """
        if nPix is None:
            nPix = self.nPix
        # Compute FFT image size
        if fwhm < 1:
            raise ValueError('FWHM must be greater than 2 to guarantee enough spatial sampling.')
        nFFT = np.round(fwhm * nPix).astype(int)

        # Rescaled phase and pupil
        phase_torch = torch.from_numpy(phase).contiguous()
        pupil_torch = torch.from_numpy(self.pupil).contiguous()

        phase_rescaled = torch.nn.functional.interpolate(phase_torch.unsqueeze(0).unsqueeze(0), size=(nPix, nPix), 
                                                         mode='bilinear', align_corners=True).squeeze().contiguous()
        pupil_rescaled = torch.nn.functional.interpolate(pupil_torch.unsqueeze(0).unsqueeze(0), size=(nPix, nPix), 
                                                         mode='bilinear', align_corners=True).squeeze().contiguous()
        # Fill Phase image, zeropadded to get to the dimensions of nFFT
        # Define pupil
        start = (nFFT - nPix) // 2
        end = start + nPix
        square_pupil = np.zeros((nFFT, nFFT), dtype=float)
        square_pupil[start:end, start:end] = pupil_rescaled

        # Phase torch
        exp_phase = torch.zeros((nFFT, nFFT), dtype=torch.complex64).contiguous()

        real = torch.cos(phase_rescaled)
        imag = torch.sin(phase_rescaled)
        
        exp_phase[start:end, start:end] = pupil_rescaled * (real + 1j * imag)

        # Compute PSF
        psf = torch.fft.fft2(exp_phase, dim=(0, 1), norm='forward').contiguous()  # same as dividing by nFFT²

        # Shift zero frequency to center
        psf = torch.fft.fftshift(psf, dim=(0, 1))

        # Compute normalized intensity
        psf = torch.abs(psf) ** 2

        return psf[start:end, start:end].contiguous()  # Crop the PSF to the original size
    
    def compute_image(self, sci_object, coherence):
        """
        Compute the convolved image from an object and a PSF.

        Parameters
        ----------
        sci_object : np.ndarray
            Intensity map of the object.
        coherence : torch.Tensor
            PSF to convolve with the object.

        Returns
        -------
        torch.Tensor
            Final image after convolution.
        """        
        object_torch = sci_object.contiguous()

        # Compute FFT2

        object_fft    = torch.fft.fft2(object_torch, norm='forward', dim=(0,1))
        coherence_fft = torch.fft.fft2(coherence, norm='forward', dim=(0,1))

        # Convolute

        image = torch.abs(torch.fft.fftshift(torch.fft.ifft2(object_fft*coherence_fft, norm='forward', dim=(0,1)), dim=(0,1)))

        return image

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