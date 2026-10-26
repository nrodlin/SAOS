# -*- coding: utf-8 -*-
"""
Created on March 26 2025
@author: nrodlin
"""

import logging
import logging.handlers
from queue import Queue

import time

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
                 lightRatio:float=50,
                 integrationTime:float=None,
                 decimation:int=1,
                 noiseFlag:bool=False,
                 logger=None,
                 **kwargs):
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
        lightRatio : float
            Threshold ratio to select valid subapertures based on flux.            
        integrationTime : float, optional
            Integration time in seconds. Defaults to samplingTime.
        decimation : int, optional
            Decimation factor for storing results. Default is 1, no decimation.            

        noiseFlag : bool, optional
            If True, the detector includes noise using the kwargs params/default config. By default, False.                 
        logger : logging.Logger, optional
            Logger instance for diagnostics.
        **kwargs : dict, optional
            Additional keyword arguments.

            fullWellCapacity : int, optional
                Detector parameter. Full Well Capacity of pixels [e-]. Default, 60ke-
            nBits : int, optional
                Detector parameter. Bit depth for quantization. Default is 12, shall be >= 8
            quantumEfficiency : float, optional
                Detector parameter. Quantum efficiency (0-1). Default is 0.64.
            shotNoise : bool, optional
                Detector parameter. Shot noise flag. Default 1.
            darkCurrent : float, optional
                Detector parameter. Dark current [e-]. Default 250e-/px/s.
            readoutNoise : float, optional
                Detector parameter. Readout noise [e-]. Default 60e-.
            gain : float, optional
                Detector parameter. Gain of the detector. Default is 1.
            quantization_conversion : float, optional.
                Detector parameter. Conversion gain to discretize the measurement [e-/px]. Default 0, assumed linear.
            sensorType : str, optional
                Detector parameter. Sensor type ('CCD', 'CMOS', 'EMCCD'). Default is 'CCD'.
            darkCalibration : int, optional
                Detector parameter. Number of frames to calibrate the dark. Default 20.
            randomState : int, optional
                Detector parameter. Seed for the random number generator. Default is None.          
        """        
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger    

        self.fieldOfView         = fieldOfView
        self.plate_scale         = plate_scale
        self.samplingTime        = samplingTime
        self.decimation          = decimation
        self.lightRatio          = lightRatio

        self.device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        if integrationTime is None:
            self.integrationTime = self.samplingTime
        else:
            self.integrationTime = integrationTime
        
        self.nPix = int(np.round(self.fieldOfView / self.plate_scale))
        self.telescope_diameter = telescope.D
        self.pupil = telescope.pupil.copy().astype(float)

        # Detector camera 

        self.camera_size  = self.nPix

        self.camera_params = dict()
        # Default parameters correspond to pco.dimax 3.6 ST @500nm

        self.camera_params['fullWellCapacity'] = kwargs.get('fullWellCapacity', 60000)
        self.camera_params['nBits'] = kwargs.get('nBits', 10)
        self.camera_params['quantumEfficiency'] = kwargs.get('quantumEfficiency', 0.64)
        self.camera_params['shotNoise'] = kwargs.get('shotNoise', 1)
        self.camera_params['darkCurrent'] = kwargs.get('darkCurrent', 250)
        self.camera_params['readoutNoise'] = kwargs.get('readoutNoise', 60)
        self.camera_params['gain'] = kwargs.get('gain', 1)
        self.camera_params['quantization_conversion'] = kwargs.get('quantization_conversion', 0)
        self.camera_params['sensorType'] = kwargs.get('sensorType', 'CMOS')
        self.camera_params['darkCalibration'] = kwargs.get('darkCalibration', 20)
        self.camera_params['randomState'] = kwargs.get('randomState', None)
        self.camera_params['integrationTime'] = kwargs.get('integrationTime', telescope.samplingTime)

        camera_kwargs = {'randomState':self.camera_params['randomState'], 
                         'integrationTime':self.integrationTime}

        self.cam = Detector(nPix=self.camera_size,
                            samplingTime=telescope.samplingTime,
                            fullWellCapacity=self.camera_params['fullWellCapacity'],
                            nBits=self.camera_params['nBits'],                 
                            quantumEfficiency=self.camera_params['quantumEfficiency'] ,
                            shotNoise=self.camera_params['shotNoise'],
                            darkCurrent=self.camera_params['darkCurrent'],
                            readoutNoise=self.camera_params['readoutNoise'],
                            gain=self.camera_params['gain'],
                            quantization_conversion=self.camera_params['quantization_conversion'],
                            sensorType=self.camera_params['sensorType'],
                            darkCalibration=self.camera_params['darkCalibration'],
                            noiseFlag=noiseFlag,
                            logger=self.logger,
                            **camera_kwargs)
        
        self.fake_src_dict = {}
        self._mft_cache = {}       # MFT kernels per (lambda/D in px, image size, pupil size)

    def get_frame(self, src, phase):
        """
        Generate a science camera frame based on the input phase and source object.

        Parameters
        ----------
        src : Source
            Input source object, either a point source or extended source (e.g., Sun). 
            During the init, computes the ideal telescope PSF at 500nm.
        phase : np.ndarray
            Optical phase at the science detector.

        Returns
        -------
        np.ndarray
            Final science frame with detector effects.
        """

        fwhm = (src.wavelength / self.telescope_diameter) * (206265 / self.plate_scale)

        if src.tag == 'source':
            # First, check whether we have an ideal PSF for the current wavelength or not

            key_wavelength = str(int(src.wavelength*1e9))

            if key_wavelength not in self.fake_src_dict.keys():
                # Compute Ideal PSF for this wavelength
                self.fake_src_dict[key_wavelength] = self.compute_psf(phase*0, fwhm).cpu().numpy()   
            
            # Continue computing the PSF of the source                     
            frame = self.compute_psf(phase, fwhm)
        
        elif src.tag == 'sun':
            if src.fov < self.fieldOfView:
                raise ValueError('ScienceCam::get_frame - The source FoV is smaller than the camera FoV.')
            t0 = time.time()
            # If the phase is not 3D, we need to repeat it for each subDir --> typically when the phase uses the DM only
            if np.ndim(phase) < 3:
                phase = np.repeat(phase[np.newaxis,:,:], src.nSubDirs**2, axis=0)
            
            list_phase = [phase[i, :, :] for i in range(phase.shape[0])]
            t1 = time.time()
            # Interpolate sun subDirs to the adequate size given the camera PS
            subDirs_torch = torch.from_numpy(src.subDirs_sun).contiguous()
            subDirs_torch = subDirs_torch.view(subDirs_torch.shape[0], subDirs_torch.shape[1], -1).permute(2, 0, 1)
            new_size = np.round((src.subDirs_coordinates[2,0,0]+src.subDir_margin)/self.plate_scale).astype(int)
            subDirs_torch = torch.nn.functional.interpolate(subDirs_torch.unsqueeze(0), size=(new_size, new_size), 
                                                            mode='bilinear', align_corners=True).squeeze(0).contiguous()  
            t2 = time.time()          
            # Compute in parallel the PSF for each subdir --> shape matches that of the source, using the camera plate scale
            psf = Parallel(n_jobs=1, prefer='threads')(delayed(self.compute_psf)(list_phase[i], fwhm, subDirs_torch.shape[2]) for i in range(len(list_phase)))
            t3 = time.time()
            # Convolute in parallel the PSF of each subDir with the sun patch of that subdir
            # This is faster than compute the FFT2 over 3D tensors and avoid looping, at least on CPU
            sun_patches = Parallel(n_jobs=1, prefer='threads')(delayed(self.compute_image)(subDirs_torch[i,:,:], psf[i]) 
                                                           for i in range(len(list_phase))) 
            t4 = time.time()
            # Resize the 2D filter
            filter_2D_torch = torch.from_numpy(src.filter_2D).contiguous().to(self.device)
            filter_2D_torch = filter_2D_torch.view(filter_2D_torch.shape[0], filter_2D_torch.shape[1], -1).permute(2, 0, 1)
            new_size = np.round(src.subDirs_coordinates[2,0,0]/self.plate_scale).astype(int)
            filter_2D_torch = torch.nn.functional.interpolate(filter_2D_torch.unsqueeze(0), size=(new_size, new_size), 
                                                            mode='bilinear', align_corners=True).squeeze(0).contiguous()
            # Combine the sun patches into a unique PSF

            sun_PSF_combined = torch.zeros(np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                           np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                           dtype=torch.float64, device=self.device).contiguous()
            
            sun_psf_tmp_3D = torch.zeros((np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                          np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), src.nSubDirs*src.nSubDirs), 
                                          dtype=torch.float64, device=self.device).contiguous()

            # The small gain corrector is used to normalize the filter after it was interpolated so that differences below 1-2% can be compensated
            small_gain_corrector = torch.zeros((np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                src.nSubDirs*src.nSubDirs), dtype=torch.float64, device=self.device).contiguous()
            
            for i in range(len(sun_patches)):
                dirX = i//src.nSubDirs
                dirY = i%src.nSubDirs
                
                gcorner_x = dirX*np.round((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale)).astype(int)
                gcorner_y = dirY*np.round((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale)).astype(int)

                gcorner_x_end = gcorner_x + filter_2D_torch[i,:,:].shape[-2]
                gcorner_y_end = gcorner_y + filter_2D_torch[i,:,:].shape[-1]

                start = sun_patches[i].shape[0]//2 - filter_2D_torch.shape[1]//2
                
                sun_psf_tmp_3D[gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end, i] = filter_2D_torch[i,:,:] * sun_patches[i][start:start+filter_2D_torch.shape[1], 
                                                                                                                              start:start+filter_2D_torch.shape[1]]
                
                small_gain_corrector[gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end, i] = filter_2D_torch[i,:,:]
            
            # Crop the region of interest
            sun_PSF_combined = torch.sum(sun_psf_tmp_3D,axis=2) / torch.sum(small_gain_corrector,axis=2)
            
            offset = sun_PSF_combined[0].shape[0]//2 - self.nPix//2

            frame = sun_PSF_combined[offset:offset+self.nPix, offset:offset+self.nPix]
            t5 = time.time()
            
            self.logger.debug(f'to 3D; {t1-t0}, interpolate: {t2-t1}, compute psf: {t3-t2}, compute image: {t4-t3}, combine: {t5-t4}, total: {t5-t0}')
        else:
            raise ValueError(f'ScienceCam::get_frame - The source tag ({src.tag}) is not supported.')
        
        # Normalize frame energy so that it sums 1
        total_energy = torch.sum(frame)
        
        if total_energy > 0:
            frame /= total_energy           

        return frame.cpu().numpy()
    
    def apply_noise(self, frame, total_photons):
        """
        Apply detector noise to the science frame.

        Parameters
        ----------
        frame : np.ndarray
            The input noise-free science frame.
        total_photons : int or float
            Total number of photons arriving.

        Returns
        -------
        np.ndarray
            The noisy science frame.
        """
        # Compute photons arriving to the sensor
        input_photons = total_photons * self.lightRatio
        # Add detector noise
        frame = self.cam.integrate(frame, input_photons)

        return frame

    def compute_psf(self, phase, fwhm, nPix=None):
        """
        Compute the PSF from a phase map with a matrix Fourier transform (MFT) evaluated directly at the camera pixels.

        The pupil field P exp(i phase) is used on the native telescope grid (no resampling of the phase, which would
        low-pass filter it and bias the Strehl ratio upwards). Pixel k of the image is at the angle
        (k - nPix//2) * plate_scale, exactly, for any plate scale. Sign and orientation are those of an FFT: a phase that
        grows along the columns moves the PSF towards larger column indices. The PSF is normalised so that its energy over
        the whole image plane is 1 (Parseval); the returned crop of nPix x nPix pixels sums to the fraction of the energy
        inside the field.

        Parameters
        ----------
        phase : np.ndarray
            Phase on the telescope pupil grid, in radians at the wavelength of the source.
        fwhm : float
            lambda/D expressed in camera pixels (lambda / D / plate_scale). Must be >= 1 (Nyquist: >= 2).
        nPix : int, optional
            Image size in pixels. If None, uses the camera size.

        Returns
        -------
        torch.Tensor
            PSF of nPix x nPix pixels.
        """
        if nPix is None:
            nPix = self.nPix
        if fwhm < 1:
            raise ValueError('FWHM must be greater than 2 to guarantee enough spatial sampling.')

        n = self.pupil.shape[0]
        A = self._mft_kernel(fwhm, nPix, n)
        phase_torch = torch.as_tensor(phase, dtype=torch.float64, device=self.device)
        pupil_torch = torch.as_tensor(self.pupil, dtype=torch.float64, device=self.device)
        field = torch.polar(pupil_torch, phase_torch)

        # E[k, l] = sum_{m, q} A[k, m] field[m, q] A[l, q]
        E = A @ field @ A.transpose(0, 1)
        psf = E.real ** 2 + E.imag ** 2

        # Energy over the whole image plane (Parseval): M^2 sum |field|^2, with M = lambda / (d plate_scale) = fwhm * n
        norma = (fwhm * n) ** 2 * torch.sum(pupil_torch ** 2)
        if norma > 0:
            psf /= norma
        else:
            self.logger.warning('ScienceCam::compute_psf - Could not normalize PSF.')

        return psf.contiguous()

    def _mft_kernel(self, fwhm, nPix, n):
        """
        MFT matrix A[k, m] = exp(-2 pi i u_k x_m) with x_m = (m - (n-1)/2) (pupil samples centred on the pupil) and
        u_k = (k - nPix//2) / (fwhm n) (cycles per pupil sample at the angle of image pixel k). Cached per (fwhm, nPix, n).
        """
        key = (round(float(fwhm), 12), int(nPix), int(n))
        A = self._mft_cache.get(key)
        if A is None:
            x = torch.arange(n, dtype=torch.float64, device=self.device) - (n - 1) / 2
            u = (torch.arange(nPix, dtype=torch.float64, device=self.device) - nPix // 2) / (fwhm * n)
            A = torch.exp(-2j * np.pi * torch.outer(u, x))
            self._mft_cache[key] = A
        return A

    def compute_image(self, sci_object, coherence):
        """
        Compute the convolved image from an object and a PSF.

        Parameters
        ----------
        sci_object : np.ndarray
            Intensity map of the object.
        coherence : torch.Tensor
            PSF to convolve with the object, same size, centred at pixel (N//2, N//2) as returned by compute_psf.

        Returns
        -------
        torch.Tensor
            Final image after convolution.
        """        
        object_torch = sci_object.contiguous().to(self.device)

        # Compute FFT2

        object_fft    = torch.fft.fft2(object_torch, norm='backward', dim=(0,1))
        # The PSF centre (pixel N//2, see compute_psf) is moved to the origin, so that the circular convolution does not shift
        # the object, for odd and even N
        coherence_fft = torch.fft.fft2(torch.fft.ifftshift(coherence, dim=(0,1)), norm='backward', dim=(0,1))

        # Convolute
        image_complex = torch.fft.ifft2(object_fft*coherence_fft, norm='forward', dim=(0,1))
        image = torch.sqrt(image_complex.real**2 + image_complex.imag**2)

        # Normalize energy

        norma = torch.sum(image)/torch.sum(sci_object)

        if norma > 0:
            image /= norma
        else:
            self.logger.warning('ScienceCam::compute_image - Could not normalize patch image generation')


        return image

    def setup_logging(self, logging_level=logging.INFO):
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