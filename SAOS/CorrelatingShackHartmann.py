# -*- coding: utf-8 -*-
"""
Created on Thu May 8 2025

@author: nlinares
"""

import time

import logging
import logging.handlers
from queue import Queue

import numpy as np

import torch

import cv2

from .Detector import Detector
from .tools.tools import bin_ndarray

"""
Shack Hartmann Wavefront Sensor Module
=================

This module contains the `CorrelatingShackHartmann` class, used for modeling a SH-WFS in adaptive optics simulations.
"""

class CorrelatingShackHartmann:
    def __init__(self,
                 nSubap:float,
                 telescope,
                 src,
                 lightRatio:float,
                 plate_scale:float,
                 fieldOfView:float,
                 guardPx:int,
                 fft_fieldOfView_oversampling:float=0,
                 use_brightest:int = 9,
                 threshold_convolution:float = 0.05,
                 unit_in_rad = False,
                 noiseFlag:bool=False,
                 logger=None,
                 **kwargs):
        
        """
        Initialize a Correlating Shack-Hartmann Wavefront Sensor (WFS).

        Parameters
        ----------
        nSubap : float
            Number of subapertures across the pupil diameter.
        telescope : Telescope
            Telescope object to which the WFS is attached.
        src : Source
            Source object (SUN).
        lightRatio : float
            Threshold ratio to select valid subapertures based on flux.
        plate_scale : float
            Plate scale of the WFS in [arcsec/px].
        fieldOfView : float
            Field of view of the WFS in [arcsec].
        guardPx : int
            Number of pixels between subapertures.
        fft_fieldOfView_oversampling : float, optional
            Extra FoV in [arcsec] that is taken for the FFT computation, in order to reduce wrapping effects.
        use_brightest : int, optional
            Picks the n brightest pixels as threshold for center-of-gravity spot detection.
        threshold_convolution : float, optional
            Cut-off threshold for Gaussian convolution.
        unit_in_rad : bool, optional
            Return slopes in radians if True, pixels otherwise.
        noiseFlag : bool, optional
            If True, the detector includes noise using the kwargs params/default config. By default, False.            
        logger : logging.Logger, optional
            Logger for WFS diagnostics.
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
                Detector parameter. Conversion gain to discretize the measurement [e-/px]. Default 70.5e-/DN.
            sensorType : str, optional
                Detector parameter. Sensor type ('CCD', 'CMOS', 'EMCCD'). Default is 'CCD'.
            darkCalibration : int, optional
                Detector parameter. Number of frames to calibrate the dark. Default 20.
            randomState : int, optional
                Detector parameter. Seed for the random number generator. Default is None.
            integrationTime : float, optional
                Detector parameter. Integration time for the detector [s]. Default is the sampling time.
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        self.tag                            = 'shackHartmann'

        # Check the source type, just in case:

        if src.tag != 'sun':
            self.logger.error(f'CorrelatingShackHartmann::init - Source type is {src.tag}, which is not supported. This module expects the type to be SOURCE')
            raise AttributeError(f'Expected SOURCE type, but {src.tag} was provided instead.')

        self.plate_scale                    = plate_scale
        self.fieldOfView                    = fieldOfView
        self.guardPx                        = guardPx

        self.fft_fieldOfView_oversampling   = fft_fieldOfView_oversampling

        self.nSubap                         = nSubap
        self.lightRatio                     = lightRatio
        self.threshold_convolution          = threshold_convolution
        self.use_brightest                  = use_brightest
        self.unit_in_rad                    = unit_in_rad

        self.device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
       
        # Subapeture definition
        self.subaperture_size           = telescope.D / self.nSubap
        self.npix_lenslet               = int(np.round((self.fieldOfView + self.fft_fieldOfView_oversampling) / self.plate_scale))
        self.npix_subap                 = int(np.round(self.fieldOfView / self.plate_scale))
        self.npix_phase                 = telescope.resolution // self.nSubap

        # Compensate the phase intensity due to the interpolation
        self.pupil_interpolation_mask   = torch.nn.functional.interpolate(torch.from_numpy(telescope.pupil.astype(float)).unsqueeze(0).unsqueeze(0), 
                                                                             size=(self.nSubap*self.npix_phase, self.nSubap*self.npix_phase), 
                                                                             mode='bilinear', align_corners=True).squeeze().numpy()

        # Detector camera 

        self.camera_size                = self.nSubap * self.npix_lenslet + (self.nSubap + 1)* self.guardPx

        self.camera_params = dict()
        # Default parameters correspond to pco.dimax 3.6 ST @500nm

        self.camera_params['fullWellCapacity'] = kwargs.get('fullWellCapacity', 60000)
        self.camera_params['nBits'] = kwargs.get('nBits', 10)
        self.camera_params['quantumEfficiency'] = kwargs.get('quantumEfficiency', 0.64)
        self.camera_params['shotNoise'] = kwargs.get('shotNoise', 1)
        self.camera_params['darkCurrent'] = kwargs.get('darkCurrent', 250)
        self.camera_params['readoutNoise'] = kwargs.get('readoutNoise', 60)
        self.camera_params['gain'] = kwargs.get('gain', 1)
        self.camera_params['quantization_conversion'] = kwargs.get('quantization_conversion', 70.5)
        self.camera_params['sensorType'] = kwargs.get('sensorType', 'CMOS')
        self.camera_params['darkCalibration'] = kwargs.get('darkCalibration', 20)
        self.camera_params['randomState'] = kwargs.get('randomState', None)
        self.camera_params['integrationTime'] = kwargs.get('integrationTime', telescope.samplingTime)

        camera_kwargs = {'randomState':self.camera_params['randomState'], 
                         'integrationTime':self.camera_params['integrationTime']}


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
        
        # Photon count info

        self.area_eff         = telescope.area_eff
        self.integration_time = telescope.samplingTime

        # index of valid slopes X and Y
        self.logger.info('CorrelatingShackHartmann::__init__ - Selecting valid subapertures based on flux considerations..')

        self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture, [self.nSubap,self.nSubap])

        self.valid_subapertures = np.zeros((self.nSubap, self.nSubap)).astype(bool)

        for i in range(self.nSubap):
            for j in range(self.nSubap):
                tmp_pupil = np.sum(self.pupil_interpolation_mask[i*self.npix_phase:(i+1)*self.npix_phase, 
                                                   j*self.npix_phase:(j+1)*self.npix_phase]) / (self.npix_phase**2)
                if tmp_pupil == 1: # Criteria: there are phase points at every point of the phase
                    self.valid_subapertures[i, j] = True

        self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])

        [self.validLenslets_x , self.validLenslets_y] = np.where(self.valid_subapertures==True)
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
        self.nSignal = 2*self.nValidSubaperture     

        self.geometric_references  = np.zeros([self.nSubap*2,self.nSubap])

        # Compute conversion from px to rad
        self.calib_unit(telescope, src)

    def calib_unit(self, telescope, src):
        """
        Calib the unit of the sensor if the user specifies radians instead of pixels.

        Parameters
        ----------
        telescope : Telescope
            The telescope object providing phase and pupil info.
        src : Source
            Source object for flux and wavelength reference.

        Returns
        -------
        None
        """  
        
        # First, we generate a phase
        self.slopes_units           = 1

        if self.unit_in_rad:
            self.logger.info('CorrelatingShackHartmann::calib_unit - Setting slopes units..') 
            [Tip, Tilt] = np.meshgrid(np.linspace(-np.pi,np.pi,telescope.resolution,endpoint=False),np.linspace(-np.pi,np.pi,telescope.resolution,endpoint=False))

            mode_amp = np.std(Tip)
            amp_list = [i for i in range(-2,3)]

            mean_slope = np.zeros(5)
            input_std = np.zeros(5)

            for i in range(len(amp_list)):
                calibration_phase_tip = amp_list[i]*Tip*telescope.pupil
                calibration_phase_tip = np.repeat(calibration_phase_tip[np.newaxis,:,:], src.nSubDirs**2, axis=0)
                signalTip,_,_ = self.wfs_measure(calibration_phase_tip, src, None, None)

                calibration_phase_tilt = amp_list[i]*Tilt*telescope.pupil
                calibration_phase_tilt = np.repeat(calibration_phase_tilt[np.newaxis,:,:], src.nSubDirs**2, axis=0)
                signalTilt,_,_ = self.wfs_measure(calibration_phase_tilt, src, None, None)

                mean_slope[i] = np.mean(signalTip[:self.nValidSubaperture] + signalTilt[:self.nValidSubaperture])
                input_std[i] = np.std(calibration_phase_tip[telescope.pupil])
                
            p = np.polyfit(amp_list*np.asarray(mode_amp), mean_slope, deg = 1)
            self.slopes_units = np.abs(p[0]) # [px/rad]
        
        
    def initialize_wfs(self, telescope, src):
        """
        Initialize the Shack-Hartmann WFS by measuring reference slopes
        and determining slope units.

        Parameters
        ----------
        telescope : Telescope
            The telescope object providing phase and pupil info.
        src : Source
            Source object for flux and wavelength reference.

        Returns
        -------
        None
        """
        self.logger.info('CorrelatingShackHartmann::initialize_wfs - Acquiring reference slopes..')
        null_phase = np.zeros((src.nSubDirs**2, telescope.resolution, telescope.resolution))       
        _, reference_slopes_maps,_, pseudoref = self.wfs_measure(null_phase, src)        
                
        self.logger.info('CorrelatingShackHartmann::initialize_wfs - Done!') 
        
        return pseudoref, reference_slopes_maps
    
    def cross_correlate(self, subaps, pseudoref):
        """
        Cross-correlate subaperture images with a pseudoref image.

        Parameters
        ----------
        subaps : np.ndarray
            Subaperture images.
        pseudoref : np.ndarray
            Pseudoref image.

        Returns
        -------
        np.ndarray
            Cross-correlation result.
        """
        self.logger.debug('CorrelatingShackHartmann::cross_correlate')
        
        # Compute FFT2

        subaps_fft = torch.fft.fft2(subaps, norm='forward', dim=(-2, -1))
        pseudoref_fft = torch.fft.fft2(pseudoref, norm='forward', dim=(-2, -1))

        # Compute cross-correlation

        cross_correlation_complex = torch.fft.fftshift(torch.fft.ifft2(subaps_fft * pseudoref_fft.conj(), norm='forward', dim=(-2, -1)), dim=(-2,-1))
        cross_correlation = torch.sqrt(cross_correlation_complex.real**2 + cross_correlation_complex.imag**2)

        return cross_correlation

    def centroid(self, image, use_brightest=9):
        """
        Compute center of gravity for subaperture spots.

        Parameters
        ----------
        image : torch.Tensor
            Subaperture image cube.
        use_brightest : float, optional
            Minimum intensity to include in centroid.

        Returns
        -------
        np.ndarray
            X and Y centroids per subaperture.
        """

        n_subap, H, W = image.shape

        # flatten each subap
        flat = image.view(n_subap, -1)

        # Obtener umbral para los brightest
        topk_vals, _ = torch.topk(flat, use_brightest, dim=1)
        thresholds = topk_vals[:, -1].unsqueeze(1)

        # Máscara para pixels válidos
        mask = flat >= thresholds

        # Aplicar máscara
        filtered = flat - thresholds
        filtered[~mask] = 0

        # Restaurar forma original
        filtered = filtered.view(n_subap, H, W)

        # Crear mapas de coordenadas
        yy, xx = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=image.dtype),
            torch.arange(W, device=image.device, dtype=image.dtype),
            indexing='ij'
        )

        # Expandir para batch
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        # Calcular normalización (suma total)
        norm = filtered.sum(dim=(1, 2))

        # Calcular centroides
        x_c = (filtered * xx).sum(dim=(1, 2)) / norm
        y_c = (filtered * yy).sum(dim=(1, 2)) / norm

        centroids = torch.stack((x_c, y_c), dim=1)

        return centroids.cpu().numpy()

    # This function takes the phase at the pupil as input. Then, the flux at each subaperture (pupil function) is multiplied 
    # by the complex phase to obtain the PSF per subaperture, as an array of dimensions (nSubap**2, n_pix_lenslet_init, n_pix_lenslet_init)
    # The subapertures are sorted from left to right, top to bottom.

    def get_psf(self, phase, fwhm, npix_sun):
        """
        Get the electromagnetic field per subaperture based on input phase.

        Parameters
        ----------
        phase : np.ndarray
            List of Wavefront phase in radians per subdirection
        fwhm : float
            Full width at half maximum for the PSF in [px].
        npix_sun : float
            Number of pixels in the sun image [px]
        
        Returns
        -------
        np.ndarray
            Complex field per subaperture.
        """

        self.logger.debug('CorrelatingShackHartmann::get_psf')
        # Get dimensions to keep the FWHM stable during the computation
        if fwhm < 1:
            raise ValueError('FWHM must be greater than 2 to guarantee enough spatial sampling.')
        nFFT = np.round(fwhm * npix_sun).astype(int)
        # Define pupil
        start = (nFFT - npix_sun) // 2
        end = start + npix_sun
        square_pupil = np.zeros((nFFT, nFFT), dtype=float)
        square_pupil[start:end, start:end] = 1.0

        t0 = time.time()
        # Rescale the phase
        input_phase_torch = torch.from_numpy(phase).contiguous().to(self.device)
        square_pupil_torch = torch.from_numpy(square_pupil).contiguous().to(self.device)

        phase_rescaled = torch.nn.functional.interpolate(input_phase_torch.unsqueeze(0), 
                                                            size=(self.nSubap*self.npix_phase, self.nSubap*self.npix_phase), 
                                                            mode='bilinear', align_corners=True).squeeze(0).contiguous()
        t1 = time.time()
        # Reshape the subapertures to a grid of subapertures. The sensor can be zeropadded, so the phase is filling the central part of the subaperture
        phase_reshaped = torch.empty((self.nSubap**2, input_phase_torch.shape[0], 
                                      self.npix_phase+1, self.npix_phase+1), dtype=torch.float32, device=self.device).contiguous()

        C, H, W = phase_rescaled.shape

        # Build row indices
        row_ids = torch.arange(H)
        insert_rows = ((row_ids % self.npix_phase == 0) & (row_ids < H - 2) & (row_ids > 0)) | (row_ids == (H - self.npix_phase - 1))
        extra_rows = row_ids[insert_rows]
        row_idx = torch.cat([row_ids, extra_rows]).sort().values
        row_idx[-self.npix_phase-2:-self.npix_phase] = row_idx[-self.npix_phase-2:-self.npix_phase].flip(0)

        # Build column indices
        col_ids = torch.arange(W)
        insert_cols = ((col_ids % self.npix_phase == 0) & (col_ids < W - 2) & (col_ids > 0)) | (col_ids == (W - self.npix_phase - 1))
        extra_cols = col_ids[insert_cols]
        col_idx = torch.cat([col_ids, extra_cols]).sort().values
        col_idx[-self.npix_phase-2:-self.npix_phase] = col_idx[-self.npix_phase-2:-self.npix_phase].flip(0)

        # Apply advanced indexing
        phase_reshaped = phase_rescaled[:, row_idx][:, :, col_idx]  # shape: (nSubDirs**2, H', W')
        phase_reshaped = phase_reshaped.view(C, self.nSubap, self.npix_phase+1, self.nSubap, self.npix_phase+1)
        phase_reshaped = phase_reshaped.permute(1, 3, 0, 2, 4).reshape(self.nSubap**2, C, self.npix_phase+1, self.npix_phase+1)

        t2 = time.time()        
        phase_rescaled_valids = torch.nn.functional.interpolate(phase_reshaped[self.valid_subapertures_1D, :, :, :], 
                                                                size=(npix_sun, npix_sun), 
                                                                mode='bilinear', align_corners=True).squeeze(1).contiguous().float()
        t3 = time.time()
        # Generate an array of phase, filling the valid subapertures  
        rows = torch.where(square_pupil_torch.any(dim=1))[0]
        cols = torch.where(square_pupil_torch.any(dim=0))[0]

        row_start, row_end = rows[0].item(), rows[-1].item() + 1
        col_start, col_end = cols[0].item(), cols[-1].item() + 1
        # Allocate full tensor
        cube_em = torch.zeros((self.nSubap**2, input_phase_torch.shape[0], nFFT, nFFT), dtype=torch.complex64, device=self.device)
        sub_mask = square_pupil_torch[row_start:row_end, col_start:col_end].float()
        # Exponential
        exp_block = torch.polar(sub_mask, phase_rescaled_valids)

        cube_em[self.valid_subapertures_1D,:, row_start:row_end, col_start:col_end] = exp_block
        # Apply light scaling
        t4 = time.time()
        # Get the PSF
        psf = torch.zeros((self.nSubap**2, input_phase_torch.shape[0], npix_sun, npix_sun), dtype=torch.float32, device=self.device)

        fft_res = torch.fft.fft2(cube_em[self.valid_subapertures_1D, :, :, :], dim=(-2, -1), norm='backward').contiguous()  # same as dividing by nFFT²

        # Shift zero frequency to center
        fft_res = torch.fft.fftshift(fft_res, dim=(-2, -1))

        # Compute normalized intensity
        fft_res = torch.sqrt(fft_res.real**2 + fft_res.imag **2)**2

        # Normalize energy
        norma = torch.sum(fft_res[:, :, row_start:row_end, col_start:col_end], dim=(-2, -1), kePhoto=True)

        fft_res = fft_res / norma
           
        # Crop to desired region
        psf[self.valid_subapertures_1D, :, :, :] = fft_res[:, :, row_start:row_end, col_start:col_end]
        t5 = time.time()
        
        self.logger.debug(f'CorrelatingShackHartmann::get_psf - Time taken for each step: '
                         f'Rescale input phase: {t1-t0} [s], Reshape into subaps: {t2-t1} [s], Interpolate to npix_sun: {t3-t2} [s], '
                         f'Compute exponential: {t4-t3} [s], PSF: {t5-t4} [s], Total processing time: {t5-t0}')
        return psf
    
    def compute_images(self, psf, subDirs_sun, new_px):
        """
        Compute correlated subaperture images using convolution.

        Parameters
        ----------
        psf : torch.Tensor
            PSF of the subapertures.
        subDirs_sun : np.ndarray
            Sun subdirectories used as object.
        new_px : int
            New size in pixels for the sun subdirectories.

        Returns
        -------
        torch.Tensor
            Convoluted images.
        """
        
        # Convert to Tensor
        sun_torch = torch.from_numpy(subDirs_sun).contiguous().to(self.device) # 4D: 
        sun_torch = sun_torch.view(sun_torch.shape[0], sun_torch.shape[1], -1).permute(2, 0, 1)

        # Dowsample the sun to match the WFS plate scale
        sun_torch = torch.nn.functional.interpolate(sun_torch.unsqueeze(0), size=(new_px, new_px), mode='area').contiguous() #, align_corners=True
        
        # Compute FFT of the sun subdirs

        sun_fft = torch.fft.fft2(sun_torch, norm='backward', dim=(-2, -1))

        # Compute FFT of the PSF

        psf_fft = torch.fft.fft2(psf[self.valid_subapertures_1D, :, :, :], norm='backward', dim=(-2, -1))

        # Convolute
        sun_patches_complex = torch.fft.fftshift(torch.fft.ifft2(sun_fft*psf_fft, norm='forward', dim=(-2, -1)), dim=(-2, -1))

        sun_patches = torch.sqrt(sun_patches_complex.real**2 + sun_patches_complex.imag**2)

        # Normalize energy
        sun_energy = torch.from_numpy(subDirs_sun).to(self.device).sum(axis=(0,1))
        sun_energy = sun_energy.reshape(1,sun_patches.shape[1],1,1)

        curr_energy = torch.sum(sun_patches, dim=(-2, -1), keepdim=True)
        curr_energy = torch.sum(curr_energy, axis=0, keepdim=True)

        sun_patches = sun_patches * (sun_energy / curr_energy)
 
        return sun_patches
    
    def merges_images(self, sun_patches, src):
        """
        Merge sun patches into a single frame.

        Parameters
        ----------
        sun_patches : torch.Tensor
            The convoluted sun patches per subaperture.
        src : Source
            The solar source object.

        Returns
        -------
        np.ndarray
            The merged full-field images.
        """
        # Resize the 2D filter
        filter_2D_torch = torch.from_numpy(src.filter_2D).contiguous().to(self.device)
        filter_2D_torch = filter_2D_torch.view(filter_2D_torch.shape[0], filter_2D_torch.shape[1], -1).permute(2, 0, 1)
        new_size = np.round(src.subDirs_coordinates[2,0,0] / self.plate_scale).astype(int)
        filter_2D_torch = torch.nn.functional.interpolate(filter_2D_torch.unsqueeze(0), size=(new_size, new_size), 
                                                          mode='bilinear', align_corners=True).squeeze(0).contiguous()

        # Combine the sun patches into a unique PSF

        sun_PSF_combined = torch.zeros(sun_patches.shape[0], np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                             np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                             dtype=torch.float32, device=self.device).contiguous()
        
        sun_psf_tmp_3D = torch.zeros(sun_patches.shape[0], src.nSubDirs*src.nSubDirs, 
                                     np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                     np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                     dtype=torch.float32, device=self.device).contiguous()

        small_gain_corrector = torch.zeros(sun_patches.shape[0], src.nSubDirs*src.nSubDirs, 
                                    np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                    np.ceil((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                    dtype=torch.float32, device=self.device).contiguous()

        for i in range(sun_patches.shape[1]):
            dirX = i//src.nSubDirs
            dirY = i%src.nSubDirs
            
            gcorner_x = dirX*np.round((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale)).astype(int)
            gcorner_y = dirY*np.round((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale)).astype(int)

            gcorner_x_end = gcorner_x + filter_2D_torch[i,:,:].shape[-2]
            gcorner_y_end = gcorner_y + filter_2D_torch[i,:,:].shape[-1]

            start = sun_patches.shape[-1]//2 - filter_2D_torch.shape[1]//2
            
            sun_psf_tmp_3D[:,i, gcorner_x:gcorner_x_end,
                                gcorner_y:gcorner_y_end] = filter_2D_torch[i,:,:] * sun_patches[:,i,start:start+filter_2D_torch.shape[1],
                                                                                                    start:start+filter_2D_torch.shape[1]]
            
            small_gain_corrector[:,i, gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end] = filter_2D_torch[i,:,:]
        
        # Crop the external region, which might be affected by the windowing effect.
        # The patch is larger so that the final crop matches the FoV of the object.
        sun_PSF_combined = torch.sum(sun_psf_tmp_3D,axis=1) / torch.sum(small_gain_corrector,axis=1)
        
        offset = sun_PSF_combined.shape[-1]//2 - self.npix_lenslet//2

        merged_image = sun_PSF_combined[:,offset:offset+self.npix_lenslet, offset:offset+self.npix_lenslet].cpu().numpy()
        
        return merged_image
  
    def create_full_frame(self, subaps):
        """
        Combine subaperture images into full detector frame.

        Parameters
        ----------
        subaps : np.ndarray
            Subaperture spot images.

        Returns
        -------
        np.ndarray
            Reconstructed full sensor image.
        """
        self.logger.debug('shackHartmann::create_full_frame')

        ideal_frame = np.zeros((self.camera_size, self.camera_size), dtype =float)
        
        index_valid = 0

        center_offset_x = self.guardPx + (self.npix_lenslet // 2) - self.npix_subap//2
        center_offset_y = self.guardPx + (self.npix_lenslet // 2) - self.npix_subap//2

        subap_offset = (self.npix_lenslet - self.npix_subap) // 2
        
        for x in range(self.nSubap):
            for y in range(self.nSubap):
                if self.valid_subapertures[x,y]: 

                    init_x = center_offset_x + x * (self.guardPx + self.npix_lenslet)
                    init_y = center_offset_y + y * (self.guardPx + self.npix_lenslet)

                    ideal_frame[init_x:init_x+self.npix_subap, 
                                init_y:init_y+self.npix_subap] = subaps[index_valid, subap_offset:subap_offset+self.npix_subap, subap_offset:subap_offset+self.npix_subap]                         
                    
                    index_valid += 1
        
        return ideal_frame

    def get_subaps(self, noisy_frame):
        """
        Extract individual lenslet spot images from full detector image.

        Parameters
        ----------
        noisy_frame : np.ndarray
            2D detector frame.

        Returns
        -------
        np.ndarray
            Subaperture cubes [nSubaps, width, height].
        """
        self.logger.debug('shackHartmann::get_subaps')

        index_valid = 0

        center_offset_x = self.guardPx + (self.npix_lenslet // 2) - self.npix_subap//2
        center_offset_y = self.guardPx + (self.npix_lenslet // 2) - self.npix_subap//2

        subaps = np.zeros((self.nValidSubaperture, self.npix_subap, self.npix_subap), dtype=float)
        
        for x in range(self.nSubap):
            for y in range(self.nSubap):
                if self.valid_subapertures[x,y]: 

                    init_x = center_offset_x + x * (self.guardPx + self.npix_lenslet)
                    init_y = center_offset_y + y * (self.guardPx + self.npix_lenslet)

                    subaps[index_valid, :, :] = noisy_frame[init_x:init_x+self.npix_subap, init_y:init_y+self.npix_subap]                         
                    
                    index_valid += 1

        return torch.Tensor(subaps).contiguous().to(self.device)
            
#%% SH Measurement
    def wfs_integrate(self, ideal_frame, subaps, nPhoton, pseudoref=None, reference_slopes=None):
        """
        Integrate the full detector image and compute valid slopes.

        Parameters
        ----------
        ideal_frame : np.ndarray
            Full-frame ideal image.
        subaps : np.ndarray
            Spot images per subaperture.
        nPhoton : int
            Number of incident photons
        pseudoref : torch.Tensor
            Subaperture used as pseudo-reference for the correlations, if None one subaperture is picked
        Returns
        -------
        tuple
            Slopes 1D, slopes 2D, and final noisy image.
        """
        # propagate to detector to add noise and detector effects
        noisy_frame = self.cam.integrate(ideal_frame, nPhoton)
        subaps = self.get_subaps(noisy_frame)

        # Check if we have a pseudo reference 
        if pseudoref is None:
            pseudoref = torch.clone(subaps[0,:,:]).contiguous().to(self.device)

        # Compute correlation images
        correlation_images = self.cross_correlate(subaps, pseudoref)
        # compute the centroid on valid subaperture
        centroid_lenslets = self.centroid(correlation_images, self.use_brightest)
        
        # discard nan and inf values
        val_inf = np.where(np.isinf(centroid_lenslets))
        val_nan = np.where(np.isnan(centroid_lenslets)) 
        
        if np.shape(val_inf)[1] !=0:
            self.logger.warning('CorrelatingShackHartmann::wfs_integrate - Some subapertures are giving inf values!')
            centroid_lenslets[np.where(np.isinf(centroid_lenslets))] = 0
        
        if np.shape(val_nan)[1] !=0:
            self.logger.warning('CorrelatingShackHartmann::wfs_integrate - Some subapertures are giving NaN values!')
            centroid_lenslets[np.where(np.isnan(centroid_lenslets))] = 0
            
        sx = np.zeros((self.nSubap, self.nSubap))
        sy = np.zeros((self.nSubap, self.nSubap))

        sx[self.validLenslets_x, self.validLenslets_y] = centroid_lenslets[:,0]
        sy[self.validLenslets_x, self.validLenslets_y] = centroid_lenslets[:,1]

        if reference_slopes is None:
            signal_2D                           = np.concatenate((sx, sy)) - self.geometric_references
        else:
            signal_2D                           = np.concatenate((sx, sy)) - reference_slopes

        signal_2D[~self.valid_slopes_maps]  = 0
        
        signal_2D                      = signal_2D/self.slopes_units
        signal                         = signal_2D[self.valid_slopes_maps]
        
        return signal, signal_2D, noisy_frame, pseudoref
    
    # Receives the phase [rad]] and return the slopes measured by the SH in [px]
    # Expects the ideal frame WITHOUT pupil applied.
    def wfs_measure(self,phase_in, src, pseudoref=None, reference_slopes=None):
        """
        Measure slopes from a wavefront phase using the SH-WFS.

        Parameters
        ----------
        phase_in : np.ndarray
            Phase map input [radians].
        src : Source
            Source object.
        pseudoref : torch.Tensor
            Pseudo-reference image used for the correlations, if None the method picks one.
        reference_slopes : np.ndarray
            Reference for the WF sensor, if None, geometric references are used.

        Returns
        -------
        tuple
            Slopes 1D, slopes 2D, and detector image.
        """
        self.logger.debug("CorrelatingShackHartmann::wfs_measure")
        # Check input parameters
        if (phase_in is None) or (src is None):
            self.logger.error("CorrelatingShackHartmann::wfs_measure - Phase or Source are none.")
            raise ValueError('CorrelatingShackHartmann::wfs_measure - Phase or Source are none.') 
        # compute fwhm
        fwhm = src.wavelength * 206265 / (self.subaperture_size * self.plate_scale)
        # Compute the new number of pixels for the sun patches to match the WFS plate scale
        new_px = np.round((src.subDirs_coordinates[2,0,0] + src.subDir_margin)/ self.plate_scale).astype(int) # Coordinate 2 has the width of the subDir
        # If the phase is not 3D, we need to repeat it for each subDir --> typically when the phase uses the DM only
        if np.ndim(phase_in) < 3:
            phase_in = np.repeat(phase_in[np.newaxis,:,:], src.nSubDirs**2, axis=0)
        t0 = time.time()
        # compute the PSF intensity
        psf = self.get_psf(phase_in, fwhm, new_px)
        t1 = time.time()
        # Convolute with the sun patches - only valid subaps

        sun_patches = self.compute_images(psf, src.subDirs_sun, new_px)
        t2 = time.time()
        # Merge the patches into a single image per subap

        I = self.merges_images(sun_patches, src)
        t3 = time.time()
        # fill camera frame with computed intensity (only valid subapertures)

        ideal_frame = self.create_full_frame(I)
        t4 = time.time()
        # Compute the number of photons considering the flux of the sun, collecting area of the telescope, sampling time and FoV of the WFS 
        # that may be smaller in the simulation than the FoV of the source
        nPhotons = np.round(src.flux * self.area_eff * self.integration_time * self.lightRatio * (self.fieldOfView / src.fov)).astype(int)
        # Compute the frame and the slopes        
        signal, signal_2D, noisy_frame, pseudoref = self.wfs_integrate(ideal_frame, I, nPhotons, pseudoref, reference_slopes)                
        t5 = time.time()
        self.logger.debug(f'PSF: {t1-t0}, Compute images: {t2-t1}, Merge images: {t3-t2}, Create full frame: {t4-t3}, Integrate:{t5-t4}')
        return signal, signal_2D, noisy_frame, pseudoref
              
      
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
