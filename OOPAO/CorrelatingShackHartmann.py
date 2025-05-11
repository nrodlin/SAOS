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
import scipy as sp
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
                 use_brightest:int = 50,
                 threshold_convolution:float = 0.05,
                 unit_in_rad = False,
                 logger=None):
        
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
        is_geometric : bool, optional
            Enable geometric mode (gradient-based measurement).
        threshold_convolution : float, optional
            Cut-off threshold for Gaussian convolution.
        unit_in_rad : bool, optional
            Return slopes in radians if True, pixels otherwise.
        logger : logging.Logger, optional
            Logger for WFS diagnostics.
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
        self.cam                        = Detector(self.camera_size, samplingTime=telescope.samplingTime, logger=self.logger)   # WFS detector object
        self.cam.photonNoise            = 0
        self.cam.readoutNoise           = 0
        
        # Flux definition

        # The flux is divided into the pixels to which the lenslets focuses the image. 
        X_map, Y_map     = np.meshgrid(np.arange(self.npix_lenslet),np.arange(self.npix_lenslet))
        self.X_coord_map = np.atleast_3d(X_map).T
        self.Y_coord_map = np.atleast_3d(Y_map).T
                
        # cube of lenslet zero padded
        self.cube                   = np.zeros([self.nSubap**2,self.npix_lenslet,self.npix_lenslet])
        self.cube_flux              = np.zeros([self.nSubap**2,self.npix_lenslet,self.npix_lenslet],dtype=(complex))
        self.index_x                = []
        self.index_y                = []

        # phasor to center spots in the center of the lenslets
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.npix_lenslet-1,self.npix_lenslet),
                                                 np.linspace(0,self.npix_lenslet-1,self.npix_lenslet))
        
        self.phasor                = np.exp(-(1j*np.pi*(self.npix_lenslet+1)/self.npix_lenslet)*(xx+yy))

        self.phasor_tiled          = np.moveaxis(np.tile(self.phasor[:,:,None],self.nSubap**2),2,0)
        
        # Get subapertures index and flux per subaperture        
        
        self.phasor_expanded                = np.exp(-(1j*np.pi*(self.npix_lenslet+1)/self.npix_lenslet)*(xx+yy))

        self.phasor_expanded_tiled          = np.moveaxis(np.tile(self.phasor_expanded[:,:,None],self.nSubap**2), 2, 0)

        # The normalized flux maps considers the efficiency in the reflectance of the light in the pupil, integration time and area and light ratio derived to the WFS
        # The flux is computed as norm_flux * nPhoton
        
        # change the resolution of the pupil to the number of points for the WFS
        pupil_reflectivity_resized = cv2.resize(telescope.pupilReflectivity, (self.npix_lenslet * self.nSubap, self.npix_lenslet* self.nSubap), interpolation=cv2.INTER_LINEAR)

        self.norm_flux_map = self.lightRatio* pupil_reflectivity_resized * telescope.samplingTime*(telescope.D/(self.nSubap*self.npix_lenslet))**2

        self.initialize_flux(src, self.norm_flux_map)
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)

        self.current_nPhoton = src.nPhoton

        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)

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

        # WFS initialization
        self.initialize_wfs(telescope, src)       
        
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
        self.isInitialized = False

        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        
        self.cam.photonNoise        = 0
        self.cam.readoutNoise       = 0       
        
        # flux per subaperture
        self.reference_slopes_maps  = np.zeros([self.nSubap*2,self.nSubap])
        self.slopes_units           = 1

        self.logger.info('CorrelatingShackHartmann::initialize_wfs - Acquiring reference slopes..')
        null_phase = np.zeros((telescope.resolution, telescope.resolution, src.nSubDirs**2))       
        self.pseudoref = None 
        _, self.reference_slopes_maps,_ = self.wfs_measure(null_phase, src)        
        self.isInitialized = True
        
        self.logger.info('CorrelatingShackHartmann::initialize_wfs - Setting slopes units..')  
        # Compute conversion from px to rad

        # First, we generate a phase
        if self.unit_in_rad:
            [Tip, Tilt] = np.meshgrid(np.linspace(-np.pi,np.pi,telescope.resolution,endpoint=False),np.linspace(-np.pi,np.pi,telescope.resolution,endpoint=False))

            mode_amp = np.std(Tip)
            amp_list = [i for i in range(-2,3)]

            mean_slope = np.zeros(5)
            input_std = np.zeros(5)

            for i in range(len(amp_list)):
                calibration_phase_tip = amp_list[i]*Tip*telescope.pupil
                calibration_phase_tip = np.repeat(calibration_phase_tip[:,:,np.newaxis], src.nSubDirs**2, axis=2)
                signalTip,_,_ = self.wfs_measure(calibration_phase_tip, src)

                calibration_phase_tilt = amp_list[i]*Tilt*telescope.pupil
                calibration_phase_tilt = np.repeat(calibration_phase_tip[:,:,np.newaxis], src.nSubDirs**2, axis=2)
                signalTilt,_,_ = self.wfs_measure(calibration_phase_tilt, src)

                mean_slope[i] = np.mean(signalTip[:self.nValidSubaperture] + signalTilt[:self.nValidSubaperture])
                input_std[i] = np.std(calibration_phase_tip[telescope.pupil])
                
            p = np.polyfit(amp_list*np.asarray(mode_amp), mean_slope, deg = 1)
            self.slopes_units = np.abs(p[0]) # [px/rad]
        
        self.logger.info('CorrelatingShackHartmann::initialize_wfs - Done!') 

        self.cam.photonNoise        = readoutNoise
        self.cam.readoutNoise       = photonNoise
        
        self.print_properties()
    
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

        cross_correlation = torch.abs(torch.fft.fftshift(torch.fft.ifft2(subaps_fft * pseudoref_fft.conj(), norm='forward', dim=(-2, -1)), dim=(-2,-1)))

        return cross_correlation

    def centroid(self, image, use_brightest=16):
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
        im = np.atleast_3d(image.numpy())

        threshold = np.partition(im.reshape(im.shape[0],-1), np.prod(im[0,:,:].shape) - use_brightest, axis=-1)[:, -use_brightest]

        # Filtering those value below the threshold
        im[im<threshold[:,None,None]] = 0

        centroid_out         = np.zeros([im.shape[0],2])
        X_map, Y_map= np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[2]))
        X_coord_map = np.atleast_3d(X_map).T
        Y_coord_map = np.atleast_3d(Y_map).T

        norma                   = np.sum(np.sum(im,axis=1),axis=1)

        centroid_out[:,0]    = np.sum(np.sum(im*X_coord_map,axis=1),axis=1)/norma
        centroid_out[:,1]    = np.sum(np.sum(im*Y_coord_map,axis=1),axis=1)/norma
        return centroid_out
#%% DIFFRACTIVE

    def initialize_flux(self, src, norm_flux_map):
        """
        Initialize per-subaperture flux distribution.

        Parameters
        ----------
        src : Source
            Light source object.
        norm_flux_map : np.ndarray
            Normalized flux across telescope pupil.

        Returns
        -------
        None
        """
        # Create the flux cube storing the flux at each subaperture
        self.cube_flux = np.zeros([self.nSubap**2,self.npix_lenslet,self.npix_lenslet],dtype=float)

        # Build the flux map
        input_flux_map = src.nPhoton * norm_flux_map

        input_flux_map = input_flux_map.reshape(self.nSubap, self.npix_lenslet, 
                         self.nSubap, self.npix_lenslet).transpose(0, 2, 1, 3).reshape(self.nSubap*self.nSubap, 
                                                                                       self.npix_lenslet, self.npix_lenslet) 
        # Assign the illumination to the region, considering zeropadding
        center = self.npix_lenslet // 2
        self.cube_flux[:,center - self.npix_lenslet//2:center+self.npix_lenslet//2,
                         center - self.npix_lenslet//2:center+self.npix_lenslet//2] = input_flux_map
      
        # Get general properties of the illumination
        self.photon_per_subaperture = np.apply_over_axes(np.sum, self.cube_flux, [1,2])
        self.current_nPhoton = src.nPhoton
        return
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
        input_phase_torch = torch.from_numpy(phase).contiguous().permute(2,0,1)
        square_pupil_torch = torch.from_numpy(square_pupil).contiguous()

        phase_rescaled = torch.nn.functional.interpolate(input_phase_torch.unsqueeze(0), 
                                                            size=(self.nSubap*self.npix_phase, self.nSubap*self.npix_phase), 
                                                            mode='bilinear', align_corners=True).squeeze(0).contiguous()
        t1 = time.time()
        # Reshape the subapertures to a grid of subapertures. The sensor can be zeropadded, so the phase is filling the central part of the subaperture
        phase_reshaped = torch.empty((self.nSubap**2, input_phase_torch.shape[0], self.npix_phase+1, self.npix_phase+1), dtype=torch.float32).contiguous()

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
        cube_em = torch.zeros((self.nSubap**2, input_phase_torch.shape[0], nFFT, nFFT), dtype=torch.complex64)
        sub_mask = square_pupil_torch[row_start:row_end, col_start:col_end].float()
        # Exponential
        real = torch.cos(phase_rescaled_valids)
        imag = torch.sin(phase_rescaled_valids)
        exp_block = sub_mask * (real + 1j * imag)

        cube_em[self.valid_subapertures_1D,:, row_start:row_end, col_start:col_end] = exp_block
        # Apply light scaling
        # cube_em *= np.sqrt(cube_flux) * phasor_tiled
        t4 = time.time()
        # Get the PSF
        psf = torch.zeros((self.nSubap**2, input_phase_torch.shape[0], npix_sun, npix_sun), dtype=torch.float32)

        fft_res = torch.fft.fft2(cube_em[self.valid_subapertures_1D, :, :, :], dim=(-2, -1), norm='forward')  # same as dividing by nFFT²

        # Shift zero frequency to center
        fft_res = torch.fft.fftshift(fft_res, dim=(-2, -1))

        # Compute normalized intensity
        fft_res = torch.abs(fft_res) ** 2
        # Crop to desired region
        psf[self.valid_subapertures_1D, :, :, :] = fft_res[:, :, row_start:row_end, col_start:col_end]
        t5 = time.time()
        
        self.logger.debug(f'CorrelatingShackHartmann::get_psf - Time taken for each step: '
                         f'Rescale input phase: {t1-t0} [s], Reshape into subaps: {t2-t1} [s], Interpolate to npix_sun: {t3-t2} [s], '
                         f'Compute exponential: {t4-t3} [s], PSF: {t5-t4} [s], Total processing time: {t5-t0}')
        return psf
    
    def compute_images(self, psf, subDirs_sun, new_px):
        
        # Convert to Tensor
        sun_torch = torch.from_numpy(subDirs_sun).contiguous() # 4D: 
        sun_torch = sun_torch.view(sun_torch.shape[0], sun_torch.shape[1], -1).permute(2, 0, 1)

        # Dowsample the sun to match the WFS plate scale
        sun_torch = torch.nn.functional.interpolate(sun_torch.unsqueeze(0), size=(new_px, new_px), mode='bilinear', align_corners=True).contiguous()
        
        # Compute FFT of the sun subdirs

        sun_fft = torch.fft.fft2(sun_torch, norm='forward', dim=(-2, -1))

        # Compute FFT of the PSF

        psf_fft = torch.fft.fft2(psf[self.valid_subapertures_1D, :, :, :], norm='forward', dim=(-2, -1))

        # Convolute

        sun_patches = torch.abs(torch.fft.fftshift(torch.fft.ifft2(sun_fft*psf_fft, norm='forward', dim=(-2, -1)), dim=(-2, -1)))
 
        return sun_patches
    
    def merges_images(self, sun_patches, src):
        # Resize the 2D filter
        filter_2D_torch = torch.from_numpy(src.filter_2D).contiguous()
        filter_2D_torch = filter_2D_torch.view(filter_2D_torch.shape[0], filter_2D_torch.shape[1], -1).permute(2, 0, 1)
        new_size = np.round(src.subDirs_coordinates[2,0,0] / self.plate_scale).astype(int)
        filter_2D_torch = torch.nn.functional.interpolate(filter_2D_torch.unsqueeze(0), size=(new_size, new_size), 
                                                          mode='bilinear', align_corners=True).squeeze(0).contiguous()

        # Combine the sun patches into a unique PSF

        sun_PSF_combined = torch.zeros(sun_patches.shape[0], np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                             np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                                             dtype=torch.float32).contiguous()
        
        sun_psf_tmp_3D = torch.zeros(sun_patches.shape[0], src.nSubDirs*src.nSubDirs, 
                                     np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                     np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), dtype=torch.float32).contiguous()

        small_gain_corrector = torch.zeros(sun_patches.shape[0], src.nSubDirs*src.nSubDirs, 
                                np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), 
                                np.round((src.fov+src.patch_padding)/self.plate_scale).astype(int), dtype=torch.float32).contiguous()

        for i in range(sun_patches.shape[1]):
            dirX = i//src.nSubDirs
            dirY = i%src.nSubDirs
            
            gcorner_x = np.round(dirX*((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale))).astype(int)
            gcorner_y = np.round(dirY*((src.subDirs_coordinates[2,0,0])/(2*self.plate_scale))).astype(int)

            gcorner_x_end = gcorner_x + np.round((src.subDirs_coordinates[2,0,0])/(self.plate_scale)).astype(int)
            gcorner_y_end = gcorner_y + np.round((src.subDirs_coordinates[2,0,0])/(self.plate_scale)).astype(int)

            start = sun_patches.shape[-1]//2 - filter_2D_torch.shape[1]//2
            
            sun_psf_tmp_3D[:,i, gcorner_x:gcorner_x_end,
                                gcorner_y:gcorner_y_end] = filter_2D_torch[i,:,:] * sun_patches[:,i,start:start+filter_2D_torch.shape[1],
                                                                                                    start:start+filter_2D_torch.shape[1]]
            
            small_gain_corrector[:,i, gcorner_x:gcorner_x_end, gcorner_y:gcorner_y_end] = filter_2D_torch[i,:,:]
        
        # Crop the external region, which might be affected by the windowing effect.
        # The patch is larger so that the final crop matches the FoV of the object.
        sun_PSF_combined = torch.sum(sun_psf_tmp_3D,axis=1) / torch.sum(small_gain_corrector,axis=1)
        
        offset = sun_patches.shape[-1]//2 - self.npix_lenslet//2

        merged_image = sun_PSF_combined[:,offset:offset+self.npix_lenslet, offset:offset+self.npix_lenslet].detach().numpy()
        
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

        return torch.Tensor(subaps).contiguous()
        
    #%% GEOMETRIC    
         
    def gradient_2D(self,arr):
        """
        Compute X and Y gradients of a phase screen.

        Parameters
        ----------
        arr : np.ndarray
            2D array of phase values.

        Returns
        -------
        tuple
            Gradient in X and Y.
        """
        res_x = (np.gradient(arr,axis=0)/self.telescope.pixelSize)*self.telescope.pupil
        res_y = (np.gradient(arr,axis=1)/self.telescope.pixelSize)*self.telescope.pupil
        return res_x,res_y
        
    def lenslet_propagation_geometric(self,arr):
        """
        Compute geometric propagation through lenslets.

        Parameters
        ----------
        arr : np.ndarray
            Input phase screen.

        Returns
        -------
        np.ndarray
            Concatenated slope vector.
        """        
        [SLx,SLy]  = self.gradient_2D(arr)
        
        sy = (bin_ndarray(SLx, [self.nSubap,self.nSubap], operation='sum'))
        sx = (bin_ndarray(SLy, [self.nSubap,self.nSubap], operation='sum'))
        
        return np.concatenate((sx,sy))
            
    
#%% SH Measurement
    def wfs_integrate(self, ideal_frame, subaps):
        """
        Integrate the full detector image and compute valid slopes.

        Parameters
        ----------
        ideal_frame : np.ndarray
            Full-frame ideal image.
        subaps : np.ndarray
            Spot images per subaperture.

        Returns
        -------
        tuple
            Slopes 1D, slopes 2D, and final noisy image.
        """
        # propagate to detector to add noise and detector effects
        noisy_frame = self.cam.integrate(ideal_frame)
        subaps = self.get_subaps(noisy_frame)

        # Check if we have a pseudo reference 
        if self.pseudoref is None:
            self.pseudoref = torch.clone(subaps[0,:,:]).contiguous()

        # Compute correlation images
        correlation_images = self.cross_correlate(subaps, self.pseudoref)
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

        signal_2D                           = np.concatenate((sx, sy)) - self.reference_slopes_maps
        signal_2D[~self.valid_slopes_maps]  = 0
        
        signal_2D                      = signal_2D/self.slopes_units
        signal                         = signal_2D[self.valid_slopes_maps]
        
        return signal, signal_2D, noisy_frame
    
    # Receives the phase [rad]] and return the slopes measured by the SH in [px]
    # Expects the ideal frame WITHOUT pupil applied.
    def wfs_measure(self,phase_in, src):
        """
        Measure slopes from a wavefront phase using the SH-WFS.

        Parameters
        ----------
        phase_in : np.ndarray
            Phase map input [radians].
        src : Source
            Source object.
        integrate : bool, optional
            Whether to include camera integration effects.

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
        # Check if it is necessary to recompute the flux per subaperture, this is important as it will take longer
        # time during the execution and the number of subaps may vary!! --> Be careful, the IM shall vary accondingly
        if self.current_nPhoton != src.nPhoton:
            self.logger.info('CorrelatingShackHartmann::wfs_measure - Number of photons changed, updating flux on subaps')
            self.initialize_flux(src, self.norm_flux)   
        
        # compute fwhm
        fwhm = src.wavelength * 206265 / (self.subaperture_size * self.plate_scale)
        # Compute the new number of pixels for the sun patches to match the WFS plate scale
        new_px = np.round((src.subDirs_coordinates[2,0,0] + src.subDir_margin)/ self.plate_scale).astype(int) # Coordinate 2 has the width of the subDir
        # Check the dimensions of the phase
        if np.ndim(phase_in) != 3:
            raise ValueError('CorrelatingShackHartmann::wfs_measure - Phase must be a 3D array, one phase per subDir.')
        # compute the PSF intensity
        psf = self.get_psf(phase_in, fwhm, new_px)
    
        # Convolute with the sun patches - only valid subaps

        sun_patches = self.compute_images(psf, src.subDirs_sun, new_px)

        # Merge the patches into a single image per subap

        I = self.merges_images(sun_patches, src)

        # fill camera frame with computed intensity (only valid subapertures)

        ideal_frame = self.create_full_frame(I)

        signal, signal_2D, noisy_frame = self.wfs_integrate(ideal_frame, I)                
        
        return signal, signal_2D, noisy_frame

    def print_properties(self):
        """
        Print Shack-Hartmann configuration and diagnostic values.

        Returns
        -------
        None
        """
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHACK HARTMANN WFS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('{: ^20s}'.format('Subapertures')         + '{: ^18s}'.format(str(self.nSubap))                                   )
        self.logger.info('{: ^20s}'.format('Subaperture Size')     + '{: ^18s}'.format(str(np.round(self.subaperture_size, 2)))         +'{: ^18s}'.format('[m]'     ))
        self.logger.info('{: ^20s}'.format('Subapertue FoV')       + '{: ^18s}'.format(str(np.round(self.fieldOfView,2)))        +'{: ^18s}'.format('[arcsec]'))
        self.logger.info('{: ^20s}'.format('Valid Subaperture')    + '{: ^18s}'.format(str(str(self.nValidSubaperture))))                           
      
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
