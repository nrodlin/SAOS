# -*- coding: utf-8 -*-
"""
Created on Thu May 8 2025

@author: nlinares
"""

import time

import logging

import numpy as np

import torch

import cv2

from .Detector import Detector
from .tools.tools import bin_ndarray
from .LoggingHelper import get_logger

"""
Shack Hartmann Wavefront Sensor Module
=================

This module contains the `CorrelatingShackHartmann` class, used for modeling a SH-WFS in adaptive optics simulations.
"""

class CorrelatingShackHartmann:
    # Keyword arguments read from **kwargs (any other one is reported as unknown)
    KNOWN_KWARGS = ('device', 'pupil_oversampling', 'n_reference_frames', 'validSubapThreshold', 'fullWellCapacity', 'nBits', 'quantumEfficiency',
                    'shotNoise', 'darkCurrent', 'readoutNoise', 'gain', 'quantization_conversion', 'sensorType', 'darkCalibration',
                    'randomState', 'integrationTime')

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
            Fraction of the source photons that reaches the WFS (photon budget). The valid subapertures are selected with
            validSubapThreshold.
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

            n_reference_frames : int, optional
                Detector frames averaged to acquire the pseudo-reference image and then the reference slopes when noiseFlag is
                True, as in operation (residual noise = single-frame noise / sqrt(n)). Default 100. Without noise one frame is used.
            pupil_oversampling : int, optional
                Pupil samples per subaperture used to form the spots, as a multiple k of the sun patch size. Default (None):
                the smallest number that keeps the bias of the spot second moment, ~1.22 (u / n)^2 with u the half-width of the
                sun patch in lambda/d, below 1%. The cost of the spot computation grows roughly as n^2.

            validSubapThreshold : float, optional
                Minimum fraction of the subaperture area inside the pupil to be a valid subaperture. Default 0.5. The partially
                illuminated subapertures are imaged through their illuminated part only (the field amplitude is the pupil), so the
                phase outside the pupil is never used.
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
        self.logger = logger if logger is not None else get_logger()
        unknown = sorted(set(kwargs) - set(self.KNOWN_KWARGS))
        if unknown:
            self.logger.warning(f'CorrelatingShackHartmann::__init__ - unknown keyword argument(s) {unknown} ignored (accepted: {", ".join(self.KNOWN_KWARGS)}).')

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
        self.n_reference_frames             = int(kwargs.get('n_reference_frames', 100))

        self.device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
       
        # Subapeture definition
        self.subaperture_size           = telescope.D / self.nSubap
        self.npix_lenslet               = int(np.round((self.fieldOfView + self.fft_fieldOfView_oversampling) / self.plate_scale))
        self.npix_subap                 = int(np.round(self.fieldOfView / self.plate_scale))
        if not (1 <= int(self.use_brightest) <= self.npix_subap ** 2):
            self.logger.error(f'{type(self).__name__}::__init__ - use_brightest = {self.use_brightest} outside [1, {self.npix_subap ** 2}] (pixels of the window).')
            raise ValueError(f'use_brightest must be in [1, {self.npix_subap ** 2}] (pixels of the {self.npix_subap}x{self.npix_subap} window), got {self.use_brightest}.')
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

        self.logger.info('CorrelatingShackHartmann::__init__ - Selecting valid subapertures based on flux considerations..')

        # Fraction of each subaperture inside the pupil: valid if it reaches validSubapThreshold
        self.validSubapThreshold = kwargs.get('validSubapThreshold', 0.5)
        self.subaperture_illumination = self.pupil_interpolation_mask.reshape(self.nSubap, self.npix_phase, self.nSubap, self.npix_phase).mean(axis=(1, 3))
        self.valid_subapertures = self.subaperture_illumination >= self.validSubapThreshold

        self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])

        [self.validLenslets_x , self.validLenslets_y] = np.where(self.valid_subapertures==True)
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
        self.nSignal = 2*self.nValidSubaperture     

        self.geometric_references  = np.zeros([self.nSubap*2,self.nSubap])

        # Field amplitude of the valid subapertures: the pupil sampled as the phase in get_psf (same interpolations), so a partially
        # illuminated subaperture is imaged through its illuminated part only. It also gives the normalized interpolation of the phase,
        # interp(pupil * phase) / interp(pupil), which keeps the phase of the edge points free of the zeros outside the pupil.
        # Energy of each subaperture relative to a fully illuminated one: sum of |amplitude|^2
        self.npix_sun = self.get_npix_sun(src)
        self.pupil_torch = torch.from_numpy(telescope.pupil.astype(float)).unsqueeze(0).contiguous().to(self.device)
        # Pupil samples per subaperture for the image formation (decoupled from the detector pixels by the MFT, see
        # ShackHartmann): smallest number keeping the bias of the spot second moment over the sun patch <= 1%, or
        # pupil_oversampling * npix_sun; rounded up to a multiple of the pupil pixels per subaperture
        self.pupil_oversampling = kwargs.get('pupil_oversampling', None)
        if self.pupil_oversampling is None:
            fwhm_px = src.wavelength * 206265 / (self.subaperture_size * self.plate_scale)
            n_min = np.sqrt(1.22 / 0.01) * (self.npix_sun / 2) / fwhm_px
        else:
            n_min = int(self.pupil_oversampling) * self.npix_sun
        self.npix_pupil = int(self.npix_phase * np.ceil(max(n_min, self.npix_phase) / self.npix_phase))
        self.valid_index = torch.from_numpy(np.flatnonzero(self.valid_subapertures_1D)).to(self.device)
        self.subap_resampling = self.cell_centred_resampling(telescope.resolution, self.nSubap * self.npix_pupil)
        self.local_resampling = None
        if telescope.resolution == self.nSubap * self.npix_phase:
            npp = self.npix_phase
            u = (np.arange(self.npix_pupil) + 0.5) * npp / self.npix_pupil - 0.5 + 1      # patch index units (margin of 1)
            m0 = np.floor(u).astype(int)
            w1 = u - m0
            Wl = np.zeros((self.npix_pupil, npp + 2))
            Wl[np.arange(self.npix_pupil), m0] = 1 - w1
            Wl[np.arange(self.npix_pupil), m0 + 1] = w1
            self.local_resampling = torch.from_numpy(Wl).float().to(self.device)
        pupil_sub = self.split_subapertures(self.pupil_torch, self.valid_index).float()          # [nValid, 1, n, n]
        # Amplitude = sqrt(covered fraction of each fine cell): the light of a partially illuminated subaperture equals its
        # illuminated area and varies continuously with the pupil geometry
        coverage = self.cell_coverage(telescope.resolution, self.nSubap * self.npix_pupil)
        covered = coverage @ self.pupil_torch[0].to(coverage.dtype) @ coverage.transpose(0, 1)
        n = self.npix_pupil
        covered = covered.view(self.nSubap, n, self.nSubap, n).permute(0, 2, 1, 3).reshape(self.nSubap * self.nSubap, 1, n, n)
        self.subap_amplitude = torch.sqrt(covered[self.valid_index].clamp(0.0, 1.0)).contiguous().float()
        self.subap_amplitude_inv = torch.where(pupil_sub > 1e-6, 1 / pupil_sub.clamp_min(1e-12), 0.0)
        self.subap_energy = torch.mean(self.subap_amplitude**2, dim=(-2, -1), keepdim=True)
        self._mft_cache = {}
        # Pseudo-reference of the correlations: the first fully illuminated subaperture
        self.pseudoref_index = int(np.argmax(self.subaperture_illumination[self.valid_subapertures] == 1))

        # Detector pixels of every valid subaperture image (create_full_frame, get_subaps)
        center_offset = self.guardPx + (self.npix_lenslet // 2) - self.npix_subap//2
        init_x = center_offset + self.validLenslets_x * (self.guardPx + self.npix_lenslet)
        init_y = center_offset + self.validLenslets_y * (self.guardPx + self.npix_lenslet)
        pix = np.arange(self.npix_subap)
        self.subap_rows = (init_x[:, None] + pix[None, :])[:, :, None]
        self.subap_cols = (init_y[:, None] + pix[None, :])[:, None, :]

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
            # Tilts are measured against the pseudo-reference and reference slopes of the unaberrated object (a pseudo-reference
            # taken from each tilted frame would move with it and the slopes would be zero)
            pseudoref, reference_slopes = self.initialize_wfs(telescope, src)

            mean_slope = np.zeros(5)
            input_std = np.zeros(5)

            for i in range(len(amp_list)):
                calibration_phase_tip = amp_list[i]*Tip*telescope.pupil
                calibration_phase_tip = np.repeat(calibration_phase_tip[np.newaxis,:,:], src.nSubDirs**2, axis=0)
                signalTip,_,_,_ = self.wfs_measure(calibration_phase_tip, src, pseudoref, reference_slopes)

                calibration_phase_tilt = amp_list[i]*Tilt*telescope.pupil
                calibration_phase_tilt = np.repeat(calibration_phase_tilt[np.newaxis,:,:], src.nSubDirs**2, axis=0)
                signalTilt,_,_,_ = self.wfs_measure(calibration_phase_tilt, src, pseudoref, reference_slopes)

                mean_slope[i] = np.mean(signalTip[:self.nValidSubaperture] + signalTilt[:self.nValidSubaperture])
                input_std[i] = np.std(amp_list[i]*Tip[telescope.pupil.astype(bool)])
                
            p = np.polyfit(amp_list*np.asarray(mode_amp), mean_slope, deg = 1)
            self.slopes_units = np.abs(p[0]) # [px/rad]
        
        
    def initialize_wfs(self, telescope, src):
        """
        Acquire the pseudo-reference image and the reference slopes at null phase. With detector noise, as in operation, the
        pseudo-reference is the average of n_reference_frames images of subaperture pseudoref_index and the reference slopes
        the average of n_reference_frames further frames correlated with it.

        Parameters
        ----------
        telescope : Telescope
            The telescope object providing phase and pupil info.
        src : Source
            Source object for flux and wavelength reference.

        Returns
        -------
        tuple
            Pseudo-reference image (torch.Tensor) and reference slopes map [2 nSubap, nSubap].
        """
        self.logger.info('CorrelatingShackHartmann::initialize_wfs - Acquiring reference slopes..')
        # As in operation: the pseudo-reference is the average of n images of subaperture pseudoref_index, then the reference
        # slopes are the average of n further frames correlated with it (the optical propagation is computed once, the noise
        # is drawn per frame)
        null_phase = np.zeros((src.nSubDirs**2, telescope.resolution, telescope.resolution))
        ideal_frame, I, nPhotons = self.ideal_frame(null_phase, src)
        n_frames = self.n_reference_frames if self.cam.noiseFlag else 1
        pseudoref = 0
        for _ in range(n_frames):
            pseudoref = pseudoref + self.get_subaps(self.cam.integrate(ideal_frame, nPhotons))[self.pseudoref_index].double()
        pseudoref = (pseudoref / n_frames).float().contiguous().to(self.device)
        reference_slopes_maps = 0
        for _ in range(n_frames):
            reference_slopes_maps = reference_slopes_maps + self.wfs_integrate(ideal_frame, I, nPhotons, pseudoref)[1]
        reference_slopes_maps = reference_slopes_maps / n_frames
                
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

    def get_npix_sun(self, src):
        """
        Number of pixels per subaperture of the sun patches, matching the WFS plate scale.

        Parameters
        ----------
        src : Source
            Solar source object.

        Returns
        -------
        int
            Pixels of the sun patches per subaperture [px].
        """
        return np.round((src.subDirs_coordinates[2,0,0] + src.subDir_margin)/ self.plate_scale).astype(int) # Coordinate 2 has the width of the subDir

    def cell_centred_resampling(self, n_in, n_out):
        """
        Linear resampling (indices and weights) between two cell-centred grids covering the same length (see
        ShackHartmann.cell_centred_resampling): output sample i = (1 - w1[i]) input[m0[i]] + w1[i] input[m0[i] + 1], with
        linear extrapolation in the outermost half input cell.
        """
        u = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        m0 = np.clip(np.floor(u).astype(int), 0, n_in - 2)
        w1 = u - m0
        return (torch.from_numpy(m0).to(self.device), torch.from_numpy(w1).to(self.device))

    def cell_coverage(self, n_in, n_out):
        """Area-weighting matrix [n_out, n_in]: fraction of output cell i covered by input cell m (see ShackHartmann)."""
        edges_in = np.arange(n_in + 1) / n_in
        edges_out = np.arange(n_out + 1) / n_out
        lo = np.maximum(edges_out[:-1, None], edges_in[None, :-1])
        hi = np.minimum(edges_out[1:, None], edges_in[None, 1:])
        return torch.from_numpy(np.clip(hi - lo, 0, None) * n_out).to(self.device)

    def split_subapertures(self, field, index=None):
        """
        Resample pupil-plane maps to npix_pupil cell-centred samples per subaperture and split them into subapertures.

        The subaperture of width d is sampled at the centres of npix_pupil cells, (j + 1/2) d / npix_pupil from its edge, by
        linear interpolation of the pupil pixels (cells as well); a linear phase is reproduced exactly (full slope across the
        subaperture, no shift of its centre).

        Parameters
        ----------
        field : torch.Tensor
            Pupil-plane maps [C, resolution, resolution] (phase per subdirection, or the pupil).
        index : torch.Tensor, optional
            Subapertures to return (default: all).

        Returns
        -------
        torch.Tensor
            Subaperture maps [nSubap**2 (or len(index)), C, npix_pupil, npix_pupil], sorted from left to right, top to bottom.
        """
        n = self.npix_pupil
        f = field.float()
        C = f.shape[0]
        if self.local_resampling is not None:
            # Pupil patch of every subaperture with a one-pixel margin (linear extrapolation outside the array)
            f = torch.cat([2 * f[:, :1] - f[:, 1:2], f, 2 * f[:, -1:] - f[:, -2:-1]], dim=1)
            f = torch.cat([2 * f[:, :, :1] - f[:, :, 1:2], f, 2 * f[:, :, -1:] - f[:, :, -2:-1]], dim=2)
            npp = self.npix_phase
            patches = f.unfold(1, npp + 2, npp).unfold(2, npp + 2, npp)            # [C, nSubap, nSubap, npp+2, npp+2]
            patches = patches.permute(1, 2, 0, 3, 4).reshape(self.nSubap * self.nSubap, C, npp + 2, npp + 2)
            if index is not None:
                patches = patches[index]
            W = self.local_resampling
            return W @ patches @ W.transpose(0, 1)
        m0, w1 = self.subap_resampling
        f = f.to(w1.dtype)
        f = f[:, m0] * (1 - w1)[None, :, None] + f[:, m0 + 1] * w1[None, :, None]
        f = f[:, :, m0] * (1 - w1)[None, None, :] + f[:, :, m0 + 1] * w1[None, None, :]
        out = f.view(C, self.nSubap, n, self.nSubap, n).permute(1, 3, 0, 2, 4).reshape(self.nSubap ** 2, C, n, n).float()
        return out if index is None else out[index]

    def _mft_kernel(self, fwhm, npix_sun):
        """
        MFT matrix A[k, j] = exp(-2 pi i (j - (n_p-1)/2) (k - npix_sun//2) / (fwhm n_p)): field of a subaperture sampled at
        n_p cell-centred points -> image pixel k at the angle (k - npix_sun//2) * plate_scale. Cached per (fwhm, npix_sun).
        """
        key = (round(float(fwhm), 12), int(npix_sun))
        A = self._mft_cache.get(key)
        if A is None:
            n_p = self.npix_pupil
            j = torch.arange(n_p, dtype=torch.float64, device=self.device) - (n_p - 1) / 2
            k = torch.arange(npix_sun, dtype=torch.float64, device=self.device) - npix_sun // 2
            A = torch.exp(-2j * np.pi * torch.outer(k, j) / (fwhm * n_p)).to(torch.complex64)
            self._mft_cache[key] = A
        return A

    def get_psf(self, phase, fwhm, npix_sun):
        """
        Spot of every valid subaperture and subdirection from the pupil phase, with a matrix Fourier transform (MFT) at the
        WFS plate scale, on the npix_sun x npix_sun grid of the sun patches.

        The field of each subaperture is sqrt(covered fraction) times exp(i phase), sampled at npix_pupil cell-centred points
        (split_subapertures); the phase uses the normalised interpolation interp(pupil * phase) / interp(pupil). Image pixel
        k is at the angle (k - npix_sun//2) * plate_scale exactly, so that the PSF is centred at pixel npix_sun//2 (the
        origin used by compute_images). Each PSF is normalised to the energy of its subaperture (1 if fully illuminated).

        Parameters
        ----------
        phase : np.ndarray
            Wavefront phase in radians per subdirection [nSubDirs**2, resolution, resolution].
        fwhm : float
            lambda / d in WFS pixels.
        npix_sun : int
            Number of pixels of the sun patches [px].

        Returns
        -------
        torch.Tensor
            PSF per subaperture and subdirection [nSubap**2, nSubDirs**2, npix_sun, npix_sun] (zero for invalid subapertures).
        """
        self.logger.debug('CorrelatingShackHartmann::get_psf')
        if fwhm < 1:
            raise ValueError('FWHM must be greater than 2 to guarantee enough spatial sampling.')
        if npix_sun != self.npix_sun:
            self.logger.error(f'CorrelatingShackHartmann::get_psf - The source gives {npix_sun} px per subaperture, the WFS was built for {self.npix_sun} px.')
            raise ValueError('CorrelatingShackHartmann::get_psf - The source does not match the one used to build the WFS.')
        t0 = time.time()
        input_phase_torch = torch.from_numpy(phase).contiguous().to(self.device) * self.pupil_torch   # no phase outside the pupil
        phase_sub = self.split_subapertures(input_phase_torch, self.valid_index) * self.subap_amplitude_inv   # [nValid, C, n, n]
        t1 = time.time()
        A = self._mft_kernel(fwhm, npix_sun)
        At = A.transpose(0, 1)
        n_valids = phase_sub.shape[0]
        psf = torch.zeros((self.nSubap**2, input_phase_torch.shape[0], npix_sun, npix_sun), dtype=torch.float32, device=self.device)
        chunk_size = 512
        for start_idx in range(0, n_valids, chunk_size):
            end_idx = min(start_idx + chunk_size, n_valids)
            field = torch.polar(self.subap_amplitude[start_idx:end_idx].expand_as(phase_sub[start_idx:end_idx]), phase_sub[start_idx:end_idx])
            E = A @ field @ At
            intensity = E.real.square().add_(E.imag.square())
            del E, field
            # Energy of the subaperture: 1 if fully illuminated, its fraction of |amplitude|^2 otherwise
            norma = torch.sum(intensity, dim=(-2, -1), keepdim=True)
            psf[self.valid_index[start_idx:end_idx]] = intensity / norma * self.subap_energy[start_idx:end_idx]
            del intensity
        t2 = time.time()
        self.logger.debug(f'CorrelatingShackHartmann::get_psf - Time taken: sample subapertures {t1-t0} [s], MFT {t2-t1} [s]')
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

        # The PSF centre (pixel npix_sun//2, see get_psf) is moved to the origin, so that the circular convolution does not shift
        # the sun patch, for odd and even npix_sun
        psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf[self.valid_subapertures_1D, :, :, :], dim=(-2, -1)), norm='backward', dim=(-2, -1))

        # Convolute
        sun_patches_complex = torch.fft.ifft2(sun_fft*psf_fft, norm='forward', dim=(-2, -1))

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

        subap_offset = (self.npix_lenslet - self.npix_subap) // 2
        ideal_frame[self.subap_rows, self.subap_cols] = subaps[:, subap_offset:subap_offset+self.npix_subap, subap_offset:subap_offset+self.npix_subap]

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

        subaps = np.asarray(noisy_frame, dtype=float)[self.subap_rows, self.subap_cols]

        return torch.Tensor(subaps).contiguous().to(self.device)
            
#%% SH Measurement
    def wfs_integrate(self, ideal_frame, subaps, nPhoton, pseudoref=None, reference_slopes=None, stream=None):
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
            Subaperture used as pseudo-reference for the correlations, if None the first fully illuminated subaperture is picked
        stream : int, optional
            Noise stream of the light path (Detector.register_stream); None uses the detector's own generator.
        Returns
        -------
        tuple
            Slopes 1D, slopes 2D, and final noisy image.
        """
        # propagate to detector to add noise and detector effects
        noisy_frame = self.cam.integrate(ideal_frame, nPhoton, stream)
        subaps = self.get_subaps(noisy_frame)

        # Check if we have a pseudo reference 
        if pseudoref is None:
            pseudoref = torch.clone(subaps[self.pseudoref_index,:,:]).contiguous().to(self.device)

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
    def wfs_measure(self,phase_in, src, pseudoref=None, reference_slopes=None, stream=None):
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
        stream : int, optional
            Noise stream of the light path (Detector.register_stream); None uses the detector's own generator.

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
        # If the phase is not 3D, we need to repeat it for each subDir --> typically when the phase uses the DM only
        if np.ndim(phase_in) < 3:
            phase_in = np.repeat(phase_in[np.newaxis,:,:], src.nSubDirs**2, axis=0)
        t0 = time.time()
        ideal_frame, I, nPhotons = self.ideal_frame(phase_in, src)
        t1 = time.time()
        # Compute the frame and the slopes
        signal, signal_2D, noisy_frame, pseudoref = self.wfs_integrate(ideal_frame, I, nPhotons, pseudoref, reference_slopes, stream)
        t2 = time.time()
        self.logger.debug(f'Ideal frame: {t1-t0}, Integrate:{t2-t1}')
        return signal, signal_2D, noisy_frame, pseudoref

    def ideal_frame(self, phase_in, src):
        """
        Noise-free detector frame of a wavefront phase (the deterministic part of wfs_measure).

        Parameters
        ----------
        phase_in : np.ndarray
            Phase map input [radians], one per sub-direction of the source.
        src : Source
            Source object.

        Returns
        -------
        tuple
            Ideal full frame, merged images of the valid subapertures and number of photons.
        """
        # compute fwhm
        fwhm = src.wavelength * 206265 / (self.subaperture_size * self.plate_scale)
        # Compute the new number of pixels for the sun patches to match the WFS plate scale
        new_px = self.get_npix_sun(src)
        # compute the PSF intensity
        psf = self.get_psf(phase_in, fwhm, new_px)
        # Convolute with the sun patches - only valid subaps

        sun_patches = self.compute_images(psf, src.subDirs_sun, new_px)
        # Merge the patches into a single image per subap

        I = self.merges_images(sun_patches, src)
        # fill camera frame with computed intensity (only valid subapertures)

        ideal_frame = self.create_full_frame(I)
        # Compute the number of photons considering the flux of the sun, collecting area of the telescope, sampling time and FoV of the WFS
        # that may be smaller in the simulation than the FoV of the source: the flux is per field, so it scales with the sky area (FoV^2)
        nPhotons = np.round(src.flux * self.area_eff * self.integration_time * self.lightRatio * (self.fieldOfView / src.fov)**2).astype(int)
        return ideal_frame, I, nPhotons
              
      
    
