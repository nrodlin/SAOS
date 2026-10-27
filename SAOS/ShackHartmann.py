# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie

Major update on March 24 2025
@author: nrodlin
"""

import time

import logging

import numpy as np

import torch

import cv2

from .Detector import Detector
from .LoggingHelper import get_logger


"""
Shack Hartmann Wavefront Sensor Module
=================

This module contains the `ShackHartmann` class, used for modeling a SH-WFS in adaptive optics simulations.
"""

class ShackHartmann:
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
                 noiseFlag:bool=False,
                 logger=None,
                 **kwargs):
        
        """
        Initialize a Shack-Hartmann Wavefront Sensor (WFS).

        Parameters
        ----------
        nSubap : float
            Number of subapertures across the pupil diameter.
        telescope : Telescope
            Telescope object to which the WFS is attached.
        src : Source
            Source object (NGS or LGS).
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
        pupil_oversampling : int, optional (keyword)
            Pupil samples per subaperture used to form the spots, as a multiple k of the lenslet image size. Default
            (None): the smallest number that keeps the bias of the spot second moment (CoG noise), ~1.22 (u / n)^2 with u
            the window half-width in lambda/d, below 1% (EST red: 48 samples). The cost of the spot computation grows
            roughly as n^2.
        use_brightest : int, optional
            Picks the n brightest pixels as threshold for center-of-gravity spot detection.
        n_reference_frames : int, optional (keyword)
            Detector frames averaged to acquire the reference slopes when noiseFlag is True, as in operation (the residual
            noise of the reference is the single-frame noise / sqrt(n)). Default 100. Without noise one frame is used.
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
                Detector parameter. Conversion gain to discretize the measurement [e-/px]. Default 0 --> adjust by well capacity.
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

        self.tag                            = 'shackHartmann'

        # Check the source type, just in case:

        if src.tag != 'source':
            self.logger.error(f'ShackHartmann::init - Source type is {src.tag}, which is not supported. This module expects the type to be SOURCE')
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
        # Pupil samples per subaperture for the image formation (decoupled from the detector pixels by the MFT). A field sampled
        # at n points gives a spot that repeats every n lambda/d, which raises its wings: the second moment of the spot (CoG
        # noise) is biased by ~1.22 (u / n)^2, u = half-width of the detector window in lambda/d (calibrated against a
        # converged Fourier sum). By default n is the smallest value that keeps that bias <= 1%; pupil_oversampling = k forces
        # n = k * npix_lenslet. In both cases n is rounded up to a multiple of the pupil pixels per subaperture.
        self.pupil_oversampling = kwargs.get('pupil_oversampling', None)
       
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
        self.camera_params['quantization_conversion'] = kwargs.get('quantization_conversion', 0)
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

        # Photon count
        self.area_eff         = telescope.area_eff
        self.integration_time = telescope.samplingTime
                
        self.logger.info('ShackHartmann::__init__ - Selecting valid subapertures based on flux considerations..')

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

        # Field amplitude of every subaperture: the pupil sampled as the phase in get_psf (same interpolations), so a partially
        # illuminated subaperture is imaged through its illuminated part only, with its fraction of the energy (sum of |amplitude|^2).
        # It also gives the normalized interpolation of the phase, interp(pupil * phase) / interp(pupil), which keeps the phase of the
        # edge points free of the zeros outside the pupil
        self.pupil_torch = torch.from_numpy(telescope.pupil.astype(float)).contiguous().to(self.device)
        # Cell-centred resampling matrix (rows and columns): npix_lenslet samples per subaperture (see split_subapertures)
        # Multiple of the native pupil pixels per subaperture, so that every fine cell lies inside one pupil pixel and the pupil
        # edge is reproduced exactly (no fractional cells straddling native pixel boundaries)
        if self.pupil_oversampling is None:
            fwhm_px = src.wavelength * 206265 / (self.subaperture_size * self.plate_scale)
            n_min = np.sqrt(1.22 / 0.01) * (self.npix_subap / 2) / fwhm_px
        else:
            n_min = int(self.pupil_oversampling) * self.npix_lenslet
        self.npix_pupil = int(self.npix_phase * np.ceil(max(n_min, self.npix_phase) / self.npix_phase))
        self.subap_resampling = self.cell_centred_resampling(telescope.resolution, self.nSubap * self.npix_pupil)
        # Fast path when every subaperture covers exactly npix_phase pupil pixels: the same small interpolation matrix
        # (npix_pupil x (npix_phase + 2)) maps the pupil patch of any subaperture (with a one-pixel margin) to its samples
        self.local_resampling = None
        if telescope.resolution == self.nSubap * self.npix_phase:
            npp = self.npix_phase
            u = (np.arange(self.npix_pupil) + 0.5) * npp / self.npix_pupil - 0.5 + 1      # patch index units (margin of 1)
            m0 = np.floor(u).astype(int)
            w1 = u - m0
            W = np.zeros((self.npix_pupil, npp + 2))
            W[np.arange(self.npix_pupil), m0] = 1 - w1
            W[np.arange(self.npix_pupil), m0 + 1] = w1
            self.local_resampling = torch.from_numpy(W).float().to(self.device)
        self.valid_index = torch.from_numpy(np.flatnonzero(self.valid_subapertures_1D)).to(self.device)
        pupil_sub = self.split_subapertures(self.pupil_torch).contiguous().float()
        # Amplitude = sqrt(covered fraction): the interpolated mask is the fraction of each fine sample covered by the pupil, so the
        # transmitted light sum |amplitude|^2 equals the illuminated area exactly and varies continuously with the pupil geometry
        coverage = self.cell_coverage(telescope.resolution, self.nSubap * self.npix_pupil)
        covered = (coverage @ self.pupil_torch.to(coverage.dtype) @ coverage.transpose(0, 1))
        n = self.npix_pupil
        covered = covered.view(self.nSubap, n, self.nSubap, n).permute(0, 2, 1, 3).reshape(self.nSubap * self.nSubap, n, n)
        self.subap_amplitude = torch.sqrt(covered.clamp(0.0, 1.0)).contiguous().float()
        self.subap_amplitude_inv = torch.where(pupil_sub > 1e-6, 1 / pupil_sub.clamp_min(1e-12), 0.0)
        # Only the valid subapertures are imaged (get_psf)
        self.subap_amplitude_valid = self.subap_amplitude[self.valid_index].contiguous()
        self.subap_amplitude_inv_valid = self.subap_amplitude_inv[self.valid_index].contiguous()
        self._mft_cache = {}       # MFT kernels per (lambda/d in px, spot centre)

        # Detector pixels of every valid subaperture image (create_full_frame, get_subaps)
        center_offset = self.guardPx + (self.npix_lenslet // 2) - self.npix_subap//2
        init_x = center_offset + self.validLenslets_x * (self.guardPx + self.npix_lenslet)
        init_y = center_offset + self.validLenslets_y * (self.guardPx + self.npix_lenslet)
        pix = np.arange(self.npix_subap)
        self.subap_rows = (init_x[:, None] + pix[None, :])[:, :, None]
        self.subap_cols = (init_y[:, None] + pix[None, :])[:, None, :]

        # LGS spot   
        if src.type == 'LGS':
            self.is_LGS                 = True
        else:
            self.is_LGS                 = False
        
        if self.is_LGS:
            self.get_convolution_spot(src) 

        # WFS initialization
        self.initialize_wfs(telescope, src)       
        
    def initialize_wfs(self, telescope, src):
        """
        Initialize the Shack-Hartmann WFS by measuring reference slopes
        and determining slope units. With detector noise the reference slopes are the average of n_reference_frames
        frames of the unaberrated spots (residual noise = single-frame noise / sqrt(n)), as in operation.

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
        
        # flux per subaperture
        self.reference_slopes_maps  = np.zeros([self.nSubap*2,self.nSubap])
        self.slopes_units           = 1

        self.logger.info('ShackHartmann::initialize_wfs - Acquiring reference slopes..')
        # Average of n detector frames of the unaberrated spots (the optical propagation is computed once, the noise is drawn per frame)
        null_phase = np.zeros((telescope.resolution, telescope.resolution))
        ideal_frame, I, nPhotons = self.ideal_frame(null_phase, src)
        n_frames = self.n_reference_frames if self.cam.noiseFlag else 1
        reference = np.zeros_like(self.reference_slopes_maps)
        for _ in range(n_frames):
            reference += self.wfs_integrate(ideal_frame, I, nPhotons)[1]
        self.reference_slopes_maps = reference / n_frames
        self.isInitialized = True
        
        self.logger.info('ShackHartmann::initialize_wfs - Setting slopes units..')  
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
                signalTip,_,_ = self.wfs_measure(calibration_phase_tip, src)

                calibration_phase_tilt = amp_list[i]*Tilt*telescope.pupil
                signalTilt,_,_ = self.wfs_measure(calibration_phase_tilt, src)

                mean_slope[i] = np.mean(signalTip[:self.nValidSubaperture] + signalTilt[:self.nValidSubaperture])
                input_std[i] = np.std(calibration_phase_tip[telescope.pupil])
                
            p = np.polyfit(amp_list*np.asarray(mode_amp), mean_slope, deg = 1)
            self.slopes_units = np.abs(p[0]) # [px/rad]
        
        self.logger.info('ShackHartmann::initialize_wfs - Done!') 
        
        self.print_properties()

    def centroid(self, image, use_brightest=50):
        """
        Compute center of gravity for subaperture spots.

        Parameters
        ----------
        image : np.ndarray
            Subaperture image cube.
        use_brightest : float, optional
            Minimum intensity to include in centroid.

        Returns
        -------
        np.ndarray
            X and Y centroids per subaperture.
        """
        image = np.atleast_3d(image.copy())

        # Convert to torch Tensor:
        image = torch.Tensor(image).to(self.device)

        n_subap, H, W = image.shape

        # flatten each subap
        flat = image.view(n_subap, -1)

        # Obtener umbral para los brightest
        topk_vals, _ = torch.topk(flat, use_brightest, dim=1)
        thresholds = topk_vals[:, -1].unsqueeze(1)

        # Máscara para pixels válidos
        mask = flat >= thresholds

        # Aplicar máscara
        filtered = flat * mask

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
#%% DIFFRACTIVE

    # This function takes the phase at the pupil as input. Then, the flux at each subaperture (pupil function) is multiplied 
    # by the complex phase to obtain the PSF per subaperture, as an array of dimensions (nSubap**2, n_pix_lenslet_init, n_pix_lenslet_init)
    # The subapertures are sorted from left to right, top to bottom.

    def cell_centred_resampling(self, n_in, n_out):
        """
        Linear resampling (indices and weights) between two cell-centred grids covering the same length: input cell m
        has its centre at (m + 1/2) / n_in, output cell i at (i + 1/2) / n_out (both in units of the full length).
        The output samples are linear interpolations of the two nearest input centres; in the outermost half input cell
        (beyond the first or last input centre) the line through the two edge samples is extrapolated, so that a linear
        phase is reproduced exactly everywhere.

        Parameters
        ----------
        n_in : int
            Number of input cells (pupil-plane resolution).
        n_out : int
            Number of output cells (nSubap * npix_lenslet).

        Returns
        -------
        tuple of torch.Tensor
            (m0, w1): output sample i = (1 - w1[i]) * input[m0[i]] + w1[i] * input[m0[i] + 1] (on the WFS device).
        """
        u = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5          # output centres in input index units
        m0 = np.clip(np.floor(u).astype(int), 0, n_in - 2)
        w1 = u - m0                                                 # < 0 or > 1 outside the input centres: extrapolation
        return (torch.from_numpy(m0).to(self.device), torch.from_numpy(w1).to(self.device))

    def cell_coverage(self, n_in, n_out):
        """
        Area-weighting matrix C [n_out, n_in] between two cell-centred grids covering the same length: C[i, m] is the
        fraction of output cell i covered by input cell m (rows sum to 1). For a map that is constant on the input cells
        (the pupil mask), C map C^T gives the exact mean over every output cell (its covered fraction), whether the output
        grid is finer or coarser than the input one, and the total is conserved inside every subaperture.

        Parameters
        ----------
        n_in : int
            Number of input cells.
        n_out : int
            Number of output cells.

        Returns
        -------
        torch.Tensor
            Coverage matrix [n_out, n_in] (float64, on the WFS device).
        """
        edges_in = np.arange(n_in + 1) / n_in
        edges_out = np.arange(n_out + 1) / n_out
        lo = np.maximum(edges_out[:-1, None], edges_in[None, :-1])
        hi = np.minimum(edges_out[1:, None], edges_in[None, 1:])
        C = np.clip(hi - lo, 0, None) * n_out
        return torch.from_numpy(C).to(self.device)

    def split_subapertures(self, field, index=None):
        """
        Resample a pupil-plane map to npix_lenslet cell-centred samples per subaperture and split it into subapertures.

        The pupil-plane pixels are cells of side D / resolution and the subaperture of width d = D / nSubap is sampled at
        the centres of npix_lenslet cells, (j + 1/2) d / npix_lenslet from its edge (cell_centred_resampling). The samples
        therefore span exactly the subaperture, with the spacing assumed by the Fourier transform (d / npix_lenslet), and a
        linear phase is reproduced exactly (full slope across the subaperture, no shift of the subaperture centre).

        Parameters
        ----------
        field : torch.Tensor
            Pupil-plane map [resolution, resolution] (phase, or the pupil).
        index : torch.Tensor, optional
            Subapertures to return (default: all).

        Returns
        -------
        torch.Tensor
            Subaperture maps [nSubap**2 (or len(index)), npix_pupil, npix_pupil], sorted from left to right, top to bottom.
        """
        n = self.npix_pupil
        if self.local_resampling is not None:
            # Pupil patch of every subaperture with a one-pixel margin; the margin outside the array is the linear
            # extrapolation of the two edge pixels, so that a linear phase stays exact
            f = field.float()
            f = torch.cat([2 * f[:1] - f[1:2], f, 2 * f[-1:] - f[-2:-1]], dim=0)
            f = torch.cat([2 * f[:, :1] - f[:, 1:2], f, 2 * f[:, -1:] - f[:, -2:-1]], dim=1)
            npp = self.npix_phase
            patches = f.unfold(0, npp + 2, npp).unfold(1, npp + 2, npp).reshape(self.nSubap * self.nSubap, npp + 2, npp + 2)
            if index is not None:
                patches = patches[index]
            W = self.local_resampling
            return W @ patches @ W.transpose(0, 1)
        m0, w1 = self.subap_resampling
        f = field.to(w1.dtype)
        # separable linear interpolation from the two nearest pupil cells (rows, then columns)
        f = f[m0] * (1 - w1)[:, None] + f[m0 + 1] * w1[:, None]
        field_rescaled = f[:, m0] * (1 - w1)[None, :] + f[:, m0 + 1] * w1[None, :]
        out = field_rescaled.view(self.nSubap, n, self.nSubap, n).permute(0, 2, 1, 3).reshape(self.nSubap * self.nSubap, n, n)
        return out if index is None else out[index]

    def _mft_kernel(self, fwhm, center):
        """
        MFT matrix A[k, j] = exp(-2 pi i (j - (n-1)/2) (k - center) / (fwhm n)), n = npix_lenslet: field of a subaperture
        sampled at n cell-centred points -> image pixel k at the angle (k - center) * plate_scale, for any plate scale.
        Cached per (fwhm, center).
        """
        key = (round(float(fwhm), 12), float(center))
        A = self._mft_cache.get(key)
        if A is None:
            n, n_p = self.npix_lenslet, self.npix_pupil
            j = torch.arange(n_p, dtype=torch.float64, device=self.device) - (n_p - 1) / 2
            k = torch.arange(n, dtype=torch.float64, device=self.device) - center
            A = torch.exp(-2j * np.pi * torch.outer(k, j) / (fwhm * n_p)).to(torch.complex64)
            self._mft_cache[key] = A
        return A

    def get_psf(self, phase, fwhm):
        """
        Spot of every subaperture from the pupil phase, with a matrix Fourier transform (MFT) at the detector plate scale.

        The field of each subaperture is the pupil amplitude times exp(i phase), both sampled at npix_lenslet cell-centred
        points per subaperture (split_subapertures). The phase uses a normalised interpolation, interp(pupil * phase) /
        interp(pupil), so that the points at the pupil edge are not biased by the zeros outside the pupil. Image pixel k is
        at the angle (k - c) * plate_scale exactly (no FFT size rounding), with c = (npix_lenslet - npix_subap) // 2 +
        (npix_subap - 1) / 2, so that a flat phase gives a spot centred in the detector window cut by create_full_frame, for odd
        and even windows. The amplitude is sqrt(covered fraction) of each sample, so the light of a partially illuminated
        subaperture equals its illuminated area.

        Parameters
        ----------
        phase : np.ndarray
            Wavefront phase in radians.
        fwhm : float
            lambda / d in detector pixels (lambda / d / plate_scale). Must be >= 1 (Nyquist: >= 2).

        Returns
        -------
        np.ndarray
            Spot intensities [nSubap**2, npix_lenslet, npix_lenslet].
        """
        self.logger.debug('ShackHartmann::get_psf')
        if fwhm < 1:
            raise ValueError('FWHM must be greater than 2 to guarantee enough spatial sampling.')
        n = self.npix_lenslet
        center = (n - self.npix_subap) // 2 + (self.npix_subap - 1) / 2   # zero tilt -> centre of the detector window

        t0 = time.time()
        input_phase_torch = torch.from_numpy(phase).contiguous().detach().to(self.device) * self.pupil_torch   # no phase outside the pupil
        # Cell-centred samples of every subaperture (npix_lenslet x npix_lenslet); normalized interpolation:
        # interp(pupil * phase) / interp(pupil) keeps only the points inside the pupil
        phase_sub = self.split_subapertures(input_phase_torch, self.valid_index).float() * self.subap_amplitude_inv_valid
        t1 = time.time()
        field = torch.polar(self.subap_amplitude_valid, phase_sub)
        A = self._mft_kernel(fwhm, center)
        E = A @ field @ A.transpose(0, 1)
        psf_valid = (E.real ** 2 + E.imag ** 2) / (fwhm * self.npix_pupil) ** 2
        t2 = time.time()
        # only the valid subapertures are imaged; the others are returned as zeros (same layout as before)
        psf = np.zeros((self.nSubap ** 2, n, n), dtype=np.float32)
        psf[self.valid_index.cpu().numpy()] = psf_valid.cpu().numpy()
        self.logger.debug(f'ShackHartmann::get_psf - Time taken: sample subapertures {t1-t0} [s], MFT {t2-t1} [s]')
        return psf

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

        return subaps
            
    #%% LGS
    def get_convolution_spot(self, src): 
        """
        Compute LGS spot convolution kernel based on sodium profile.

        Parameters
        ----------
        src : Source
            Laser guide star source object.

        Returns
        -------
        None
        """
        # compute the projection of the LGS on the subaperture to simulate 
        #  the spots elongation using a convulotion with gaussian spot
        [X0,Y0]             = [src.laser_coordinates[1],-src.laser_coordinates[0]]     # coordinates of the LLT in [m] from the center (sign convention adjusted to match display position on camera)
        
        # 3D coordinates
        coordinates_3D           = np.zeros([3,len(src.Na_profile[0,:])])
        coordinates_3D_ref       = np.zeros([3,len(src.Na_profile[0,:])])
        delta_dx                 = np.zeros([2,len(src.Na_profile[0,:])])
        delta_dy                 = np.zeros([2,len(src.Na_profile[0,:])])
        
        # coordinates of the subapertures
        
        x_subap                 = np.linspace(-self.telescope.D//2,self.telescope.D//2,self.nSubap)
        y_subap                 = np.linspace(-self.telescope.D//2,self.telescope.D//2,self.nSubap)  
        # pre-allocate memory for shift x and y to apply to the gaussian spot

        # number of pixel
        n_pix                   = self.npix_lenslet
        # size of a pixel in m
        d_pix                   = (self.telescope.D/self.nSubap)/self.npix_lenslet
        v                       = np.linspace(-n_pix*d_pix/2,n_pix*d_pix/2,n_pix)
        [alpha_x,alpha_y]       = np.meshgrid(v,v)
        
        # FWHM of gaussian converted into pixel in arcsec
        sigma_spot              = src.FWHM_spot_up/(2*np.sqrt(np.log(2)))
        for i in range(len(src.Na_profile[0,:])):
                        coordinates_3D[:2,i]           = (self.telescope.D/4)*([X0,Y0]/src.Na_profile[0,i])
                        coordinates_3D[2,i]            = self.telescope.D**2./(8.*src.Na_profile[0,i])/(2.*np.sqrt(3.))
                        coordinates_3D_ref[:,i]        = coordinates_3D[:,i]-coordinates_3D[:,len(src.Na_profile[0,:])//2]
        C_elung                       = []
        C_gauss                       = []
        shift_x_buffer                = []
        shift_y_buffer                = []

        C_gauss                       = []
        criterion_elungation          = self.npix_lenslet*(self.telescope.D/self.nSubap)/self.npix_lenslet

        valid_subap_1D = np.copy(self.valid_subapertures_1D[:])
        count = -1
        # gaussian spot (for calibration)
        I_gauss  = (src.Na_profile[1,:][0]/(src.Na_profile[0,:][0]**2)) * np.exp(- ((alpha_x)**2 + (alpha_y)**2)/(2*sigma_spot**2))
        I_gauss /= I_gauss.sum()

        for i_subap in range(len(x_subap)):
            for j_subap in range(len(y_subap)):
                count += 1
                if valid_subap_1D[count]:
                    I = np.zeros([n_pix,n_pix],dtype=(complex))
                    # I_gauss = np.zeros([n_pix,n_pix],dtype=(complex))
                    shift_X                 = np.zeros(len(src.Na_profile[0,:]))
                    shift_Y                 = np.zeros(len(src.Na_profile[0,:]))
                    for i in range(len(src.Na_profile[0,:])):
                        coordinates_3D[:2,i]           = (self.telescope.D/4)*([X0,Y0]/src.Na_profile[0,i])
                        coordinates_3D[2,i]            = self.telescope.D**2./(8.*src.Na_profile[0,i])/(2.*np.sqrt(3.))
                        
                        coordinates_3D_ref[:,i]        = coordinates_3D[:,i]-coordinates_3D[:,len(src.Na_profile[0,:])//2]

                        delta_dx[0,i]   = coordinates_3D_ref[0,i]*(4/self.telescope.D)
                        delta_dy[0,i]   = coordinates_3D_ref[1,i]*(4/self.telescope.D)
            
                        delta_dx[1,i]   = coordinates_3D_ref[2,i]*(np.sqrt(3)*(4/self.telescope.D)**2)*x_subap[i_subap]
                        delta_dy[1,i]   = coordinates_3D_ref[2,i]*(np.sqrt(3)*(4/self.telescope.D)**2)*y_subap[j_subap]
                        
                        # resulting shift + conversion from radians to pixels in m
                        shift_X[i]          = 206265*self.fov_pixel_arcsec*(delta_dx[0,i] + delta_dx[1,i])
                        shift_Y[i]          = 206265*self.fov_pixel_arcsec*(delta_dy[0,i] + delta_dy[1,i])
     
        
                        I_tmp               = (src.Na_profile[1,:][i]/(src.Na_profile[0,:][i]**2))*np.exp(- ((alpha_x-shift_X[i])**2 + (alpha_y-shift_Y[i])**2)/(2*sigma_spot**2))
                                                
                        I                   += I_tmp
                        
                    # truncation of the wings of the gaussian
                    I[I<self.threshold_convolution*I.max()] = 0 
                    # normalization to conserve energy
                    I /= I.sum()
                    # save 
                    shift_x_buffer.append(shift_X)
                    shift_y_buffer.append(shift_Y)

                    C_elung.append((np.fft.fft2(I)))
                    C_gauss.append((np.fft.fft2(I_gauss)))
                    
        self.shift_x_buffer = np.asarray(shift_x_buffer)
        self.shift_y_buffer = np.asarray(shift_y_buffer)

        self.shift_x_max_arcsec = np.max(shift_x_buffer,axis=1)
        self.shift_y_max_arcsec = np.max(shift_y_buffer,axis=1)
        
        self.shift_x_min_arcsec = np.min(shift_x_buffer,axis=1)
        self.shift_y_min_arcsec = np.min(shift_y_buffer,axis=1)

        self.max_elung_x = np.max(self.shift_x_max_arcsec-self.shift_x_min_arcsec)
        self.max_elung_y = np.max(self.shift_y_max_arcsec-self.shift_y_min_arcsec)
        self.elungation_factor = np.max([self.max_elung_x,self.max_elung_y])/criterion_elungation

        if self.max_elung_x>criterion_elungation or self.max_elung_y>criterion_elungation:            
            self.logger.warning(f'ShackHartmann::get_convolution_spot - The largest spot elongation is {+str(np.round(self.elungation_factor,3))} times \
                                larger than a subaperture! Consider increasing the Field of view parameter')

        self.C = np.asarray(C_elung.copy())
        self.C_gauss = np.asarray(C_gauss)
        self.C_elung = np.asarray(C_elung)
        
        return
    
#%% SH Measurement
    def wfs_integrate(self, ideal_frame, subaps, nPhoton, stream=None):
        """
        Integrate the full detector image and compute valid slopes.

        Parameters
        ----------
        ideal_frame : np.ndarray
            Full-frame ideal image.
        subaps : np.ndarray
            Spot images per subaperture.
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

        # compute the centroid on valid subaperture
        centroid_lenslets = self.centroid(subaps, self.use_brightest)
        
        # discard nan and inf values
        val_inf = np.where(np.isinf(centroid_lenslets))
        val_nan = np.where(np.isnan(centroid_lenslets)) 
        
        if np.shape(val_inf)[1] !=0:
            self.logger.warning('ShackHartmann::wfs_integrate - Some subapertures are giving inf values!')
            centroid_lenslets[np.where(np.isinf(centroid_lenslets))] = 0
        
        if np.shape(val_nan)[1] !=0:
            self.logger.warning('ShackHartmann::wfs_integrate - Some subapertures are giving NaN values!')
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
    def wfs_measure(self,phase_in, src, stream=None):
        """
        Measure slopes from a wavefront phase using the SH-WFS.

        Parameters
        ----------
        phase_in : np.ndarray
            Phase map input [radians].
        src : Source
            Source object.
        stream : int, optional
            Noise stream of the light path (Detector.register_stream); None uses the detector's own generator.

        Returns
        -------
        tuple
            Slopes 1D, slopes 2D, and detector image.
        """
        self.logger.debug("ShackHartmann::wfs_measure")
        # Check input parameters
        if (phase_in is None) or (src is None):
            self.logger.error("ShackHartmann::wfs_measure - Phase or Source are none.")
            raise ValueError('ShackHartmann::wfs_measure - Phase or Source are none.')
       
        t0 = time.time()
        ideal_frame, I, nPhotons = self.ideal_frame(phase_in, src)
        t1 = time.time()
        # Compute the frame and the slopes
        signal, signal_2D, noisy_frame = self.wfs_integrate(ideal_frame, I, nPhotons, stream)
        t2 = time.time()
        self.logger.debug(f'Ideal frame: {(t1-t0)*1000}')
        self.logger.debug(f'Integrate: {(t2-t1)*1000}')
        return signal, signal_2D, noisy_frame

    def ideal_frame(self, phase_in, src):
        """
        Noise-free detector frame of a wavefront phase (the deterministic part of wfs_measure).

        Parameters
        ----------
        phase_in : np.ndarray
            Phase map input [radians].
        src : Source
            Source object.

        Returns
        -------
        tuple
            Ideal full frame, spot images of the valid subapertures and number of photons.
        """
        # compute fwhm
        fwhm = src.wavelength * 206265 / (self.subaperture_size * self.plate_scale)
        # compute the PSF intensity
        I = self.get_psf(phase_in, fwhm)
        # reduce to valid subaperture
        I = I[self.valid_subapertures_1D,:,:]    
                
        # in case of LGS sensor, convolve with LGS spots to create spot elungation
        if self.is_LGS:
            I = np.fft.fftshift(np.abs((np.fft.ifft2(np.fft.fft2(I)*self.C))),axes = [1,2])

        # fill camera frame with computed intensity (only valid subapertures)

        ideal_frame = self.create_full_frame(I)
        # Compute the number of photons considering the flux of the sun, collecting area of the telescope and sampling time
        nPhotons = np.round(src.flux * self.area_eff * self.integration_time * self.lightRatio).astype(int)
        return ideal_frame, I, nPhotons

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

        if self.is_LGS:    
            self.logger.info('{: ^20s}'.format('Spot Elungation')    + '{: ^18s}'.format(str(100*np.round(self.elungation_factor,3)))      +'{: ^18s}'.format('% of a subap' ))
        
      
    
