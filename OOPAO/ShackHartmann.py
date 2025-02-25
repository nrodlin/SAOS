# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""

import inspect
import time

import logging
import logging.handlers
from queue import Queue

import numpy as np
import scipy as sp

from .Detector import Detector
from .tools.tools import bin_ndarray

try:
    from joblib import Parallel, delayed
except:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('WARNING: The joblib module is not installed. This would speed up considerably the operations.')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

class ShackHartmann:
    def __init__(self,nSubap:float,
                 telescope,
                 src,
                 lightRatio:float,
                 threshold_cog:float = 0.01,
                 is_geometric:bool = False, 
                 binning_factor:int = 1,
                 padding_extension_factor:int = 1,
                 threshold_convolution:float = 0.05,
                 shannon_sampling:bool = False,
                 unit_in_rad = False,
                 logger=None):
        
        """SHACK-HARTMANN
        A Shack Hartmann object consists in defining a 2D grd of lenslet arrays located in the pupil plane of the telescope to estimate the local tip/tilt seen by each lenslet. 
        By default the Shack Hartmann detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)
        It requires the following parameters: 

        Parameters
        ----------
        nSubap : float
            The number of subapertures (micro-lenses) along the diameter defined by the telescope.pupil.
        telescope : TYPE
            The telescope object to which the Shack Hartmann is associated. 
            This object carries the phase, flux and pupil information.
        src : TYPE
            The source object to which the Shack Hartmann is associated.
            This objects carries the flux, number of photons and wavelength information.
        lightRatio : float
            Criterion to select the valid subaperture based on flux considerations.
        threshold_cog : float, optional
            Threshold (with respect to the maximum value of the image) 
            to apply to compute the center of gravity of the spots.
            The default is 0.01.
        is_geometric : bool, optional
            Flag to enable the geometric WFS. 
            If True, enables the geometric Shack Hartmann (direct measurement of gradient).
            If False, the diffractive computation is considered.
            The default is False.
        binning_factor : int, optional
            Binning factor of the detector.
            The default is 1.
        padding_extension_factor : int, optional
            Zero-padding factor on the spots intensity images. 
            This is a fast way to provide a larger field of view before the convolution 
            with LGS spots is achieved and allow to prevent wrapping effects.
            The default is 1.
        threshold_convolution : float, optional
            Threshold considered to force the gaussian spots (elungated spots) to go to zero on the edges.
            The default is 0.05.
        shannon_sampling : bool, optional
            If True, the lenslet array spots are sampled at the same sampling as the FFT (2 pix per FWHM).
            If False, the sampling is 1 pix per FWHM (default).
            The default is False.
        unit_in_rad : bool, optional
                If True, the slopes units are in radians.
                If False, the slopes units are in px.
                The default is False.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        ************************** PROPAGATING THE LIGHT TO THE SH OBJECT **************************
        The light can be propagated from a telescope object tel through the Shack Hartmann object wfs using the * operator:        
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the Shack Hartmann detector (phase and flux)
            _ binning of the SH signals
            _ addition of eventual photon noise and readout noise
            _ computation of the Shack Hartmann signals
        
    
        ************************** PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ wfs.signal                     : signal measured by the Shack Hartmann
        _ wfs.signal_2D                  : 2D map of the signal measured by the Shack Hartmann
        _ wfs.random_state_photon_noise  : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time 
        _ wfs.random_state_readout_noise : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time   
        _ wfs.random_state_background    : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time   
        _ wfs.fov_lenslet_arcsec         : Field of View of the subapertures in arcsec
        _ wfs.fov_pixel_binned_arcsec    : Field of View of the pixel in arcsec
        
        The main properties of the object can be displayed using :
            wfs.print_properties()
        
        the following properties can be updated on the fly:
            _ wfs.is_geometric          : switch between diffractive and geometric shackHartmann
            _ wfs.cam.photonNoise       : Photon noise can be set to True or False
            _ wfs.cam.readoutNoise      : Readout noise can be set to True or False
            _ wfs.lightRatio            : reset the valid subaperture selection considering the new value
        
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

        if src.tag != 'source':
            self.logger.error(f'ShackHartmann::init - Source type is {src.tag}, which is not supported. This module expects the type to be SOURCE')
            raise AttributeError(f'Expected SOURCE type, but {src.tag} was provided instead.')

        self.is_geometric                   = is_geometric
        self.nSubap                         = nSubap
        self.lightRatio                     = lightRatio
        self.binning_factor                 = binning_factor
        self.zero_padding                   = 2
        self.padding_extension_factor       = padding_extension_factor
        self.threshold_convolution          = threshold_convolution
        self.threshold_cog                  = threshold_cog
        self.shannon_sampling               = shannon_sampling
        self.unit_in_rad                    = unit_in_rad
        # case where the spots are zeropadded to provide larger fOV
        if padding_extension_factor>=2:
            self.n_pix_subap            = int(padding_extension_factor*telescope.resolution// self.nSubap)            
            self.is_extended            = True
            self.binning_factor         = padding_extension_factor
            self.zero_padding           = 1 
        else:
            self.n_pix_subap            = telescope.resolution // self.nSubap 
            self.is_extended            = False
        

        # different resolutions needed
        self.subaperture_size           = telescope.D / self.nSubap
        self.n_pix_subap_init           = telescope.resolution // self.nSubap    
        self.extra_pixel                = (self.n_pix_subap-self.n_pix_subap_init)//2         
        self.n_pix_lenslet_init         = self.n_pix_subap_init*self.zero_padding 
        self.n_pix_lenslet              = self.n_pix_subap*self.zero_padding 
        self.center                     = self.n_pix_lenslet//2 
        self.center_init                = self.n_pix_lenslet_init//2 
        self.lenslet_frame              = np.zeros([self.n_pix_subap*self.zero_padding,self.n_pix_subap*self.zero_padding], dtype =complex)
        self.outerMask                  = np.ones([self.n_pix_subap_init*self.zero_padding, self.n_pix_subap_init*self.zero_padding ])
        self.outerMask[1:-1,1:-1]       = 0
        
        # Compute camera frame in case of multiple measurements
        self.get_raw_data_multi     = False
        # detector camera
        self.cam                        = Detector(round(nSubap*self.n_pix_subap))                     # WFS detector object
        self.cam.photonNoise            = 0
        self.cam.readoutNoise           = 0        # single lenslet
        # noies random states
        self.random_state_photon_noise      = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_readout_noise     = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_background        = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        
        # field of views
        self.fov_lenslet_arcsec         = (self.n_pix_subap*206265*self.binning_factor/self.padding_extension_factor*src.wavelength/(telescope.D/self.nSubap))/(1+self.shannon_sampling)
        self.fov_pixel_arcsec           = self.fov_lenslet_arcsec/ self.n_pix_subap
        self.fov_pixel_binned_arcsec    = self.fov_lenslet_arcsec/ self.n_pix_subap_init

        X_map, Y_map= np.meshgrid(np.arange(self.n_pix_subap//self.binning_factor),np.arange(self.n_pix_subap//self.binning_factor))
        self.X_coord_map = np.atleast_3d(X_map).T
        self.Y_coord_map = np.atleast_3d(Y_map).T
        
        if src.type == 'LGS':
            self.is_LGS                 = True
        else:
            self.is_LGS                 = False
                
        # cube of lenslet zero padded
        self.cube                   = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init])
        self.cube_flux              = np.zeros([self.nSubap**2,self.n_pix_subap_init,self.n_pix_subap_init],dtype=(complex))
        self.index_x                = []
        self.index_y                = []

        # phasor to center spots in the center of the lenslets
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.n_pix_lenslet_init-1,self.n_pix_lenslet_init),np.linspace(0,self.n_pix_lenslet_init-1,self.n_pix_lenslet_init))
        self.phasor                = np.exp(-(1j*np.pi*(self.n_pix_lenslet_init+1)/self.n_pix_lenslet_init)*(xx+yy))
        self.phasor_tiled          = np.moveaxis(np.tile(self.phasor[:,:,None],self.nSubap**2),2,0)
        
        # Get subapertures index and flux per subaperture        
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet),np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet))
        self.phasor_expanded       = np.exp(-(1j*np.pi*(self.n_pix_lenslet+1)/self.n_pix_lenslet)*(xx+yy))
        self.phasor_expanded_tiled          = np.moveaxis(np.tile(self.phasor_expanded[:,:,None],self.nSubap**2),2,0)

        # The normalized flux maps considers the efficiency in the reflectance of the light in the pupil, integration time and area and light ratio derived to the WFS
        # The flux is computed as norm_flux * nPhoton
        self.norm_flux_map = self.lightRatio*telescope.pupilReflectivity*telescope.samplingTime*(telescope.D/telescope.resolution)**2

        self.initialize_flux(src, self.norm_flux_map)
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)
        self.current_nPhoton = src.nPhoton
        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)

        self.flux_flag = False     

        self.logger.info('ShackHartmann::__init__ - Selecting valid subapertures based on flux considerations..')
        self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture,[self.nSubap,self.nSubap])
        self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(self.photon_per_subaperture), [self.nSubap,self.nSubap])
        self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])
        [self.validLenslets_x , self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
        self.nSignal = 2*self.nValidSubaperture        
 
        if self.is_LGS:
            self.get_convolution_spot(src) 

        # WFS initialization
        self.initialize_wfs(telescope, src)       
        
    def initialize_wfs(self, telescope, src):
        self.isInitialized = False

        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        
        self.cam.photonNoise        = 0
        self.cam.readoutNoise       = 0       
        
        # flux per subaperture
        self.reference_slopes_maps  = np.zeros([self.nSubap*2,self.nSubap])
        self.slopes_units           = 1

        self.logger.info('ShackHartmann::initialize_wfs - Acquiring reference slopes..')
        null_phase = np.zeros((telescope.resolution, telescope.resolution))        
        _, self.reference_slopes_maps,_ = self.wfs_measure(null_phase, src)        
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

        self.cam.photonNoise        = readoutNoise
        self.cam.readoutNoise       = photonNoise
        
        self.print_properties()

    def centroid(self,image,threshold =0.01):
        im = np.atleast_3d(image.copy())    
        im[im<(threshold*im.max())] = 0
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

        # Create the flux cube storing the flux at each subaperture
        self.cube_flux = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init],dtype=float)

        # Build the flux map
        input_flux_map = src.nPhoton * norm_flux_map

        input_flux_map = input_flux_map.reshape(self.nSubap, self.n_pix_subap_init, 
                         self.nSubap, self.n_pix_subap_init).transpose(0, 2, 1, 3).reshape(self.nSubap*self.nSubap, 
                                                                                            self.n_pix_subap_init, self.n_pix_subap_init) 
        # Assign the illumination to the region, considering zeropadding
        self.cube_flux[:,self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2,
                        self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2] = input_flux_map
      
        # Get general properties of the illumination
        self.photon_per_subaperture = np.apply_over_axes(np.sum, self.cube_flux, [1,2])
        self.current_nPhoton = src.nPhoton
        return
    # This function takes the phase at the pupil as input. Then, the flux at each subaperture (pupil function) is multiplied 
    # by the complex phase to obtain the PSF per subaperture, as an array of dimensions (nSubap**2, n_pix_lenslet_init, n_pix_lenslet_init)
    # The subapertures are sorted from left to right, top to bottom.

    def get_lenslet_em_field(self, phase):
        self.logger.debug('ShackHartmann::get_lenslet_em_field')
        # Create the output with the dimensions
        cube_em = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init],dtype=complex)
        # Reshape the subapertures to a grid of subapertures. The sensor can be zeropadded, so the phase is filling the central part of the subaperture
        phase_reshaped = phase.reshape(self.nSubap, self.n_pix_subap_init, 
                                       self.nSubap, self.n_pix_subap_init).transpose(0, 2, 1, 3).reshape(self.nSubap*self.nSubap, 
                                                                                                           self.n_pix_subap_init, self.n_pix_subap_init)
        # Apply the exponential to full matrix
        cube_em[:,self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2,
                  self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2] = np.exp(1j * phase_reshaped)

        # Apply light scaling
        cube_em *= np.sqrt(self.cube_flux) * self.phasor_tiled

        return cube_em
  
    def create_full_frame(self, subaps):
        self.logger.debug('shackHartmann::create_full_frame')
        
        ideal_frame = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,
                                self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)
        
        index_valid = 0
        
        for x in range(self.nSubap):
            for y in range(self.nSubap):
                if self.valid_subapertures[x,y]: 
                    ideal_frame[x*self.n_pix_subap//self.binning_factor:(x+1)*self.n_pix_subap//self.binning_factor,
                                y*self.n_pix_subap//self.binning_factor:(y+1)*self.n_pix_subap//self.binning_factor] = subaps[index_valid,:,:]                         
                    index_valid += 1
        
        return ideal_frame

    def get_subaps(self, noisy_frame):
        self.logger.debug('shackHartmann::get_subaps')
        
        subaps = noisy_frame.reshape(self.nSubap, self.n_pix_subap, 
                                            self.nSubap, self.n_pix_subap).transpose(0, 2, 1, 3).reshape(self.nSubap*self.nSubap, 
                                                                                                         self.n_pix_subap, self.n_pix_subap)
        center = self.n_pix_subap//2

        subaps[:,center - self.n_pix_subap//self.binning_factor//2:center+self.n_pix_subap//self.binning_factor//2,
                center - self.n_pix_subap//self.binning_factor//2:center+self.n_pix_subap//self.binning_factor//2] = subaps
        
        subaps = subaps[self.valid_subapertures_1D,:,:]

        return subaps
        
    #%% GEOMETRIC    
         
    def gradient_2D(self,arr):
        res_x = (np.gradient(arr,axis=0)/self.telescope.pixelSize)*self.telescope.pupil
        res_y = (np.gradient(arr,axis=1)/self.telescope.pixelSize)*self.telescope.pupil
        return res_x,res_y
        
    def lenslet_propagation_geometric(self,arr):
        
        [SLx,SLy]  = self.gradient_2D(arr)
        
        sy = (bin_ndarray(SLx, [self.nSubap,self.nSubap], operation='sum'))
        sx = (bin_ndarray(SLy, [self.nSubap,self.nSubap], operation='sum'))
        
        return np.concatenate((sx,sy))
            
    #%% LGS
    def get_convolution_spot(self, src): 
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
        n_pix                   = self.n_pix_lenslet
        # size of a pixel in m
        d_pix                   = (self.telescope.D/self.nSubap)/self.n_pix_lenslet_init
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
        criterion_elungation          = self.n_pix_lenslet*(self.telescope.D/self.nSubap)/self.n_pix_lenslet_init

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
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Warning: The largest spot elongation is '+str(np.round(self.elungation_factor,3))+' times larger than a subaperture! Consider using a higher resolution or increasing the padding_extension_factor parameter')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        self.C = np.asarray(C_elung.copy())
        self.C_gauss = np.asarray(C_gauss)
        self.C_elung = np.asarray(C_elung)
        
        return
    
#%% SH Measurement
    def wfs_integrate(self, ideal_frame, subaps):
        # propagate to detector to add noise and detector effects
        noisy_frame = self.cam.integrate(ideal_frame)
        subaps = self.get_subaps(noisy_frame)

        # compute the centroid on valid subaperture
        centroid_lenslets = self.centroid(subaps, self.threshold_cog)
        
        # discard nan and inf values
        val_inf = np.where(np.isinf(centroid_lenslets))
        val_nan = np.where(np.isnan(centroid_lenslets)) 
        
        if np.shape(val_inf)[1] !=0:
            self.logger.warning('ShackHartmann::wfs_integrate - Some subapertures are giving inf values!')
            centroid_lenslets[np.where(np.isinf(centroid_lenslets))] = 0
        
        if np.shape(val_nan)[1] !=0:
            self.logger.warning('ShackHartmann::wfs_integrate - Some subapertures are giving NaN values!')
            centroid_lenslets[np.where(np.isnan(centroid_lenslets))] = 0
            
        # compute slopes-maps
        # self.SX[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[:,0]
        # self.SY[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[:,1]

        sx = np.zeros((self.nSubap, self.nSubap))
        sy = np.zeros((self.nSubap, self.nSubap))

        sx[self.validLenslets_x, self.validLenslets_y] = centroid_lenslets[:,0]
        sy[self.validLenslets_x, self.validLenslets_y] = centroid_lenslets[:,1]

        # signal_2D                           = np.concatenate((self.SX,self.SY)) - self.reference_slopes_maps

        signal_2D                           = np.concatenate((sx, sy)) - self.reference_slopes_maps
        signal_2D[~self.valid_slopes_maps]  = 0
        
        signal_2D                      = signal_2D/self.slopes_units
        signal                         = signal_2D[self.valid_slopes_maps]
        
        return signal, signal_2D, noisy_frame
    
    # Receives the phase [rad]] and return the slopes measured by the SH in [px]
    # Expects the ideal frame WITHOUT pupil applied.
    def wfs_measure(self,phase_in, src, integrate = True):
        self.logger.debug("ShackHartmann::wfs_measure")
        # Check input parameters
        if (phase_in is None) or (src is None):
            self.logger.error("ShackHartmann::wfs_measure - Phase or Source are none.")
            raise ValueError('ShackHartmann::wfs_measure - Phase or Source are none.')
        # Check if it is necessary to recompute the flux per subaperture, this is important as it will take longer
        # time during the execution and the number of subaps may vary!! --> Be careful, the IM shall vary accondingly
        if self.current_nPhoton != src.nPhoton:
            self.logger.info('ShackHartmann::wfs_measure - Number of photons changed, updating flux on subaps')
            self.initialize_flux(src, self.norm_flux)   
        
        # There are two possible WFS strategies: diffractive or geometric. 
        # Diffractive applies the phase and compute the Fourier transformed object, later applying a centroid algorithm.
        # Geometric computes the slope directly from the phase points, which reduces the execution time, though it is less realistic

        if self.is_geometric is False:
            ##%%%%%%%%%%%%  DIFFRACTIVE SH WFS %%%%%%%%%%%%
            # normalization for FFT
            norma = self.cube.shape[1]

            # compute spot intensity
            sp.fft.set_workers(self.nSubap)  # Use 1 thread per row of subapertures
            I = (np.abs(sp.fft.fft2(np.asarray(self.get_lenslet_em_field(phase_in)), axes=[1,2]) / norma) ** 2)
            # I = (np.abs(np.fft.fft2(np.asarray(self.get_lenslet_em_field(phase_in)),axes=[1,2])/norma)**2)

            # reduce to valid subaperture
            I = I[self.valid_subapertures_1D,:,:]
            # Check if there are wrapping effects at the edges of the pupil
            if self.flux_flag == False:
                self.sum_I   = np.sum(I,axis=0)
                self.edge_subaperture_criterion = np.sum(I*self.outerMask)/np.sum(I)

            if (self.edge_subaperture_criterion>0.05) and (self.flux_flag == False):
                self.flux_flag = True
                self.logger.warning('ShackHartmann::wfs_measure - THE LIGHT IN THE SUBAPERTURE MAY BE WRAPPING !!!'\
                                   + str(np.round(100*self.edge_subaperture_criterion,1))+'% of the total flux detected on the edges of the subapertures.'\
                                   'You may want to lower the seeing value or increase the resolution')            
            
            # if FoV is extended, zero pad the spot intensity
            if self.is_extended:
                I = np.pad(I,[[0,0],[self.extra_pixel,self.extra_pixel],
                              [self.extra_pixel,self.extra_pixel]]) 
            
            # in case of LGS sensor, convolve with LGS spots to create spot elungation
            if self.is_LGS:
                I = np.fft.fftshift(np.abs((np.fft.ifft2(np.fft.fft2(I)*self.C))),axes = [1,2])

            # Crop to get the spot at shannon sampling
            if self.shannon_sampling:
                subaps =  I[:,self.n_pix_subap//2:-self.n_pix_subap//2,self.n_pix_subap//2:-self.n_pix_subap//2]

            if self.binning_factor>1:
                subaps =  bin_ndarray(subaps,[subaps.shape[0], self.n_pix_subap//self.binning_factor,
                                                                        self.n_pix_subap//self.binning_factor], operation='sum')
            else:
                subaps =  bin_ndarray(I,[I.shape[0], self.n_pix_subap//self.binning_factor,
                                                      self.n_pix_subap//self.binning_factor], operation='sum')
            
            # bin the 2D spots intensity to get the desired number of pixel per subaperture
            if self.binning_factor>1:
                subaps =  bin_ndarray(subaps,[subaps.shape[0], self.n_pix_subap//self.binning_factor,
                                                                        self.n_pix_subap//self.binning_factor], operation='sum')
            # fill camera frame with computed intensity (only valid subapertures)

            ideal_frame = self.create_full_frame(subaps)

            if integrate:
                signal, signal_2D, noisy_frame = self.wfs_integrate(ideal_frame, subaps)                
        else:
            ##%%%%%%%%%%%%  GEOMETRIC SH WFS %%%%%%%%%%%%
            ideal_frame   = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)

            signal_2D = self.lenslet_propagation_geometric(phase_in)*self.valid_slopes_maps/self.slopes_units
                
            signal = signal_2D[self.valid_slopes_maps]
            
            noisy_frame = self.cam.integrate(ideal_frame)
        
        return signal, signal_2D, noisy_frame

    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHACK HARTMANN WFS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^20s}'.format('Subapertures')         + '{: ^18s}'.format(str(self.nSubap))                                   )
        print('{: ^20s}'.format('Subaperture Size')     + '{: ^18s}'.format(str(np.round(self.subaperture_size, 2)))         +'{: ^18s}'.format('[m]'     ))
        print('{: ^20s}'.format('Pixel FoV')            + '{: ^18s}'.format(str(np.round(self.fov_pixel_binned_arcsec,2)))   +'{: ^18s}'.format('[arcsec]'))
        print('{: ^20s}'.format('Subapertue FoV')       + '{: ^18s}'.format(str(np.round(self.fov_lenslet_arcsec,2)))        +'{: ^18s}'.format('[arcsec]'))
        print('{: ^20s}'.format('Valid Subaperture')    + '{: ^18s}'.format(str(str(self.nValidSubaperture))))                   
        print('{: ^20s}'.format('Binning Factor')    + '{: ^18s}'.format(str(str(self.binning_factor))))                   

        if self.is_LGS:    
            print('{: ^20s}'.format('Spot Elungation')    + '{: ^18s}'.format(str(100*np.round(self.elungation_factor,3)))      +'{: ^18s}'.format('% of a subap' ))
        print('{: ^20s}'.format('Geometric WFS')    + '{: ^18s}'.format(str(self.is_geometric)))
        print('{: ^20s}'.format('Shannon Sampling')    + '{: ^18s}'.format(str(self.shannon_sampling)))

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')        
        if self.is_geometric:
            print('WARNING: THE PHOTON AND READOUT NOISE ARE NOT CONSIDERED FOR GEOMETRIC SH-WFS')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    @property
    def is_geometric(self):
        return self._is_geometric
    
    @is_geometric.setter
    def is_geometric(self,val):
        self._is_geometric = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Re-initializing WFS...')
                self.initialize_wfs()
    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self,val):
        self._C = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Re-initializing WFS...')
                self.initialize_wfs()      
    @property
    def lightRatio(self):
        return self._lightRatio
    
    @lightRatio.setter
    def lightRatio(self,val):
        self._lightRatio = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Selecting valid subapertures based on flux considerations..')

                self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(self.photon_per_subaperture), [self.nSubap,self.nSubap])
        
                self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])

                [self.validLenslets_x , self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))
        
                # index of valid slopes X and Y
                self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
                # number of valid lenslet
                self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
                self.nSignal = 2*self.nValidSubaperture
                
                print('Re-initializing WFS...')
                self.initialize_wfs()
                print('Done!')
      
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
