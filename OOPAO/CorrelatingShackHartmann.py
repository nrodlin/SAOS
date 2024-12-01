# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""

import inspect
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from .Detector import Detector
from .tools.tools import bin_ndarray

class CorrelatingShackHartmann:
    def __init__(self,nSubap:float,
                 plate_scale:float,
                 telescope,
                 lightRatio:float,
                 ideal_pattern:bool = False,
                 pattern_criteria = None,
                 subaperture_sel = [],
                 fov_crop = None,
                 kernel_size = 5):
        
        """SHACK-HARTMANN
        A Shack Hartmann object consists in defining a 2D grd of lenslet arrays located in the pupil plane of the telescope to estimate the local tip/tilt seen by each lenslet. 
        By default the Shack Hartmann detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)
        It requires the following parameters: 

        Parameters
        ----------
        nSubap : float
            The number of subapertures (micro-lenses) along the diameter defined by the telescope.pupil.
        plate_scale : float
            The Plate Scale of the WFS image in "/px.
        telescope : TYPE
            The telescope object to which the Shack Hartmann is associated. 
            This object carries the phase, flux and pupil information.
        lightRatio : float
            Criterion to select the valid subaperture based on flux considerations.
        idea_pattern : bool, optional
            Flag to enable the geometric WFS. 
            If True, enables the geometric Shack Hartmann (direct measurement of gradient).
            If False, the diffractive computation is considered.
            The default is False.
        unit_P2V : bool, optional
                If True, the slopes units are calibrated using a Tip/Tilt normalized to 2 Pi peak-to-valley.
                If False, the slopes units are calibrated using a Tip/Tilt normalized to 1 in the pupil (Default). In that case the slopes are expressed in [rad].
                The default is False.
        pattern_criteria : str, optional
                None by default, using the ideal pattern
                If specified, the str defines the method to select the pattern among the subapertures
                if 'contrast', the subaperture with the highest contrast is selected
                If 'subap', then the subaperture specified in subaperture_sel is used
        subaperture_sel : list, optional
                Empty by default, requires pattern_criteria to be 'subap' in order to be used.
                The list contains the 2D position of the subaperture that is selected as pseudo-reference
        fov_crop : float, optional
                By default it is None. This parameter is used to crop the window of the subaperture after convoluting the phase by the sun patch. 
                This is useful to remove difraction from the subapertures. Must be smaller than the FoV of the patch.
        kernel_size : int, optional
                This parameter sets the Kernel size to select the region around the maximum

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
            _ wfs.cam.photonNoise       : Photon noise can be set to True or False
            _ wfs.cam.readoutNoise      : Readout noise can be set to True or False
            _ wfs.lightRatio            : reset the valid subaperture selection considering the new value
        
        """ 
        self.tag                            = 'correlatingShackHartmann'
        self.plate_scale                    = plate_scale
        self.telescope                      = telescope
        if self.telescope.src is None:
            raise AttributeError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        self.ideal_pattern                  = ideal_pattern
        self.nSubap                         = nSubap
        self.lightRatio                     = lightRatio       
        self.fov_crop                       = fov_crop
        self.kernel_size                    = kernel_size
        self.subaperture_sel                = subaperture_sel
        self.pattern_criteria               = pattern_criteria

        if self.fov_crop is not None:
            if self.fov_crop > self.telescope.src.fov:
                raise AttributeError('The crop of the FoV cannot be larger than the base FoV.')
            else:
                self.n_pix_fov_crop             = np.round(self.fov_crop / self.plate_scale).astype(int)

        if self.telescope.src.tag == "sun":
            self.n_pix_lenslet = int((self.telescope.src.fov + self.telescope.src.patch_padding) // self.plate_scale)
            self.n_pix_subap = int(self.telescope.src.fov // self.plate_scale)
        else:
            raise AttributeError('Correlating SH WFS only supports sun objects')
                
        # different resolutions needed
        self.n_pix_subap_init           = int(self.telescope.src.fov // self.plate_scale)
        self.extra_pixel                = (self.n_pix_subap-self.n_pix_subap_init)//2         
        self.n_pix_lenslet_init         = self.n_pix_subap_init
        self.n_pix_lenslet              = self.n_pix_subap
        self.center                     = self.n_pix_lenslet//2 
        self.center_init                = self.n_pix_lenslet_init//2 
        self.lenslet_frame              = np.zeros([self.n_pix_subap,self.n_pix_subap], dtype =complex)
        self.outerMask                  = np.ones([self.n_pix_subap_init, self.n_pix_subap_init ])
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
        self.fov_lenslet_arcsec         = self.telescope.src.fov
        self.fov_pixel_arcsec           = self.fov_lenslet_arcsec / self.n_pix_subap
        self.fov_pixel_binned_arcsec    = self.fov_lenslet_arcsec / self.n_pix_subap_init

        X_map, Y_map= np.meshgrid(np.arange(self.n_pix_subap),np.arange(self.n_pix_subap))
        self.X_coord_map = np.atleast_3d(X_map).T
        self.Y_coord_map = np.atleast_3d(Y_map).T

        # camera frame
        self.raw_img_spacing    = 2 # in px
        if self.fov_crop is None:
            self.raw_data           = np.zeros([(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap], dtype =float)
        else:
            self.raw_data           = np.zeros([(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap], dtype =float)
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

        self.initialize_flux()
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)
        self.current_nPhoton = self.telescope.src.nPhoton
        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)        
        print('Selecting valid subapertures based on flux considerations..')
        self.valid_subapertures = np.zeros((self.nSubap, self.nSubap), dtype=bool)
        n_points_per_subap = self.telescope.pupil.shape[0] // self.nSubap

        for i in range(self.nSubap):
            for j in range(self.nSubap):
                tmp_pupil = np.sum(self.telescope.pupil[i*n_points_per_subap:(i+1)*n_points_per_subap, 
                                                        j*n_points_per_subap:(j+1)*n_points_per_subap]) / (n_points_per_subap**2)
                if tmp_pupil == 1: # Criteria: there are phase points at every point of the phase
                    self.valid_subapertures[i, j] = True
        tmp_valid_subapertures = np.copy(self.valid_subapertures)
        """ for i in range(self.nSubap):
            for j in range(self.nSubap):
                if (j+1 < self.nSubap) and (j-1 >= 0):
                    if (tmp_valid_subapertures[i, j] == True and tmp_valid_subapertures[i, j-1] == False) or \
                    (tmp_valid_subapertures[i, j] == True and tmp_valid_subapertures[i, j+1] == False):
                        self.valid_subapertures[i, j] = False
                elif (j-1 < 0):
                    self.valid_subapertures[i, j] = False """
        self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture,[self.nSubap,self.nSubap])
        self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])
        [self.validLenslets_x , self.validLenslets_y] = np.where(self.valid_subapertures == True)
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
        self.nSignal = 2*self.nValidSubaperture 

        self.max_kernel = np.ones((self.kernel_size, self.kernel_size))
        # Build sun base images and filters for the WFS
        self.subDirs_size_wfs = np.round( (self.telescope.src.subDirs_sun.shape[0] * self.telescope.src.img_PS) / self.plate_scale).astype(int)
        self.subDirs_filt_size_wfs = np.round(self.telescope.src.subDirs_coordinates[2,0,0] / self.plate_scale).astype(int)
        self.subDirs_sun_wfs_padded = None

        self.subDirs_sun_wfs = np.zeros((self.subDirs_size_wfs,self.subDirs_size_wfs, self.telescope.src.nSubDirs, self.telescope.src.nSubDirs))
        self.filt_2D_wfs = np.zeros((self.subDirs_filt_size_wfs, self.subDirs_filt_size_wfs, self.telescope.src.nSubDirs, self.telescope.src.nSubDirs))

        for i in range(self.telescope.src.nSubDirs):
            for j in range(self.telescope.src.nSubDirs):
                self.subDirs_sun_wfs[:,:,i,j] = cv2.resize(self.telescope.src.subDirs_sun[:,:,i,j], (self.subDirs_size_wfs, self.subDirs_size_wfs))
                self.filt_2D_wfs[:,:,i,j]     = cv2.resize(self.telescope.src.filter_2D[:,:,i,j],   (self.subDirs_filt_size_wfs, self.subDirs_filt_size_wfs))

        # WFS initialization
        self.initialize_wfs()

    def get_subap_img(self, apply_atmosphere=True,index=0):

        # Make sure that the phase is updated w.r.t. OPD
        for i in range(self.telescope.src.nSubDirs**2):
            if np.ndim(self.telescope.OPD) > 3:
                self.telescope.src.sun_subDir_ast.src[i].phase = np.squeeze(self.telescope.OPD[i][:,:,index]* (2*np.pi/self.telescope.src.wavelength))
            else:
                self.telescope.src.sun_subDir_ast.src[i].phase = self.telescope.OPD[i]* (2*np.pi/self.telescope.src.wavelength)

        # Read phase per subdir and subap

        tmp_phase_subDirs = np.asarray([self.telescope.src.sun_subDir_ast.src[i].phase for i in range(self.telescope.src.nSubDirs**2)])

        if apply_atmosphere == False: # This means that the user wants the WFS without the atmosphere applied
            tmp_phase_subDirs[np.abs(tmp_phase_subDirs)>0] = 1

        nPoints_per_subap = tmp_phase_subDirs.shape[1] // self.nSubap
        phase_per_subap = np.zeros((self.telescope.src.nSubDirs**2, self.nSubap**2, self.subDirs_size_wfs, self.subDirs_size_wfs))

        for i in range(self.nSubap):
            for j in range(self.nSubap):
                for k in range(self.telescope.src.nSubDirs**2):
                    index = i*self.nSubap + j
                    phase_per_subap[k, index, :, :] = cv2.resize(tmp_phase_subDirs[k, nPoints_per_subap*i:nPoints_per_subap*(i+1), nPoints_per_subap*j:nPoints_per_subap*(j+1)],
                                                                  (self.subDirs_size_wfs, self.subDirs_size_wfs), interpolation=cv2.INTER_LINEAR)

        # Define padding and image intensity
        pad_img_size = (2 ** np.ceil(np.log2(self.subDirs_size_wfs * 2))).astype(int)

        pad_top = (pad_img_size - self.subDirs_sun_wfs.shape[0]) // 2
        pad_bottom = pad_img_size - self.subDirs_sun_wfs.shape[0] - pad_top
        pad_left = (pad_img_size - self.subDirs_sun_wfs.shape[1]) // 2
        pad_right = pad_img_size - self.subDirs_sun_wfs.shape[1] - pad_left

        if (self.subDirs_sun_wfs_padded is None) or (self.subDirs_sun_wfs_padded.shape[-1] != pad_img_size):
            self.subDirs_sun_wfs_padded = np.zeros((self.telescope.src.nSubDirs**2, 1, pad_img_size, pad_img_size))

            for dirx in range(self.telescope.src.nSubDirs):
                for diry in range(self.telescope.src.nSubDirs):
                    index = dirx * self.telescope.src.nSubDirs + diry
                    self.subDirs_sun_wfs_padded[index, 0, :, :] = np.pad(np.squeeze(self.subDirs_sun_wfs[:,:,dirx,diry]), pad_width=((pad_top, pad_bottom), (pad_left, pad_right)))
        
        # Compute and normalize PSF
        #phase_valid = phase_per_subap[0,self.valid_subapertures_1D,:,:]
        #plt.imshow(phase_valid[0,:,:]), plt.show()
        #plt.imshow(phase_valid[1,:,:]), plt.show()
        #plt.imshow(phase_valid[8,:,:]), plt.show()
        complex_phase = np.exp(1j*(phase_per_subap[:,self.valid_subapertures_1D,:,:]-np.pi))
        complex_phase_ffted = np.fft.fft2(complex_phase, s=(pad_img_size, pad_img_size), axes=(2,3))
        complex_phase_ffted_shift = np.fft.fftshift(complex_phase_ffted, axes=(2,3))

        psf_per_subdir_per_subap = np.power(np.abs(complex_phase_ffted_shift), 2)

        psf_per_subdir_per_subap = psf_per_subdir_per_subap / np.sum(psf_per_subdir_per_subap[0,0,:,:])

        # Compute subDirs imgs

        psf_fft = np.fft.fft2(psf_per_subdir_per_subap, axes=(2,3))
        sun_fft = np.fft.fft2(self.subDirs_sun_wfs_padded, axes=(2,3))

        sun_fft_conv = np.abs(np.fft.fftshift(np.fft.ifft2(sun_fft*psf_fft, axes=(2,3)), axes=(2,3)))

        # Compute subap image
        subap_padding_fov_px = np.round((self.telescope.src.fov+self.telescope.src.patch_padding)/self.plate_scale).astype(int)
        
        temp_sun_subap = np.zeros((self.nValidSubaperture, subap_padding_fov_px, subap_padding_fov_px))
                
        for dirX in range(self.telescope.src.nSubDirs):
            for dirY in range(self.telescope.src.nSubDirs):
                index = dirX *self.telescope.src.nSubDirs + dirY

                gl_cx = np.floor(dirX*self.subDirs_filt_size_wfs/2).astype(int)
                gl_cy = np.floor(dirY*self.subDirs_filt_size_wfs/2).astype(int)

                gl_cx_end = gl_cx + self.subDirs_filt_size_wfs
                gl_cy_end = gl_cy + self.subDirs_filt_size_wfs

                cx_subDir = pad_top + np.round(self.telescope.src.subDir_margin/self.plate_scale).astype(int)
                cy_subDir = pad_left + np.round(self.telescope.src.subDir_margin/self.plate_scale).astype(int)

                temp_sun_subap[:,gl_cx:gl_cx_end,gl_cy:gl_cy_end] += np.squeeze(sun_fft_conv[index,:,cx_subDir:cx_subDir+self.subDirs_filt_size_wfs, 
                                                                                                     cy_subDir:cy_subDir+self.subDirs_filt_size_wfs] * self.filt_2D_wfs[:,:,dirX,dirY])

        # Crop central fov, without external padding | also consider if the fov_crop is selected
        if self.fov_crop is not None:
            cx = (temp_sun_subap.shape[1] - self.n_pix_fov_crop) // 2
            cy = (temp_sun_subap.shape[2] - self.n_pix_fov_crop) // 2
            sun_subap = temp_sun_subap[:, cx:cx+self.n_pix_fov_crop, cy:cy+self.n_pix_fov_crop]

        else:
            cx = (temp_sun_subap.shape[1]- self.n_pix_subap) // 2
            cy = (temp_sun_subap.shape[2]- self.n_pix_subap) // 2
            sun_subap = temp_sun_subap[:, cx:cx+self.n_pix_subap, cy:cy+self.n_pix_subap]     

        return sun_subap
        
    def initialize_wfs(self):
        self.isInitialized = False

        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        
        self.cam.photonNoise        = 0
        self.cam.readoutNoise       = 0       
        
        # reference signal
        self.sx0                    = np.zeros([self.nSubap,self.nSubap])
        self.sy0                    = np.zeros([self.nSubap,self.nSubap])
        # signal vector
        self.sx                     = np.zeros([self.nSubap,self.nSubap])
        self.sy                     = np.zeros([self.nSubap,self.nSubap])
        # signal map
        self.SX                     = np.zeros([self.nSubap,self.nSubap])
        self.SY                     = np.zeros([self.nSubap,self.nSubap])
        # flux per subaperture
        self.references_2D  = np.zeros([self.nSubap*2,self.nSubap])
        self.geometric_centroid = np.zeros([self.nSubap*2,self.nSubap])

        print('Acquiring reference slopes..')
        self.telescope.resetOPD() 
        pseudo_ref_patch = self.get_subap_img(False) 
        self.pseudo_ref_img = self.generate_pseudo_reference_img(pseudo_ref_patch)
        self.corrImg = self.wfs_measure() 
        self.references_1D = np.copy(self.signal)
        self.references_2D = np.copy(self.signal_2D) 

        for x in range(self.nSubap):
            for y in range(self.nSubap):
                cx = (2*x+1)*self.raw_img_spacing + x*self.n_pix_subap + self.n_pix_subap // 2
                cy = (2*y+1)*self.raw_img_spacing + y*self.n_pix_subap + self.n_pix_subap // 2
                
                self.geometric_centroid[x, y] = cx
                self.geometric_centroid[self.nSubap+x, y] = cy

        self.isInitialized = True
        print('Done!')

        self.cam.photonNoise        = photonNoise
        self.cam.readoutNoise       = readoutNoise
        self.telescope.resetOPD()
        self.print_properties()

    def centroid(self,image):
        im = np.atleast_3d(image.copy())  
        centroid_out         = np.zeros([im.shape[0],2])
        X_map, Y_map= np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[2]))
        X_coord_map = np.atleast_3d(X_map).T
        Y_coord_map = np.atleast_3d(Y_map).T
        norma                   = np.sum(np.sum(im,axis=1),axis=1)
        centroid_out[:,0]    = np.sum(np.sum(im*X_coord_map,axis=1),axis=1)/norma
        centroid_out[:,1]    = np.sum(np.sum(im*Y_coord_map,axis=1),axis=1)/norma
        return centroid_out

    def generate_pseudo_reference_img(self, imgs_patch):
        # The selection is done with the criteria specified
        pseudo_reference_img = np.zeros_like(imgs_patch)
        ind_sel = 0
        if self.pattern_criteria == 'contrast':
            contrast = np.std(imgs_patch, axis=(1,2))
            ind_sel = np.argmax(contrast)
        else:
            ind_sel = self.subaperture_sel[0]*self.nSubap+self.subaperture_sel[1]

        pseudo_reference_img = np.tile(imgs_patch[ind_sel, : ,:], (pseudo_reference_img.shape[0], 1, 1))

        return pseudo_reference_img
    #%% DIFFRACTIVE

    def initialize_flux(self,input_flux_map = None):
        cx = self.center_init - self.n_pix_subap_init//2
        cy = self.center_init - self.n_pix_subap_init//2
        if self.telescope.src.tag == 'source':
            if input_flux_map is None:
                input_flux_map = self.telescope.src.fluxMap.T
            tmp_flux_h_split = np.hsplit(input_flux_map,self.nSubap)
            self.cube_flux = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init],dtype=float)
            for i in range(self.nSubap):
                tmp_flux_v_split = np.vsplit(tmp_flux_h_split[i],self.nSubap)
                self.cube_flux[i*self.nSubap:(i+1)*self.nSubap,cx:cx+self.n_pix_subap_init,cy:cy+self.n_pix_subap_init] = np.asarray(tmp_flux_v_split)
            self.photon_per_subaperture = np.apply_over_axes(np.sum, self.cube_flux, [1,2])
            self.current_nPhoton = self.telescope.src.nPhoton
        elif self.telescope.src.tag == 'sun':
            if input_flux_map is None:
                input_flux_map = np.zeros_like(self.telescope.src.sun_subDir_ast.src[0].fluxMap.T)
                for i in range(self.telescope.src.nSubDirs**2):
                    input_flux_map += self.telescope.src.sun_subDir_ast.src[i].fluxMap.T
            tmp_flux_h_split = np.hsplit(input_flux_map,self.nSubap)
            self.cube_flux = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init],dtype=float)
            for i in range(self.nSubap):
                tmp_flux_v_split = np.asarray(np.vsplit(tmp_flux_h_split[i],self.nSubap))
                for j in range(self.nSubap):
                    self.cube_flux[i*self.nSubap+j, cx:cx+self.n_pix_subap_init,cy:cy+self.n_pix_subap_init] = cv2.resize(tmp_flux_v_split[j,:,:], (self.n_pix_subap_init, self.n_pix_subap_init), interpolation=cv2.INTER_CUBIC)

            self.photon_per_subaperture = np.apply_over_axes(np.sum, self.cube_flux, [1,2])
            self.current_nPhoton = self.telescope.src.nPhoton 
        return
  
    def fill_raw_data(self,ind_x,ind_y,I,type='raw', index_frame=None):
        cx = (2*ind_x+1)*self.raw_img_spacing + ind_x*I.shape[0]
        cy = (2*ind_y+1)*self.raw_img_spacing + ind_y*I.shape[1]

        if index_frame is None:
            if type == 'raw':
                self.raw_data[cx:cx+I.shape[0], cy:cy+I.shape[1]] = I
            else:
                self.corr_raw_data[cx:cx+I.shape[0], cy:cy+I.shape[1]] = I        
        else:
            if type == 'raw':
                self.raw_data[index_frame,cx:cx+I.shape[0], cy:cy+I.shape[1]] = I
            else:
                self.corr_raw_data[cx:cx+I.shape[0], cy:cy+I.shape[1]] = I        

    def split_raw_data(self):
        raw_data_h_split = np.vsplit((self.cam.frame),self.nSubap)
        self.maps_intensity = np.zeros([self.nSubap**2,self.n_pix_subap,self.n_pix_subap],dtype=float)
        center = self.n_pix_subap//2
        for i in range(self.nSubap):
            raw_data_v_split = np.hsplit(raw_data_h_split[i],self.nSubap)
            self.maps_intensity[i*self.nSubap:(i+1)*self.nSubap,center - self.n_pix_subap//2:center+self.n_pix_subap//2,center - self.n_pix_subap//2:center+self.n_pix_subap//2] = np.asarray(raw_data_v_split)
        self.maps_intensity = self.maps_intensity[self.valid_subapertures_1D,:,:]
    
#%% SH Measurement 
    
    def wfs_measure(self):
        
        if self.current_nPhoton != self.telescope.src.nPhoton:
            print('updating the flux of the SHWFS object')
            self.initialize_flux()
   
        ##%%%%%%%%%%%%  DIFFRACTIVE SH WFS %%%%%%%%%%%%
        if self.fov_crop is None:
            self.raw_data           = np.zeros([(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap], dtype =float)
            self.corr_raw_data      = np.zeros([(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap], dtype =float)
        else:
            self.raw_data           = np.zeros([(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap], dtype =float)
            self.corr_raw_data      = np.zeros([(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap], dtype =float)

        # normalization for FFT
        norma = self.cube.shape[1]

        # compute spot intensity
        if self.telescope.spatialFilter is None:   
            self.initialize_flux()

        else:  
            self.initialize_flux(((self.telescope.amplitude_filtered)**2).T*self.telescope.src.fluxMap.T)
        
        # Compute WFS image, only valid subapertures
        self.wfs_img = self.get_subap_img(True)
        # add photon/readout noise to 2D spots
        if self.cam.photonNoise!=0:
            self.wfs_img  = self.random_state_photon_noise.poisson(self.wfs_img)
                
        if self.cam.readoutNoise!=0:
            self.wfs_img += np.int64(np.round(self.random_state_readout_noise.randn(self.wfs_img.shape[0],self.wfs_img.shape[1],self.wfs_img.shape[2])*self.cam.readoutNoise))
                     
        # reset camera frame

        if self.fov_crop is None:
            self.raw_data           = np.zeros([(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap], dtype =float)
            self.corr_raw_data      = np.zeros([(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_subap+2*self.raw_img_spacing)*self.nSubap], dtype =float)
        else:
            self.raw_data           = np.zeros([(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap], dtype =float)
            self.corr_raw_data      = np.zeros([(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap,(self.n_pix_fov_crop+2*self.raw_img_spacing)*self.nSubap], dtype =float)     
        
        # bin the 2D spots arrays
        self.corrImg = np.zeros_like(self.wfs_img)
        
        # compute correlation and normalize the energy to 1

        # print("WFS Measure: ", self.wfs_img.shape, self.pseudo_ref_img.shape)

        self.corrImg = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(self.wfs_img, axes=(1,2)) * np.conj(np.fft.fft2(self.pseudo_ref_img, axes=(1,2))), axes=(1,2)), axes=(1,2)))

        self.corrImg = self.corrImg / (self.corrImg.shape[1]**2)

        # take N brightest

        mask_max = np.zeros_like(self.corrImg)
        
        for i in range(self.corrImg.shape[0]):
            # get mask index
            max_index = np.unravel_index(np.argmax(self.corrImg[i], axis=None), self.corrImg[i].shape)
            mask = np.zeros_like(self.corrImg[i], dtype=int)
            mask[max_index] = 1  # Select maximum
            # Apply the kernel through convolution
            mask_max[i] = convolve(mask, self.max_kernel, mode='constant')
        mask_max[mask_max == 0] = np.nan

        # Apply the mask to self.corrImg
        self.corrImg *= mask_max

        # Normalize the 2-D matrices
        tmp_max = np.nanmax(self.corrImg, axis=(1, 2), keepdims=True)
        tmp_min = np.nanmin(self.corrImg, axis=(1, 2), keepdims=True)
        self.corrImg[np.isnan(self.corrImg)]  = 0
        self.corrImg = np.where(self.corrImg >= tmp_min, (self.corrImg - tmp_min) / (tmp_max - tmp_min), 0)

        # centroid computation
        self.centroid_lenslets = self.centroid(self.corrImg)

        # re-organization of signals according to number of wavefronts considered
        self.signal_2D = np.zeros([self.nSubap*2,self.nSubap])
        
        self.SX[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[:,0]
        self.SY[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[:,1]
        signal_2D = np.concatenate((self.SX,self.SY)) - self.references_2D - (self.corrImg.shape[1]/2) # we set the origin at the center of the subap
        signal_2D[~self.valid_slopes_maps] = 0
        self.signal_2D = signal_2D

        self.signal = self.signal_2D[self.valid_slopes_maps].T
        #plt.plot(self.signal), plt.show()

        # Save raw frame
        index_valid = 0
        for x in range(self.nSubap):
            for y in range(self.nSubap):
                if self.valid_subapertures[x,y]:
                    self.fill_raw_data(x, y, self.corrImg[index_valid, :, :], 'corr')
                    self.fill_raw_data(x, y, self.wfs_img[index_valid, :, :], 'raw')
                    index_valid += 1
        self*self.cam

    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHACK HARTMANN WFS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^20s}'.format('Subapertures')         + '{: ^18s}'.format(str(self.nSubap))                                   )
        print('{: ^20s}'.format('Subaperture Size')     + '{: ^18s}'.format(str(np.round(self.telescope.D/self.nSubap,2)))      +'{: ^18s}'.format('[m]'   ))
        print('{: ^20s}'.format('Pixel FoV')            + '{: ^18s}'.format(str(np.round(self.fov_pixel_binned_arcsec,2)))      +'{: ^18s}'.format('[arcsec]'   ))
        print('{: ^20s}'.format('Subapertue FoV')       + '{: ^18s}'.format(str(np.round(self.fov_lenslet_arcsec,2)))           +'{: ^18s}'.format('[arcsec]'  ))
        print('{: ^20s}'.format('Valid Subaperture')    + '{: ^18s}'.format(str(str(self.nValidSubaperture))))                   


        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS INTERACTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj._integrated_time+=self.telescope.samplingTime
            obj.integrate(self.raw_data)
            # self.raw_data = obj.frame
        else:
            print('Error light propagated to the wrong type of object')
        return -1
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag+':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    if not np.shape(a[1]):
                        tmp=a[1]
                        try:
                            print('          '+str(a[0])+': '+str(tmp.tag)+' object') 
                        except:
                            print('          '+str(a[0])+': '+str(a[1])) 
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+': '+str(np.shape(a[1])))   
    
            
    def __repr__(self):
        self.print_properties()
        return ' '
