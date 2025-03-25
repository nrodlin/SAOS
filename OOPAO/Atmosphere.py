# -*- coding: utf-8 -*-
"""
Originally created on Fri Aug 14 10:59:02 2020
@author: cheritie

Major update on March 24 2025
@author: nrodlin
"""

import time

import logging
import logging.handlers
from queue import Queue

from astropy.io import fits

from joblib import Parallel, delayed

import numpy as np
from numpy.random import RandomState
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Self dependencies
from .phaseStats import ft_phase_screen, ft_sh_phase_screen, makeCovarianceMatrix
from .tools.displayTools import makeSquareAxes
from .tools.interpolateGeometricalTransformation import interpolate_cube
from .tools.tools import globalTransformation, pol2cart, translationImageMatrix
from .Layer import LayerClass


"""
Atmosphere Module
=================

This module contains the `Atmosphere` class, used for modeling a multi-layer turbulent atmosphere in adaptive optics simulations.
"""

class Atmosphere:
    """
    The `Atmosphere` class represents a multi-layer atmosphere with Von Kármán statistics.

    Each layer is defined with altitude, wind speed/direction, and Cn2 contribution. Layers can be updated, saved to disk,
    and used to compute the OPD (Optical Path Difference) for various line-of-sight sources.

    Attributes:
        r0 (float): Fried parameter at 500 nm.
        L0 (float): Outer scale of turbulence.
        windSpeed (List[float]): Wind speeds per layer.
        windDirection (List[float]): Wind directions per layer.
        altitude (List[float]): Altitudes of each layer.
        fractionalR0 (List[float]): Relative contribution of each layer to turbulence.
    """
    def __init__(self,
                 r0:float,
                 L0:float,
                 windSpeed:list,
                 fractionalR0:list,
                 windDirection:list,
                 altitude:list,
                 mode:float=1,
                 logger=None):
        """
        Initialize an Atmosphere object representing layered atmospheric turbulence.

        Parameters
        ----------
        r0 : float
            Fried parameter at 500 nm [m].
        L0 : float
            Outer scale of turbulence [m].
        windSpeed : list of float
            Wind speed for each layer [m/s].
        fractionalR0 : list of float
            Cn2 profile; fractional contribution to turbulence per layer.
        windDirection : list of float
            Wind direction for each layer [degrees].
        altitude : list of float
            Altitude of each layer [m].
        mode : float, optional
            Method for computing phase screen (1 = with subharmonics), by default 1.
        logger : logging.Logger, optional
            Logger instance for this object, by default None.
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        self.hasNotBeenInitialized  = True
        self.r0_def                 = 0.15              # Fried Parameter in m 
        self.r0                     = r0                # Fried Parameter in m 
        self.fractionalR0           = fractionalR0      # CFractional Cn2 profile of the turbulence
        self.L0                     = L0                # Outer Scale in m
        self.altitude               = altitude          # altitude of the layers
        self.nLayer                 = len(fractionalR0) # number of layer
        self.windSpeed              = windSpeed         # wind speed of the layers in m/s
        self.windDirection          = windDirection     # wind direction in degrees
        self.tag                    = 'atmosphere'      # Tag of the object
        self.nExtra                 = 2                 # number of extra pixel to generate the phase screens
        self.wavelength             = 500*1e-9          # Wavelengt used to define the properties of the atmosphere
        self.cn2 = (self.r0**(-5. / 3) / (0.423 * (2*np.pi/self.wavelength)**2))/np.max([1, np.max(self.altitude)]) # Cn2 m^(-2/3) [unused]
        self.seeingArcsec           = 206265*(self.wavelength/self.r0)
        self.mode = mode
        
    def initializeAtmosphere(self,telescope,compute_covariance =True, randomState=None):
        """
        Initialize the atmosphere layers and associate them with a telescope.

        Parameters
        ----------
        telescope : Telescope or None
            Telescope object to derive spatial and temporal resolution.
        compute_covariance : bool, optional
            Whether to compute covariance matrices, by default True.
        randomState : int or None
            Seed for reproducible random number generation, by default None.

        Returns
        -------
        bool
            True if initialization succeeded, False otherwise.
        """
        self.logger.debug('Atmosphere::initializeAtmosphere')
        self.compute_covariance = compute_covariance

        if telescope is not None:
            self.logger.info('Atmosphere::initializeAtmosphere - Taking key parameters from the telescope.')
            self.resolution = telescope.resolution
            self.D = telescope.D
            self.samplingTime = telescope.samplingTime

            self.fov      = telescope.fov
            self.fov_rad  = telescope.fov_rad
            from_file = False
        else:
            if (self.resolution is None) or (self.D is None) or (self.samplingTime is None) or (self.fov is None)\
                or (self.fov_rad is None) or (randomState is None):

                self.logger.error('Atmosphere::initializeAtmosphere - It seems that you did not provide a telescope object and not all the necessary\
                                  attributes are defined.')
                return False
            else:
                from_file = True

        if self.hasNotBeenInitialized:
            self.initial_r0 = self.r0
            self.logger.info('Atmosphere::initializeAtmosphere - Creating layers...')
            results_layers = Parallel(n_jobs=self.nLayer, prefer="threads")(delayed(self.buildLayer)(i_layer, randomState, from_file) for i_layer in range(self.nLayer))
            for i_layer in range(self.nLayer):
                setattr(self,'layer_'+str(i_layer+1),results_layers[i_layer])
        else:
            self.logger.warning('Atmosphere::initializeAtmosphere - The atmosphere has already been initialized.')
            return True
        
        self.hasNotBeenInitialized  = False 
        self.print_properties()
        return True
            
    def buildLayer(self, i_layer, randomState = None, from_file=False):
        """
        Build and initialize a single atmospheric layer.

        Parameters
        ----------
        i_layer : int
            Index of the layer to build.
        randomState : int, optional
            Seed for the random number generator.
        from_file : bool, optional
            If True, reuse existing layer data loaded from file.

        Returns
        -------
        LayerClass
            The initialized atmospheric layer.
        """
        self.logger.debug('Atmosphere::buildLayer - layer '+str(i_layer+1))
         
        # initialize layer object
        if from_file:
            layer = getattr(self,'layer_'+str(i_layer+1))
        else:
            layer               = LayerClass()
        layer.id            = i_layer
        # create a random state to allow reproductible sequences of phase screens
        if randomState is not None:
            seed = randomState
        else:
            t = time.localtime()
            seed = t.tm_hour*3600 + t.tm_min*60 + t.tm_sec
        
        layer.seed          = seed
        layer.randomState   = RandomState(seed+i_layer*1000)
        
        # gather properties of the atmosphere
        layer.altitude      = self.altitude[i_layer]       
        layer.windSpeed     = self.windSpeed[i_layer]
        layer.direction     = self.windDirection[i_layer]
        layer.fov_rad       = self.fov_rad
        layer.fractionalR0  = self.fractionalR0[i_layer]
        
        # compute the X and Y wind speed
        layer.vY            = layer.windSpeed*np.cos(np.deg2rad(layer.direction))
        layer.vX            = layer.windSpeed*np.sin(np.deg2rad(layer.direction))      
        layer.extra_sx      = 0
        layer.extra_sy      = 0
        
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        
        layer.D_fov             = self.D+2*np.tan(layer.fov_rad/2)*layer.altitude
        layer.resolution_fov    = int(np.ceil((self.resolution/self.D)*layer.D_fov))
        layer.resolution        = layer.resolution_fov + 4 # 4 pixels are added as a margin for the edges
        layer.D                 = layer.resolution * self.D/ self.resolution # in px
        # layer pixel size
        layer.d0 = layer.D/layer.resolution        
        # number of pixel for the phase screens computation
        layer.nExtra        = self.nExtra
        layer.nPixel        = int(1+np.round(layer.D/layer.d0))
        self.logger.info('Atmosphere::buildLayer - Creating layer '+str(i_layer+1))    
        a=time.time()
        if self.mode == 1:  # with subharmonics
            layer.phase        = ft_sh_phase_screen(self, layer.resolution, layer.D/layer.resolution, seed=seed)
        else: 
            layer.phase        = ft_phase_screen(self, layer.resolution, layer.D/layer.resolution, seed=seed)                    
        layer.initialPhase = layer.phase.copy()
        b=time.time()
        self.logger.info('Atmosphere::buildLayer - elapsed time to generate the phase screen : ' +str(b-a) +' s')
        
        # Outer ring of pixel for the phase screens update 
        layer.outerMask             = np.ones([layer.resolution+layer.nExtra,layer.resolution+layer.nExtra])
        layer.outerMask[1:-1,1:-1]  = 0
        
        # inner pixels that contains the phase screens
        layer.innerMask             = np.ones([layer.resolution+layer.nExtra,layer.resolution+layer.nExtra])
        layer.innerMask -= layer.outerMask
        layer.innerMask[1+layer.nExtra:-1-layer.nExtra,1+layer.nExtra:-1-layer.nExtra] = 0
        
        l = np.linspace(0,layer.resolution+1,layer.resolution+2) * layer.D/(layer.resolution-1)
        u,v = np.meshgrid(l,l)
        
        layer.innerZ = u[layer.innerMask!=0] + 1j*v[layer.innerMask!=0]
        layer.outerZ = u[layer.outerMask!=0] + 1j*v[layer.outerMask!=0]
        if self.compute_covariance:

            self.get_covariance_matrices(layer) # Computes: XXt_r0, ZXt_r0, ZZt_r0, ZZt_inv, ZZt_inv_r0 an their non r0 counter-part                                           
            
        layer.A         = np.matmul(layer.ZXt_r0.T,layer.ZZt_inv_r0)    
        layer.BBt       = layer.XXt_r0 -  np.matmul(layer.A,layer.ZXt_r0)
        layer.B         = np.linalg.cholesky(layer.BBt)
        layer.mapShift  = np.zeros([layer.nPixel+1,layer.nPixel+1])        
        Z               = layer.phase[layer.innerMask[1:-1,1:-1]!=0]
        X               = np.matmul(layer.A,Z) + np.matmul(layer.B,layer.randomState.normal(size=layer.B.shape[1]))
        
        layer.mapShift[layer.outerMask!=0] = X
        layer.mapShift[layer.outerMask==0] = np.reshape(layer.phase,layer.resolution*layer.resolution)
        layer.notDoneOnce                  = True

        self.logger.info('Atmosphere::buildLayer - Layer '+str(i_layer+1)+' created.')      
    
        return layer
    
    def get_covariance_matrices(self, layer):
        """
        Compute the covariance matrices (Assemat et al. 2006) for a layer.

        Parameters
        ----------
        layer : LayerClass
            The atmospheric layer for which to compute matrices.

        Returns
        -------
        bool
            True upon successful computation.
        """
        self.logger.debug('Atmosphere::get_covariance_matrices')
        # Compute the covariance matrices - Implements the paper from Assemat et al. (2006)
        self.logger.debug('Atmosphere::get_covariance_matrices')
        c=time.time()        
        layer.ZZt = makeCovarianceMatrix(layer.innerZ,layer.innerZ,self)
        layer.ZZt_inv = np.linalg.pinv(layer.ZZt)

        d=time.time()
        self.logger.info('Atmosphere::get_covariance_matrices - Layer ' + str(layer.id) + ' ZZt.. : ' +str(d-c) +' s')
        layer.ZXt = makeCovarianceMatrix(layer.innerZ,layer.outerZ,self)
        e=time.time()
        self.logger.info('Atmosphere::get_covariance_matrices - Layer ' + str(layer.id) + ' ZXt.. : ' +str(e-d) +' s')
        layer.XXt = makeCovarianceMatrix(layer.outerZ,layer.outerZ,self)
        f=time.time()
        self.logger.info('Atmosphere::get_covariance_matrices - Layer ' + str(layer.id) + ' XXt.. : ' +str(f-e) +' s')
        
        layer.ZZt_r0     = layer.ZZt*(self.r0_def/self.r0)**(5/3)
        layer.ZXt_r0     = layer.ZXt*(self.r0_def/self.r0)**(5/3)
        layer.XXt_r0     = layer.XXt*(self.r0_def/self.r0)**(5/3)
        layer.ZZt_inv_r0 = layer.ZZt_inv/((self.r0_def/self.r0)**(5/3))
        return True
    
    def save(self, filename=None):
        """
        Save the state of the atmosphere to a FITS file.

        Parameters
        ----------
        filename : str
            Path and base filename (without extension) to save the FITS file.

        Returns
        -------
        bool
            True if saved successfully, False otherwise.
        """
        self.logger.debug('Atmosphere::save')

        if self.hasNotBeenInitialized:
            self.logger.error('Atmosphere::save - The atmosphere has not been initialized yet.')
            return False

        self.logger.info('Atmosphere::save - Creating the HDU')
        
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['nLayers'] = self.nLayer
        primary_hdu.header['FOV']     = self.fov
        primary_hdu.header['FOVRAD']  = self.fov_rad
        primary_hdu.header['RES']     = self.resolution
        primary_hdu.header['D']       = self.D
        primary_hdu.header['TS']      = self.samplingTime
        primary_hdu.header['SEED']    = self.layer_1.seed

        ps_hdu_list = []
        ps_hdu_list.append(primary_hdu)

        for i_layer in range(self.nLayer):
            layer = getattr(self,'layer_'+str(i_layer+1))
            
            ps_hdu_list[0].header['L'+str(layer.id) + 'alt']    = layer.altitude
            ps_hdu_list[0].header['L'+str(layer.id) + 'wSp']   = layer.windSpeed
            ps_hdu_list[0].header['L'+str(layer.id) + 'wDir']   = layer.direction

            ps_hdu_list.append(fits.ImageHDU(layer.XXt, name='L'+str(layer.id) + 'XXt'))
            ps_hdu_list.append(fits.ImageHDU(layer.XXt_r0, name='L'+str(layer.id) + 'XXtr0'))
            ps_hdu_list.append(fits.ImageHDU(layer.ZXt, name='L'+str(layer.id) + 'ZXt'))
            ps_hdu_list.append(fits.ImageHDU(layer.ZXt_r0, name='L'+str(layer.id) + 'ZXtr0'))
            ps_hdu_list.append(fits.ImageHDU(layer.ZZt, name='L'+str(layer.id) + 'ZZt'))
            ps_hdu_list.append(fits.ImageHDU(layer.ZZt_inv, name='L'+str(layer.id) + 'ZZtI'))
            ps_hdu_list.append(fits.ImageHDU(layer.ZZt_inv_r0, name='L'+str(layer.id) + 'ZZtI0'))
            ps_hdu_list.append(fits.ImageHDU(layer.ZZt_r0, name='L'+str(layer.id) + 'ZZtr0'))

        
        self.logger.info('Atmosphere::save - Writting...')
        hdul = fits.HDUList(ps_hdu_list)
        hdul.writeto(filename + '.fits', overwrite=True)
        self.logger.info('Atmosphere::save - Saved.')
    
    def load(self, filename): 
        """
        Load an atmosphere configuration from a FITS file.

        Parameters
        ----------
        filename : str
            Path and base filename (without extension) of the FITS file to load.

        Returns
        -------
        bool
            True if loaded successfully, False otherwise.
        """
        self.logger.debug('Atmosphere::load')

        with fits.open(filename + '.fits') as hdul:
            
            self.nLayer = hdul[0].header['NLAYERS']

            self.fov           = hdul[0].header['FOV']
            self.fov_rad       = hdul[0].header['FOVRAD']
            self.resolution    = hdul[0].header['RES']
            self.D             = hdul[0].header['D']
            self.samplingTime  = hdul[0].header['TS']

            seed               = hdul[0].header['SEED']

            altitude = []
            windSpeed = []
            windDirection = []

            for i_layer in range(self.nLayer):
                layer = LayerClass()
                self.logger.info(f"Atmosphere::load - Loading layer {i_layer} base data")
                layer.id = i_layer

                altitude.append(hdul[0].header[f"L{i_layer}ALT"])
                windSpeed.append(hdul[0].header[f"L{i_layer}WSP"])
                windDirection.append(hdul[0].header[f"L{i_layer}WDIR"])
               
                layer.seed       = seed

                layer.XXt        = hdul[f"L{i_layer}XXT"].data
                layer.XXt_r0     = hdul[f"L{i_layer}XXTR0"].data
                layer.ZXt        = hdul[f"L{i_layer}ZXT"].data
                layer.ZXt_r0      = hdul[f"L{i_layer}ZXTR0"].data
                layer.ZZt        = hdul[f"L{i_layer}ZZT"].data
                layer.ZZt_inv    = hdul[f"L{i_layer}ZZTI"].data
                layer.ZZt_inv_r0 = hdul[f"L{i_layer}ZZTI0"].data
                layer.ZZt_r0     = hdul[f"L{i_layer}ZZTR0"].data

                setattr(self,'layer_'+str(i_layer+1),layer)

            self.altitude      = altitude.copy()
            self.windSpeed     = windSpeed.copy()
            self.windDirection = windDirection.copy()

            self.hasNotBeenInitialized = True

            self.logger.info(f"Atmosphere::load - Computing the remaining parameters.") 
            out = self.initializeAtmosphere(telescope=None, compute_covariance=False, randomState=seed)

            if out:
                self.logger.info(f"Atmosphere::load - All layers finished. You should check that your telescope matches this atmosphere using Atmosphere::matchParameters()")     
            else:
                self.logger.error(f"Atmosphere::load - Could not initialize the atmosphere, check the input file content and formatting.")     
    
    def matchParameters(self, telescope):
        """
        Check if telescope parameters match the atmosphere configuration.

        Parameters
        ----------
        telescope : Telescope
            Telescope object to compare with.

        Returns
        -------
        bool
            True if parameters match, False otherwise.
        """
        self.logger.info('Atmosphere::matchParameters - Checking parameters')
        if self.hasNotBeenInitialized:
            self.logger.error('Atmosphere::matchParameters - The atmosphere is not initialized.')
            return False

        comp_tol = 1e-5

        if not math.isclose(telescope.resolution, self.resolution, rel_tol=comp_tol):
            self.logger.warning('Atmosphere::matchParameters - Resolution does not match.')
            return False
        
        if not math.isclose(telescope.samplingTime, self.samplingTime, rel_tol=comp_tol):
            self.logger.warning('Atmosphere::matchParameters - Sampling Time does not match.')
            return False

        if not math.isclose(telescope.D, self.D, rel_tol=comp_tol):
            self.logger.warning('Atmosphere::matchParameters - Diameter does not match.')
            return False

        if not math.isclose(telescope.fov, self.fov, rel_tol=comp_tol):
            self.logger.warning('Atmosphere::matchParameters - Field of View (FoV) does not match.')
            return False

        if not math.isclose(telescope.fov_rad, self.fov_rad, rel_tol=comp_tol):
            self.logger.warning('Atmosphere::matchParameters - Field of View (FoV) in [rad] does not match.')
            return False
        
        self.logger.info('Atmosphere::matchParameters - All the parameters between both objects match.')
        
        return True

    def update(self):
        """
        Update all atmospheric layers based on wind and time step.

        Returns
        -------
        bool
            True if update was successful.
        """
        self.logger.debug('Atmosphere::update')
        # Update each layer at a different process to speed up the computation.
        # Threads are faster than multiprocessing due to the overheads of making the copies. 
        results = Parallel(n_jobs=self.nLayer, prefer="threads")(
            delayed(self.updateLayer)(getattr(self,'layer_'+str(i_layer+1))) for i_layer in range(self.nLayer))
        for i_layer in range(self.nLayer):
            setattr(self,'layer_'+str(i_layer+1),results[i_layer])
        
        # Serialize layers into shared memory before the getOPD method call

        self.logger.debug('Atmosphere::update - Updated.')
        return True
        
    def updateLayer(self,updatedLayer,shift = None):
        """
        Update a single atmospheric layer, shifting the phase screen.

        Parameters
        ----------
        updatedLayer : LayerClass
            The layer to update.
        shift : list of float, optional
            Optional shift to apply in pixels [x, y].

        Returns
        -------
        LayerClass
            The updated layer.
        """
        self.logger.debug('Atmosphere::updateLayer')
        ps_loop         = updatedLayer.D / (updatedLayer.resolution)
        ps_turb_x       = updatedLayer.vX*self.samplingTime 
        ps_turb_y       = updatedLayer.vY*self.samplingTime 
        
        if not((updatedLayer.vX==0) and (updatedLayer.vY==0) and (shift is None)):
            if updatedLayer.notDoneOnce:
                updatedLayer.notDoneOnce = False
                updatedLayer.ratio = np.zeros(2)
                updatedLayer.ratio[0] = ps_turb_x/ps_loop  
                updatedLayer.ratio[1] = ps_turb_y/ps_loop
                updatedLayer.buff = np.zeros(2)
            
            if shift is None:
                ratio = updatedLayer.ratio
            else:
                 ratio = shift    # shift in pixels
            
            tmpRatio = np.abs(ratio)
            tmpRatio[np.isinf(tmpRatio)]=0
            nScreens = (tmpRatio)
            nScreens = nScreens.astype('int')
            
            stepInPixel =np.zeros(2)
            stepInSubPixel =np.zeros(2)
            
            for i in range(nScreens.min()):
                stepInPixel[0]=1
                stepInPixel[1]=1
                stepInPixel=stepInPixel*np.sign(ratio)
                updatedLayer.phase = self.add_row(updatedLayer,stepInPixel)
                
            for j in range(nScreens.max()-nScreens.min()):   
                stepInPixel[0]=1
                stepInPixel[1]=1
                stepInPixel=stepInPixel*np.sign(ratio)
                stepInPixel[np.where(nScreens==nScreens.min())]=0
                updatedLayer.phase = self.add_row(updatedLayer,stepInPixel)
            
            
            stepInSubPixel[0] =  (np.abs(ratio[0])%1)*np.sign(ratio[0])
            stepInSubPixel[1] =  (np.abs(ratio[1])%1)*np.sign(ratio[1])
            
            updatedLayer.buff += stepInSubPixel
            if np.abs(updatedLayer.buff[0])>=1 or np.abs(updatedLayer.buff[1])>=1:   
                stepInPixel[0] = 1*np.sign(updatedLayer.buff[0])
                stepInPixel[1] = 1*np.sign(updatedLayer.buff[1])
                stepInPixel[np.where(np.abs(updatedLayer.buff)<1)]=0    
                
                updatedLayer.phase = self.add_row(updatedLayer,stepInPixel)
    
            updatedLayer.buff[0]   =  (np.abs(updatedLayer.buff[0])%1)*np.sign(updatedLayer.buff[0])
            updatedLayer.buff[1]   =  (np.abs(updatedLayer.buff[1])%1)*np.sign(updatedLayer.buff[1])
                
            shiftMatrix            = translationImageMatrix(updatedLayer.mapShift,[updatedLayer.buff[0],updatedLayer.buff[1]]) # units are in pixel of the M1            
            updatedLayer.phase     = globalTransformation(updatedLayer.mapShift,shiftMatrix)[1:-1,1:-1]

        return updatedLayer

    def add_row(self,layer,stepInPixel,map_full = None):
        """
        Shift the phase screen of a layer by one or more pixels.

        Parameters
        ----------
        layer : LayerClass
            Atmospheric layer to update.
        stepInPixel : list of int
            Pixel shift [dx, dy] to apply.
        map_full : np.ndarray, optional
            Full phase map to use; if None, uses layer.mapShift.

        Returns
        -------
        np.ndarray
            Updated one-pixel-shifted phase screen.
        """
        self.logger.debug('Atmosphere::add_row')
        if map_full is None:
            map_full = layer.mapShift
        shiftMatrix                         = translationImageMatrix(map_full,[stepInPixel[0],stepInPixel[1]]) #units are in pixel of the M1            
        tmp                                 = globalTransformation(map_full,shiftMatrix)
        onePixelShiftedPhaseScreen          = tmp[1:-1,1:-1]        
        Z                                   = onePixelShiftedPhaseScreen[layer.innerMask[1:-1,1:-1]!=0]
        X                                   = layer.A@Z + layer.B@layer.randomState.normal(size=layer.B.shape[1])
        map_full[layer.outerMask!=0]  = X
        map_full[layer.outerMask==0]  = np.reshape(onePixelShiftedPhaseScreen,layer.resolution*layer.resolution)
        return onePixelShiftedPhaseScreen 

    def getOPD(self, src):
        """
        Compute the Optical Path Difference (OPD) for a given source.

        Parameters
        ----------
        src : Source or ExtendedSource
            The light source for which to compute the OPD.

        Returns
        -------
        np.ndarray
            OPD values in meters (or list of OPDs if multiple sources).
        """
        self.logger.debug('Atmosphere::getOPD - Getting the OPD for each source.')

        # First, we need to check the source tbecause the sun is made of an asterism, 
        # it will be more efficient to run in parallel the process for each individual star and then combine it.

        list_src = []

        if src.tag == 'sun':
            for subDir in src.sun_subDir_ast.src:
                list_src.append(subDir)
        else:
            list_src.append(src)

        # Then, we get the footprint for each element of the list -> we do this in parallel.
        # result_footprint contains a list of nSources, each element containing a tuple of two list of size nLayers. The first list contains the footprint per layer, 
        # and the second list the offset of the centroid due to discretization

        result_footprint = Parallel(n_jobs=len(list_src), prefer="threads")(delayed(self.get_pupil_footprint)(list_src[i]) for i in range(len(list_src)))

        # Once the pupil is defined, we should use it to get the phase
        # result_phase contains a list of nSources, each element containing a list of size nLayers, whose elements contain the phase per layer [in rad]
        result_phase = Parallel(n_jobs=len(list_src), prefer="threads")(delayed(self.project_phase)(
                                list_src[i], result_footprint[i][0], result_footprint[i][1]) for i in range(len(list_src)))

        # Finally, the phases are merged to get the resulting OPD per line of sight
        # result_opd_no_pupil contains one list of size nSources containing the OPD without pupil per source. 
        # The OPD is in [meters]
        result_opd_no_pupil = Parallel(n_jobs=len(list_src), prefer="threads")(delayed(self.get_opd_per_src)(list_src[i], result_phase[i]) for i in range(len(list_src)))

        return np.squeeze(result_opd_no_pupil)
           
    def get_pupil_footprint(self, src):
        """
        Determine the pupil footprint and discretization offset for a source.

        Parameters
        ----------
        src : Source
            Light source object.

        Returns
        -------
        Tuple[list, list]
            A tuple of two lists: pupil footprint per layer and center offset.
        """
        self.logger.debug('Atmosphere::set_pupil_footprint')
        footprint_per_layer = []
        extra_s = []

        for i_layer in range(self.nLayer):
            layer = getattr(self,'layer_'+str(i_layer+1))

            # If defined, chromatic shift shall be define per layer. We need to check its format
            if src.chromatic_shift is None:
                chromatic_shift = 0
            else:
                if len(src.chromatic_shift) != self.nLayer:
                    raise ValueError('Atmosphere::get_pupil_footprint - If defined, the chromatic shift shall be defined per layer.')
                chromatic_shift = src.chromatic_shift[i_layer]  

            # The source coordinates are defined as: [elevation, azimuth] taking as reference the Zenith, i.e. [0,x] has the telescope pointing to the sky vertically.
            # Elevation is measured in arcsec, azimuth in degrees.
            # We need to compute the location of the source w.r.t zenith to define the pupil centered at the source
            # The polar coordinates, at an altiudue z_i are: r = z_i*tan(elevation), theta = azimuth. Then, we convert to cartesian coordinates.
            # Note that before changing coordinates, the radius in meters is converted to px using the layer diameter (layer.D) and phase screen size (layer.resolution)
            [x_z,y_z] = pol2cart(layer.altitude*np.tan((src.coordinates[0]+chromatic_shift)/206265) * layer.resolution / layer.D, np.deg2rad(src.coordinates[1]))
            # [x_z, y_z] are the cartesian coordinates in px w.r.t zenith. 
            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2

            # As the coordinates are discretized in pixels, there may be an offset. We need to pass this offset when the phase is projected,
            # so we store it and return it to the main function
            extra_s.append([int(x_z)-x_z, int(y_z)-y_z])
    
            # Finally, create the pupil and set the points inside the pupil to 1
            pupil_footprint = np.zeros([layer.resolution,layer.resolution])
            pupil_footprint[center_x-self.resolution//2:center_x+self.resolution//2,
                            center_y-self.resolution//2:center_y+self.resolution//2 ] = 1
            
            footprint_per_layer.append(pupil_footprint)
        return footprint_per_layer, extra_s
            
    def project_phase(self, src, pupil_footprint, extra_s):
        """
        Project atmospheric phase through layers for a given source.

        Parameters
        ----------
        src : Source
            Light source object.
        pupil_footprint : list of np.ndarray
            Pupil masks per layer.
        extra_s : list
            Offset due to discretization for each layer.

        Returns
        -------
        list
            List of phase screens per layer.
        """
        self.logger.debug('Atmosphere::fill_phase_support')

        # Each layer shall run in parallel to reduce time consumption. Returns a list containing the phase per layer for the source line of sight
        result_phase = Parallel(n_jobs=1, prefer="threads")(
            delayed(self.get_phase_layered)(src, getattr(self,'layer_'+str(i_layer+1)), pupil_footprint[i_layer], extra_s[i_layer]) for i_layer in range(self.nLayer))
        
        return result_phase

    def get_phase_layered(self, src, layer, pupil_footprint, extra_s):
        """
        Compute phase at a given atmospheric layer for a specific source.

        Parameters
        ----------
        src : Source
            Light source.
        layer : LayerClass
            Atmospheric layer.
        pupil_footprint : np.ndarray
            Pupil mask for the layer.
        extra_s : list
            Offset due to discretization.

        Returns
        -------
        np.ndarray
            Phase screen for the layer.
        """
        if src.tag == 'LGS':
            sub_im = np.reshape(layer.phase[np.where(pupil_footprint==1)],[self.resolution,self.resolution])

            alpha_cone = np.arctan(self.D/2/src.altitude)
            h = src.altitude-layer.altitude

            if np.isinf(h):
                r =self.D/2
            else:
                r = h*np.tan(alpha_cone)

            ratio = self.D/r/2

            cube_in = np.atleast_3d(sub_im).T
            
            pixel_size_in   = layer.D/layer.resolution
            pixel_size_out  = pixel_size_in/ratio
            resolution_out  = self.resolution
            return np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out)).T* np.sqrt(layer.fractionalR0)
        else:
            return np.reshape(layer.phase[np.where(pupil_footprint==1)],[self.resolution,self.resolution])* np.sqrt(layer.fractionalR0)
    
    def get_opd_per_src(self, src, phase):
        """
        Sum up the phases from all layers to compute the OPD for a source.

        Parameters
        ----------
        src : Source
            Light source.
        phase : list of np.ndarray
            Phase per layer for the source.

        Returns
        -------
        np.ndarray
            Optical Path Difference (OPD) in meters.
        """
        self.logger.debug('Atmosphere::get_opd_per_src')

        opd_no_pupil = np.sum(phase,axis=0) * src.wavelength/2/np.pi # the phase is defined in rad, and the OPD in m
        return opd_no_pupil

    def print_atm_at_wavelength(self,wavelength):
        """
        Print r0 and seeing for the atmosphere at a specific wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength at which to compute seeing [m].

        Returns
        -------
        bool
            True when completed.
        """

        r0_wvl              = self.r0*((wavelength/self.wavelength)**(5/3))
        seeingArcsec_wvl    = 206265*(wavelength/r0_wvl)

        self.logger.info('Atmosphere::print_atm_at_wavelength - ATMOSPHERE AT '+str(wavelength)+' nm %%%%%%%%')
        self.logger.info('r0 \t\t'+str(r0_wvl) + ' \t [m]') 
        self.logger.info('Seeing \t' + str(np.round(seeingArcsec_wvl,2)) + str('\t ["]'))
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return True            
       
    def print_properties(self):
        """
        Print the properties of the atmosphere object.

        Returns
        -------
        bool
            True when completed.
        """
        self.logger.info('%%%%%%%%%%%Atmosphere::print_properties - ATMOSPHERE%%%%%%%%%%%%')
        self.logger.info('{: ^12s}'.format('Layer') + '{: ^12s}'.format('Direction')+ '{: ^12s}'.format('Speed')+ '{: ^12s}'.format('Altitude')+ '{: ^12s}'.format('Cn2')+ '{: ^12s}'.format('Diameter') )
        self.logger.info('{: ^12s}'.format('') + '{: ^12s}'.format('[deg]')+ '{: ^12s}'.format('[m/s]')+ '{: ^12s}'.format('[m]')+ '{: ^12s}'.format('[m-2/3]') + '{: ^12s}'.format('[m]'))

        self.logger.info('======================================================================')
        
        for i_layer in range(self.nLayer):
            self.logger.info('{: ^12s}'.format(str(i_layer+1)) + '{: ^12s}'.format(str(self.windDirection[i_layer]))+ '{: ^12s}'.format(str(self.windSpeed[i_layer]))+ '{: ^12s}'.format(str(self.altitude[i_layer]))+ '{: ^12s}'.format(str(self.fractionalR0[i_layer]) ) + '{: ^12s}'.format(str(getattr(self,'layer_'+str(i_layer+1)).D )))
            if i_layer<self.nLayer-1:
                self.logger.info('----------------------------------------------------------------------')

        self.logger.info('======================================================================')

        self.logger.info('{: ^18s}'.format('r0 @500 nm') + '{: ^18s}'.format(str(self.r0)+' [m]' ))
        self.logger.info('{: ^18s}'.format('L0') + '{: ^18s}'.format(str(self.L0)+' [m]' ))
        self.logger.info('{: ^18s}'.format('Seeing @500nm') + '{: ^18s}'.format(str(np.round(self.seeingArcsec,2))+' ["]'))
        self.logger.info('{: ^18s}'.format('Frequency') + '{: ^18s}'.format(str(np.round(1/self.samplingTime,2))+' [Hz]' ))
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return True
    
    def display_atm_layers(self,layer_index= None,fig_index = None,list_src = None):
        """
        Display the OPD of specified atmospheric layers and optionally source beams.

        Parameters
        ----------
        layer_index : list of int, optional
            Indices of layers to display. If None, shows all layers.
        fig_index : int, optional
            Figure index for matplotlib.
        list_src : list of Source, optional
            List of sources to project beams.

        Returns
        -------
        None
        """
        display_cn2 = False
        
        if layer_index is None:
            layer_index = list(np.arange(self.nLayer))
            n_sp        = len(layer_index) 
            display_cn2 = True
            
        if type(layer_index) is not list:
            raise TypeError(' layer_index should be a list') 
        normalized_speed = np.asarray(self.windSpeed)/max(self.windSpeed)

        if self.telescope.src.tag =='asterism' or self.telescope.src.tag =='sun':
            list_src = self.asterism
        
        plt.figure(fig_index,figsize = [n_sp*4,3*(1+display_cn2)], edgecolor = None)
        if display_cn2:
            gs = gridspec.GridSpec(1,n_sp+1, height_ratios=[1], width_ratios=np.ones(n_sp+1), hspace=0.5, wspace=0.5)
        else:
            gs = gridspec.GridSpec(1,n_sp, height_ratios=np.ones(1), width_ratios=np.ones(n_sp), hspace=0.25, wspace=0.25)
            
        axis_list = []
        for i in range(len(layer_index)):
            axis_list.append(plt.subplot(gs[0,i]))  
                      
        if display_cn2:
            # axCn2 = f.add_subplot(gs[1, :])
            ax = plt.subplot(gs[0,-1])         
            plt.imshow(np.tile(np.asarray(self.fractionalR0)[:,None],self.nLayer),origin='lower',interpolation='gaussian',extent=[0,1,self.altitude[0],self.altitude[-1]+5000],cmap='jet'),plt.clim([0,np.max(self.fractionalR0)])
            
                
            for i_layer in range(self.nLayer):
                plt.text(0.5,self.altitude[i_layer],str(self.fractionalR0[i_layer]*100)+'%',color = 'w',fontweight='bold')
            plt.ylabel('Altitude [m]')
            plt.title('Cn2 Profile')
            ax.set_xticks([])
            makeSquareAxes(plt.gca())
            
        for i_l,ax in enumerate(axis_list):
            
            tmpLayer = getattr(self, 'layer_'+str(layer_index[i_l]+1))
            ax.imshow(tmpLayer.phase,extent = [-tmpLayer.D/2,tmpLayer.D/2,-tmpLayer.D/2,tmpLayer.D/2])
            center = tmpLayer.D/2
            [x_tel,y_tel] = pol2cart(tmpLayer.D_fov/2, np.linspace(0,2*np.pi,100,endpoint=True))  
            if list_src is not None:
                cm = plt.get_cmap('gist_rainbow')
                col = []
                for i_source in range(len(list_src)):                   
                    col.append(cm(1.*i_source/len(list_src))) 
                    
                    [x_c,y_c] = pol2cart(self.telescope.D/2, np.linspace(0,2*np.pi,100,endpoint=True))
                    alpha_cone = np.arctan(self.telescope.D/2/list_src[i_source].altitude)
                    
                    h = list_src[i_source].altitude-tmpLayer.altitude
                    if np.isinf(h):
                        r =self.telescope.D/2
                    else:
                        r = h*np.tan(alpha_cone)
                    [x_cone,y_cone] = pol2cart(r, np.linspace(0,2*np.pi,100,endpoint=True))
                    if list_src[i_source].chromatic_shift is not None:
                        if len(list_src[i_source].chromatic_shift) == self.nLayer:
                            chromatic_shift = list_src[i_source].chromatic_shift[i_l]
                        else:
                            raise ValueError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')                            
                    else:
                        chromatic_shift = 0
                    
                    [x_z,y_z] = pol2cart(tmpLayer.altitude*np.tan((list_src[i_source].coordinates[0]+chromatic_shift)/206265) ,np.deg2rad(list_src[i_source].coordinates[1]))
                    center = 0
                    [x_c,y_c] = pol2cart(tmpLayer.D_fov/2, np.linspace(0,2*np.pi,100,endpoint=True))  
                    nm = (list_src[i_source].type) +'@'+str(list_src[i_source].coordinates[0])+'"'
    
                    ax.plot(x_cone+x_z+center,y_cone+y_z+center,'-', color = col [i_source],label=nm)        
                    ax.fill(x_cone+x_z+center,y_cone+y_z+center,y_z+center, alpha = 0.25, color = col[i_source])
            else:
                [x_c,y_c] = pol2cart(self.telescope.D/2, np.linspace(0,2*np.pi,100,endpoint=True))
                alpha_cone = np.arctan(self.telescope.D/2/self.telescope.src.altitude)
                
                h = self.telescope.src.altitude-tmpLayer.altitude
                if np.isinf(h):
                    r =self.telescope.D/2
                else:
                    r = h*np.tan(alpha_cone)
                [x_cone,y_cone] = pol2cart(r, np.linspace(0,2*np.pi,100,endpoint=True))
                
                if self.telescope.src.chromatic_shift is not None:
                    if len(self.telescope.src.chromatic_shift) == self.nLayer:
                        chromatic_shift = self.telescope.src.chromatic_shift[i_l]
                    else:
                        raise ValueError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')        
                else:
                    chromatic_shift = 0
                    
                    
                [x_z,y_z] = pol2cart((self.telescope.src.coordinates[0]+chromatic_shift)*np.tan((self.telescope.src.coordinates[0]+chromatic_shift)/206265) * tmpLayer.resolution / tmpLayer.D,np.deg2rad(self.telescope.src.coordinates[1]))
            
                center = 0
                [x_c,y_c] = pol2cart(tmpLayer.D_fov/2, np.linspace(0,2*np.pi,100,endpoint=True))  
            
                ax.plot(x_cone+x_z+center,y_cone+y_z+center,'-')        
                ax.fill(x_cone+x_z+center,y_cone+y_z+center,y_z+center, alpha = 0.6)     
            
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.set_title('Altitude '+str(tmpLayer.altitude)+' m')
            ax.plot(x_tel+center,y_tel+center,'--',color = 'k')
            ax.legend(loc ='upper left')

            makeSquareAxes(plt.gca())

            ax.arrow(center, center, center+normalized_speed[i_l]*(tmpLayer.D_fov/2)*np.cos(np.deg2rad(tmpLayer.direction)),center+normalized_speed[i_l]*(tmpLayer.D_fov/2)*np.sin(np.deg2rad(tmpLayer.direction)),length_includes_head=True,width=0.25, facecolor = [0,0,0],alpha=0.3,edgecolor= None)
            ax.text(center+tmpLayer.D_fov/8*np.cos(np.deg2rad(tmpLayer.direction)), center+tmpLayer.D_fov/8*np.sin(np.deg2rad(tmpLayer.direction)),str(self.windSpeed[i_l])+' m/s', fontweight=100,color=[1,1,1],fontsize = 18)
 # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    @property
    def r0(self):
         return self._r0
    
    @r0.setter
    def r0(self,val):
         self._r0 = val

         if self.hasNotBeenInitialized is False:
             self.logger.info('Atmosphere::r0.setter - Updating the Atmosphere covariance matrices...')             
             self.seeingArcsec           = 206265*(self.wavelength/val)
             self.cn2 = (self.r0**(-5. / 3) / (0.423 * (2*np.pi/self.wavelength)**2))/np.max([1, np.max(self.altitude)]) # Cn2 m^(-2/3)             
             for i_layer in range(self.nLayer):
                 tmpLayer = getattr(self,'layer_'+str(i_layer+1))

                 tmpLayer.ZZt_r0        = tmpLayer.ZZt*(self.r0_def/self.r0)**(5/3)
                 tmpLayer.ZXt_r0        = tmpLayer.ZXt*(self.r0_def/self.r0)**(5/3)
                 tmpLayer.XXt_r0        = tmpLayer.XXt*(self.r0_def/self.r0)**(5/3)
                 tmpLayer.ZZt_inv_r0    = tmpLayer.ZZt_inv/((self.r0_def/self.r0)**(5/3))
                 BBt                    = tmpLayer.XXt_r0 -  np.matmul(tmpLayer.A,tmpLayer.ZXt_r0)
                 tmpLayer.B             = np.linalg.cholesky(BBt)

    @property
    def L0(self):
         return self._L0
    
    @L0.setter
    def L0(self,val):
         self._L0 = val
         if self.hasNotBeenInitialized is False:
             self.logger.info('Atmosphere::L0.setter - Updating the Atmosphere covariance matrices...')

             self.hasNotBeenInitialized = True
             del self.ZZt
             del self.XXt
             del self.ZXt
             del self.ZZt_inv
             self.initializeAtmosphere(self.telescope)
    @property
    def windSpeed(self):
         return self._windSpeed
    
    @windSpeed.setter
    def windSpeed(self,val):
        self._windSpeed = val

        if self.hasNotBeenInitialized is False:
            if len(val)!= self.nLayer:
                self.logger.error('Atmosphere::windSpeed.setter - Error! Wrong value for the wind-speed! Make sure that you input a wind-speed for each layer')
            else:
                self.logger.info('Atmosphere::windSpeed.setter - Updating the wind speed...')
                for i_layer in range(self.nLayer):
                    tmpLayer = getattr(self,'layer_'+str(i_layer+1))
                    tmpLayer.windSpeed = val[i_layer]
                    tmpLayer.vY            = tmpLayer.windSpeed*np.cos(np.deg2rad(tmpLayer.direction))                    
                    tmpLayer.vX            = tmpLayer.windSpeed*np.sin(np.deg2rad(tmpLayer.direction))
                    ps_turb_x = tmpLayer.vX*self.telescope.samplingTime
                    ps_turb_y = tmpLayer.vY*self.telescope.samplingTime
                    tmpLayer.ratio[0] = ps_turb_x/self.ps_loop
                    tmpLayer.ratio[1] = ps_turb_y/self.ps_loop
                    setattr(self,'layer_'+str(i_layer+1),tmpLayer )
                
    @property
    def windDirection(self):
         return self._windDirection
    
    @windDirection.setter
    def windDirection(self,val):
        self._windDirection = val

        if self.hasNotBeenInitialized is False:
            if len(val)!= self.nLayer:
                self.logger.error('Atmosphere::windSpeed.setter - Error! Wrong value for the wind-speed! Make sure that you inpute a wind-speed for each layer')
            else:
                self.logger.info('Atmosphere::windSpeed.setter - Updating the wind direction...')
                for i_layer in range(self.nLayer):
                    tmpLayer = getattr(self,'layer_'+str(i_layer+1))
                    tmpLayer.direction = val[i_layer]
                    tmpLayer.vY            = tmpLayer.windSpeed*np.cos(np.deg2rad(tmpLayer.direction))                    
                    tmpLayer.vX            = tmpLayer.windSpeed*np.sin(np.deg2rad(tmpLayer.direction))
                    ps_turb_x = tmpLayer.vX*self.telescope.samplingTime
                    ps_turb_y = tmpLayer.vY*self.telescope.samplingTime
                    tmpLayer.ratio[0] = ps_turb_x/self.ps_loop
                    tmpLayer.ratio[1] = ps_turb_y/self.ps_loop
                    setattr(self,'layer_'+str(i_layer+1),tmpLayer )                
                            
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
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()
