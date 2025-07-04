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

import h5py
import os

from joblib import Parallel, delayed

import numpy as np
import math

# Self dependencies
from .infinitePhaseScreen import PhaseScreenVonKarman
from .tools.tools import pol2cart
from .tools.interpolateGeometricalTransformation import interpolate_cube
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
                 telescope,
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
        telescope : Telescope object
            Telescope class instantiation to take certain parameters required for the atmosphere.
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
        self.r0                     = r0                # Fried Parameter in m 
        self.fractionalR0           = fractionalR0      # CFractional Cn2 profile of the turbulence
        self.L0                     = L0                # Outer Scale in m
        self.altitude               = altitude          # altitude of the layers
        self.nLayer                 = len(fractionalR0) # number of layer
        self.windSpeed              = windSpeed         # wind speed of the layers in m/s
        self.windDirection          = windDirection     # wind direction in degrees
        self.tag                    = 'atmosphere'      # Tag of the object

        self.wavelength             = 500*1e-9          # Wavelength used to define the properties of the atmosphere

        self.logger.debug('Atmosphere::initializeAtmosphere - Taking key parameters from the telescope.')
        self.resolution = telescope.resolution
        self.D = telescope.D
        self.samplingTime = telescope.samplingTime

        self.fov      = telescope.fov
        self.fov_rad  = telescope.fov_rad
        
    def initializeAtmosphere(self, randomState=None):
        """
        Initialize the atmosphere layers and associate them with a telescope.

        Parameters
        ----------
        telescope : Telescope or None
            Telescope object to derive spatial and temporal resolution.
        randomState : int or None
            Seed for reproducible random number generation, by default None.

        Returns
        -------
        bool
            True if initialization succeeded, False otherwise.
        """
        self.logger.debug('Atmosphere::initializeAtmosphere')

        if self.hasNotBeenInitialized:
            self.initial_r0 = self.r0
            self.logger.debug('Atmosphere::initializeAtmosphere - Creating layers...')
            results_layers = Parallel(n_jobs=self.nLayer, prefer="threads")(delayed(self.buildLayer)(i_layer, randomState) for i_layer in range(self.nLayer))
            for i_layer in range(self.nLayer):
                setattr(self,'layer_'+str(i_layer+1),results_layers[i_layer])
        else:
            self.logger.warning('Atmosphere::initializeAtmosphere - The atmosphere has already been initialized.')
            return True
        
        self.hasNotBeenInitialized  = False 
        return True
            
    def buildLayer(self, i_layer, seed = None):
        """
        Build and initialize a single atmospheric layer.

        Parameters
        ----------
        i_layer : int
            Index of the layer to build.
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        LayerClass
            The initialized atmospheric layer.
        """
        self.logger.debug('Atmosphere::buildLayer - layer '+str(i_layer+1))
         
        # initialize layer object
        layer               = LayerClass()
        layer.id            = i_layer
        # Seed for the random phase generation  
        if seed is None:      
            t = time.localtime()
            seed = t.tm_hour*3600 + t.tm_min*60 + t.tm_sec
        layer.seed          = seed + i_layer*1000
        
        # gather properties of the atmosphere
        layer.altitude      = self.altitude[i_layer]       
        layer.windSpeed     = self.windSpeed[i_layer]
        layer.windDirection = self.windDirection[i_layer]
        
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        
        layer.D_fov        = self.D+2*np.tan(self.fov_rad/2)*layer.altitude
        layer.npix         = int(np.ceil((self.resolution/self.D)*layer.D_fov))
        layer.spatial_res  = layer.D_fov/layer.npix

        self.logger.debug('Atmosphere::buildLayer - Creating layer '+str(i_layer+1))    

        layer.fractionalR0 = self.r0*self.fractionalR0[i_layer]**(-3/5)

        layer.screen = PhaseScreenVonKarman(nx_size=layer.npix, pixel_scale=layer.spatial_res, r0 = layer.fractionalR0, 
                                            L0 = self.L0, random_seed = layer.seed, n_columns=2)

        self.logger.debug('Atmosphere::buildLayer - Layer '+str(i_layer+1)+' created.')      
    
        return layer
   
    def save(self, filename):
        """
        Save the state of the atmosphere to a H5 file.

        Parameters
        ----------
        filename : str
            Path and base filename (with extension) to save the H5.

        Returns
        -------
        bool
            True if saved successfully, False otherwise.
        """
        self.logger.debug('Atmosphere::save')

        if self.hasNotBeenInitialized:
            self.logger.error('Atmosphere::save - The atmosphere has not been initialized yet.')
            return False
        
        # Create folder if it doe snot exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        self.logger.info('Atmosphere::save - Creating the H5 file tree')

        with h5py.File(filename, 'a') as f:
            for i_layer in range(self.nLayer):
                group = f.create_group('layer_' + str(i_layer+1))
                # Atmosphere layer params
                group.attrs['id'] = getattr(self,'layer_'+str(i_layer+1)).id
                group.attrs['seed'] = getattr(self,'layer_'+str(i_layer+1)).seed
                group.attrs['D_fov'] = getattr(self,'layer_'+str(i_layer+1)).D_fov
                group.attrs['spatial_res'] = getattr(self,'layer_'+str(i_layer+1)).spatial_res
                group.attrs['npix'] = getattr(self,'layer_'+str(i_layer+1)).npix
                group.attrs['fractionalR0'] = getattr(self,'layer_'+str(i_layer+1)).fractionalR0
                group.attrs['windSpeed'] = getattr(self,'layer_'+str(i_layer+1)).windSpeed
                group.attrs['windDirection'] = getattr(self,'layer_'+str(i_layer+1)).windDirection
                group.attrs['altitude'] = getattr(self,'layer_'+str(i_layer+1)).altitude
                # Von Karman infinite layer params:
                # Vertical movement matrices
                group.create_dataset('A_vert', data=getattr(self,'layer_'+str(i_layer+1)).screen.A_vert)
                group.create_dataset('B_vert', data=getattr(self,'layer_'+str(i_layer+1)).screen.B_vert)
                # Horizontal movement matrices
                group.create_dataset('A_horz', data=getattr(self,'layer_'+str(i_layer+1)).screen.A_horz)
                group.create_dataset('B_horz', data=getattr(self,'layer_'+str(i_layer+1)).screen.B_horz)
                # Common variables
                group.create_dataset('phase', data=getattr(self,'layer_'+str(i_layer+1)).screen.scrn)
                group.attrs['n_columns'] = getattr(self,'layer_'+str(i_layer+1)).screen.n_columns
        
        self.logger.info('Atmosphere::save - Saved.')
    
    def load(self, filename): 
        """
        Load an atmosphere configuration from a H5 file.

        Parameters
        ----------
        filename : str
            Path and base filename (with extension) of the H5 file to load.

        Returns
        -------
        bool
            True if loaded successfully, False otherwise.
        """
        self.logger.debug('Atmosphere::load')

        with h5py.File(filename, 'r') as f:
            for i_layer, grp_name in enumerate(f.keys()):
                layer = LayerClass()
                layer.id = f[grp_name].attrs['id']
                layer.seed = f[grp_name].attrs['seed']
                layer.D_fov = f[grp_name].attrs['D_fov']
                layer.spatial_res = f[grp_name].attrs['spatial_res']
                layer.npix = f[grp_name].attrs['npix']
                layer.fractionalR0 = f[grp_name].attrs['fractionalR0']
                layer.windSpeed = f[grp_name].attrs['windSpeed']
                layer.windDirection = f[grp_name].attrs['windDirection']
                layer.altitude = f[grp_name].attrs['altitude']
              
                screen_dict = {'A_vert': np.array(f[grp_name]['A_vert']),
                               'B_vert': np.array(f[grp_name]['B_vert']),
                               'A_horz': np.array(f[grp_name]['A_horz']),
                               'B_horz': np.array(f[grp_name]['B_horz']),
                               'phase': np.array(f[grp_name]['phase'])}
                layer.screen = PhaseScreenVonKarman(nx_size=layer.npix, pixel_scale=layer.spatial_res, r0=layer.fractionalR0, L0=self.L0, 
                                                    random_seed=layer.seed, n_columns=f[grp_name].attrs['n_columns'], from_file=True, screen_file=screen_dict)
                
                setattr(self, 'layer_'+str(i_layer+1), layer)

            self.hasNotBeenInitialized = False


            self.logger.info(f"Atmosphere::load - All layers finished.")     
    

            return False

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
        
    def updateLayer(self,updatedLayer):
        """
        Update a single atmospheric layer, shifting the phase screen.
        ^ +Vy (adds row --> + to the bottom, - to the top)
        |
        |
        |----> +Vx (adds col --> + to the left, - to the right)

        Parameters
        ----------
        updatedLayer : LayerClass
            The layer to update.

        Returns
        -------
        LayerClass
            The updated layer.
        """
        self.logger.debug('Atmosphere::updateLayer')

        # Compute speed per axis

        vx = updatedLayer.windSpeed * np.cos(np.deg2rad(updatedLayer.windDirection))
        vy = updatedLayer.windSpeed * np.sin(np.deg2rad(updatedLayer.windDirection))

        # Compute displacement in px
        updatedLayer.displ_buffer_x += (vx*self.samplingTime) / updatedLayer.spatial_res
        updatedLayer.displ_buffer_y += (vy*self.samplingTime) / updatedLayer.spatial_res

        # Check if there is a full pixel displacement and update the screen:

        if (np.abs(updatedLayer.displ_buffer_y) > 0):
            updatedLayer.screen.add_row(np.sign(updatedLayer.displ_buffer_y))
            # updatedLayer.displ_buffer_y = updatedLayer.displ_buffer_y - np.sign(updatedLayer.displ_buffer_y) * 1
        
        if (np.abs(updatedLayer.displ_buffer_x) > 0):
            updatedLayer.screen.add_col(np.sign(updatedLayer.displ_buffer_x))
            # updatedLayer.displ_buffer_x = updatedLayer.displ_buffer_x - np.sign(updatedLayer.displ_buffer_x) * 1

        return updatedLayer

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
            D = layer.npix * (self.D/ self.resolution)
            [x_z,y_z] = pol2cart(layer.altitude*np.tan((src.coordinates[0]+chromatic_shift)/206265) * layer.npix / D, np.deg2rad(src.coordinates[1]))
            # [x_z, y_z] are the cartesian coordinates in px w.r.t zenith. 
            center_x = int(y_z)+layer.npix//2
            center_y = int(x_z)+layer.npix//2

            # As the coordinates are discretized in pixels, there may be an offset. We need to pass this offset when the phase is projected,
            # so we store it and return it to the main function
            extra_s.append([int(x_z)-x_z, int(y_z)-y_z])
    
            # Finally, create the pupil and set the points inside the pupil to 1
            pupil_footprint = np.zeros([layer.npix,layer.npix])
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
            return np.reshape(layer.screen.scrn[np.where(pupil_footprint==1)],[self.resolution,self.resolution])
    
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
