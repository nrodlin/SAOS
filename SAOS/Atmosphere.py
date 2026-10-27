# -*- coding: utf-8 -*-
"""
Originally created on Fri Aug 14 10:59:02 2020
@author: cheritie

Major update on March 24 2025
@author: nrodlin
"""

import time

import logging

import h5py
import json
import os
import threading

from joblib import Parallel, delayed

import numpy as np

import torch

# Self dependencies
from .infinitePhaseScreen import PhaseScreenVonKarman
from .tools.tools import pol2cart
from .Layer import LayerClass
from .LoggingHelper import get_logger


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
                 zenith:float=0.0,
                 logger=None,
                 n_jobs=None):
        """
        Initialize an Atmosphere object representing layered atmospheric turbulence.

        Parameters
        ----------
        r0 : float
            Fried parameter at 500 nm [m] and pointing to zenith.
        L0 : float
            Outer scale of turbulence [m].
        windSpeed : list of float
            Wind speed for each layer [m/s].
        fractionalR0 : list of float
            Cn2 profile; fractional contribution to turbulence per layer.
        windDirection : list of float
            Wind direction for each layer [degrees]: direction towards which the turbulence moves, in the frame of the source
            azimuths (0 deg -> +x = +column, 90 deg -> +y = +row of the phase screens).
        altitude : list of float
            Altitude of each layer [m].
        telescope : Telescope object
            Telescope class instantiation to take certain parameters required for the atmosphere.
        zenith : float, optional, by default 0.0
            Zenith angle [degrees] of the telescope during the simulation, to automatically convert the r0 given to the specific conditions.
        logger : logging.Logger, optional
            Logger instance for this object, by default None.
        n_jobs : int, optional
            Threads used to compute the OPD of the sub-directions of an extended source and to update the layers. Default (None):
            the CPUs available to the process (os.sched_getaffinity, which honours Slurm, cgroups and taskset) when called from the
            main thread, and 1 when called from another thread (e.g. LightPaths propagated in parallel), so that thread pools are
            never nested.
        """
        self.logger = logger if logger is not None else get_logger()

        self.hasNotBeenInitialized  = True
        self.r0                     = r0                     # Fried Parameter in m, at 500 nm and pointing to the zenith
        self.fractionalR0           = fractionalR0           # Fractional Cn2 profile of the turbulence
        self.L0                     = L0                     # Outer Scale in m
        self.altitude               = altitude               # altitude of the layers
        self.nLayer                 = len(fractionalR0)      # number of layer
        self.windSpeed              = windSpeed              # wind speed of the layers in m/s
        self.windDirection          = windDirection          # wind direction in degrees
        self.zenith                 = zenith * (np.pi/180)   # Zenith angle of the telescope during simulation
        self.n_jobs                 = n_jobs                 # threads of the internal pools (None: automatic, see _n_jobs)
        self.tag                    = 'atmosphere'           # Tag of the object

        self.wavelength             = 500*1e-9               # Wavelength used to define the properties of the atmosphere

        self.logger.debug('Atmosphere::initializeAtmosphere - Taking key parameters from the telescope.')
        self.resolution = telescope.resolution
        self.D = telescope.D
        self.samplingTime = telescope.samplingTime

        self.fov      = telescope.fov
        self.fov_rad  = telescope.fov_rad
        
        # Sanity check
        if len(self.windSpeed) < self.nLayer:
            self.logger.error('Atmosphere::__init - Wind speed list size is smaller than the number of layers')
            raise ValueError('Wind speed list size is smaller than the number of layers')
        elif len(self.windSpeed) > self.nLayer:
            self.logger.warning('Atmosphere::__init - Wind speed list size is larger than the number of layers. Only the first layers are considered.')
        
        if len(self.windDirection) < self.nLayer:
            self.logger.error('Atmosphere::__init - Wind direction list size is smaller than the number of layers')
            raise ValueError('Wind direction list size is smaller than the number of layers')
        elif len(self.windDirection) > self.nLayer:
            self.logger.warning('Atmosphere::__init - Wind direction list size is larger than the number of layers. Only the first layers are considered.')

        if len(self.altitude) < self.nLayer:
            self.logger.error('Atmosphere::__init - Altitude list size is smaller than the number of layers')
            raise ValueError('Altitude list size is smaller than the number of layers')
        elif len(self.altitude) > self.nLayer:
            self.logger.warning('Atmosphere::__init - Altitude list size is larger than the number of layers. Only the first layers are considered.')            

    def initializeAtmosphere(self, randomState=None):
        """
        Initialize the atmosphere layers using the telescope linked at construction time.

        Parameters
        ----------
        randomState : int or None, optional
            Seed for reproducible random number generation (layer i uses randomState + 1000 i). None: a random seed, logged
            and stored in self.randomState so that the atmosphere can be reproduced.

        Returns
        -------
        bool
            True if initialization succeeded, False otherwise.
        """
        self.logger.debug('Atmosphere::initializeAtmosphere')

        if self.hasNotBeenInitialized:
            self.initial_r0 = self.r0
            self.logger.debug('Atmosphere::initializeAtmosphere - Creating layers...')
            if randomState is None:
                # Random base seed (not the time of day), logged so that the atmosphere can be reproduced
                randomState = int(np.random.default_rng().integers(0, 2**31 - 1000*self.nLayer))
                self.logger.info(f'Atmosphere::initializeAtmosphere - randomState not given: using {randomState}.')
            self.randomState = randomState
            results_layers = Parallel(n_jobs=self._n_jobs(self.nLayer), prefer="threads")(delayed(self.buildLayer)(i_layer, randomState) for i_layer in range(self.nLayer))
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
            seed = int(np.random.default_rng().integers(0, 2**31 - 1000*self.nLayer))
        layer.seed          = seed + i_layer*1000
        
        # gather properties of the atmosphere
        layer.altitude      = self.altitude[i_layer] / np.cos(self.zenith)       
        layer.windSpeed     = self.windSpeed[i_layer]
        layer.windDirection = self.windDirection[i_layer]
        
        # Diameter of the layer seen by the Field Of View (metapupil). The layer pixel is the pupil pixel D / resolution, so the
        # resolution pixels cut for a footprint span exactly D; the margin around the pupil is rounded up to whole pixels and kept
        # symmetric (npix - resolution even), so the on-axis footprint is centred on the layer
        layer.D_fov        = self.D+2*np.tan(self.fov_rad/2)*layer.altitude
        layer.spatial_res  = self.D/self.resolution
        layer.npix         = self.resolution + 2*int(np.ceil((layer.D_fov - self.D)/(2*layer.spatial_res) - 1e-9))

        self.logger.debug('Atmosphere::buildLayer - Creating layer '+str(i_layer+1))    

        # Fried parameter of the layer (a layer with zero Cn2 fraction has r0 = infinity: no turbulence)
        frac = float(self.fractionalR0[i_layer])
        layer.fractionalR0 = self.r0*(np.cos(self.zenith)**0.6)*(frac**(-3/5)) if frac > 0 else np.inf

        layer.screen = PhaseScreenVonKarman(nx_size=layer.npix, pixel_scale=layer.spatial_res, r0 = layer.fractionalR0, 
                                            L0 = self.L0, random_seed = layer.seed, n_columns=2, logger=self.logger)

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
        
        # Create folder if it does not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not filename.endswith(".h5"):
            filename += ".h5"
        
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
                # Evolution state: random generator of the extrusion and sub-pixel displacement buffers, so that a loaded
                # atmosphere continues exactly as the saved one would have
                group.attrs['rng_state'] = json.dumps(getattr(self,'layer_'+str(i_layer+1)).screen._R.bit_generator.state)
                group.attrs['displ_buffer_x'] = getattr(self,'layer_'+str(i_layer+1)).displ_buffer_x
                group.attrs['displ_buffer_y'] = getattr(self,'layer_'+str(i_layer+1)).displ_buffer_y
        
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


        if not filename.endswith(".h5"):
            filename += ".h5"        

        with h5py.File(filename, 'r') as f:
            # Layers in their numeric order (the h5 groups are listed alphabetically: layer_10 before layer_2)
            layer_names = sorted([k for k in f.keys() if k.startswith('layer_')], key=lambda k: int(k.split('_')[1]))
            for i_layer, grp_name in enumerate(layer_names):
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
                                                    random_seed=layer.seed, n_columns=f[grp_name].attrs['n_columns'], from_file=True, screen_file=screen_dict, logger=self.logger)
                # Evolution state (files saved before 2026-10-27 do not have it: the generator restarts from the seed)
                if 'rng_state' in f[grp_name].attrs:
                    layer.screen._R.bit_generator.state = json.loads(f[grp_name].attrs['rng_state'])
                layer.displ_buffer_x = float(f[grp_name].attrs.get('displ_buffer_x', 0.0))
                layer.displ_buffer_y = float(f[grp_name].attrs.get('displ_buffer_y', 0.0))
                
                setattr(self, 'layer_'+str(i_layer+1), layer)

            self.hasNotBeenInitialized = False


            self.logger.info(f"Atmosphere::load - All layers finished.")     
    

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
        results = Parallel(n_jobs=self._n_jobs(self.nLayer), prefer="threads")(
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

        # Move the screen by every whole pixel accumulated in the buffers (several per step for fast layers); the fraction stays
        # in the buffer. Same frame as the source azimuths (getOPD): x = +column, y = +row. add_col(+1) shifts the content
        # towards +column; add_row(+1) appends at the bottom, so the content moves towards -row: the row sign is inverted
        n_y = int(np.trunc(updatedLayer.displ_buffer_y))
        for _ in range(abs(n_y)):
            updatedLayer.screen.add_row(-np.sign(n_y))
        updatedLayer.displ_buffer_y -= n_y
        n_x = int(np.trunc(updatedLayer.displ_buffer_x))
        for _ in range(abs(n_x)):
            updatedLayer.screen.add_col(np.sign(n_x))
        updatedLayer.displ_buffer_x -= n_x

        return updatedLayer

    def _n_jobs(self, n_tasks):
        """
        Threads for an internal pool of n_tasks tasks: n_jobs if given; otherwise 1 when called from a thread other than the main one
        (the caller already runs in parallel: no nested pools), else the CPUs available to the process (not os.cpu_count, which
        ignores the Slurm allocation).
        """
        if self.n_jobs is not None:
            n = int(self.n_jobs)
        elif threading.current_thread() is not threading.main_thread():
            n = 1
        else:
            n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        return max(1, min(n, n_tasks))

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
            for subDir in src.subDirs_stars:
                list_src.append(subDir)
        else:
            list_src.append(src)

        # One task per line of sight: footprint per layer, phase per layer [rad], sum of the layers -> OPD without pupil [m].
        # A single pool (sequential when called from a parallel LightPath, see _n_jobs)
        def opd_of(s):
            return self.get_opd_per_src(s, self.project_phase(s, self.get_pupil_footprint(s)))
        n_jobs = self._n_jobs(len(list_src))
        t0 = time.time()
        if n_jobs == 1:
            result_opd_no_pupil = [opd_of(s) for s in list_src]
        else:
            result_opd_no_pupil = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(opd_of)(s) for s in list_src)
        self.logger.debug(f'getOPD: {time.time()-t0} s for {len(list_src)} lines of sight, {n_jobs} threads')
        return np.squeeze(result_opd_no_pupil)
           
    def get_pupil_footprint(self, src):
        """
        Determine the pupil footprint of a source on every layer. The footprint centre is h tan(theta) rounded to the nearest
        layer pixel (integer-pixel model, like the screen motion): position error <= 0.5 px per axis, unbiased.

        Parameters
        ----------
        src : Source
            Light source object.

        Returns
        -------
        list
            Pupil footprint (mask of the layer) per layer.
        """
        self.logger.debug('Atmosphere::set_pupil_footprint')
        footprint_per_layer = []

        for i_layer in range(self.nLayer):
            layer = getattr(self,'layer_'+str(i_layer+1))

            # Chromatic shift: one on-sky vector (dx, dy) [arcsec] per layer, x = +column, y = +row, common to every source of
            # this wavelength (atmospheric dispersion moves all the lines of sight alike); added to the source direction
            if src.chromatic_shift is None:
                chromatic_shift = np.zeros(2)
            else:
                chromatic_shift = np.asarray(src.chromatic_shift, dtype=float)
                if chromatic_shift.shape != (self.nLayer, 2):
                    raise ValueError(f'Atmosphere::get_pupil_footprint - chromatic_shift must be one (dx, dy) [arcsec] per layer, '
                                     f'shape ({self.nLayer}, 2); got {chromatic_shift.shape}.')
                chromatic_shift = chromatic_shift[i_layer]

            # The source coordinates are defined as: [elevation, azimuth] taking as reference the Zenith, i.e. [0,x] has the telescope pointing to the sky vertically.
            # Elevation is measured in arcsec, azimuth in degrees.
            # We need to compute the location of the source w.r.t zenith to define the pupil centered at the source
            # The polar coordinates, at an altiudue z_i are: r = z_i*tan(elevation), theta = azimuth. Then, we convert to cartesian coordinates.
            # Note that before changing coordinates, the radius in meters is converted to px using the layer diameter (layer.D) and phase screen size (layer.resolution)
            D = layer.npix * (self.D/ self.resolution)
            # On-sky direction vector [arcsec] of the line of sight (source + chromatic shift), then its footprint h tan(|v|) along v
            v = np.array(pol2cart(src.coordinates[0], np.deg2rad(src.coordinates[1]))) + chromatic_shift
            ang = np.hypot(v[0], v[1])
            r_px = layer.altitude*np.tan(ang/206265) * layer.npix / D
            [x_z, y_z] = (r_px * v / ang) if ang > 0 else (0.0, 0.0)
            # [x_z, y_z] are the cartesian coordinates in px w.r.t zenith. 
            # Rounded to the nearest layer pixel (integer-pixel model)
            center_x = int(np.round(y_z))+layer.npix//2
            center_y = int(np.round(x_z))+layer.npix//2

            # Finally, create the pupil and set the points inside the pupil to 1
            pupil_footprint = np.zeros([layer.npix,layer.npix])
            row0, col0 = center_x - self.resolution//2, center_y - self.resolution//2      # resolution pixels, odd or even
            pupil_footprint[row0:row0 + self.resolution, col0:col0 + self.resolution] = 1
            
            footprint_per_layer.append(pupil_footprint)
        return footprint_per_layer
            
    def project_phase(self, src, pupil_footprint):
        """
        Project atmospheric phase through layers for a given source.

        Parameters
        ----------
        src : Source
            Light source object.
        pupil_footprint : list of np.ndarray
            Pupil masks per layer.

        Returns
        -------
        list
            List of phase screens per layer.
        """
        self.logger.debug('Atmosphere::fill_phase_support')

        # Each layer shall run in parallel to reduce time consumption. Returns a list containing the phase per layer for the source line of sight
        result_phase = Parallel(n_jobs=1, prefer="threads")(
            delayed(self.get_phase_layered)(src, getattr(self,'layer_'+str(i_layer+1)), pupil_footprint[i_layer]) for i_layer in range(self.nLayer))
        
        return result_phase

    def get_phase_layered(self, src, layer, pupil_footprint):
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

        Returns
        -------
        np.ndarray
            Phase screen for the layer.
        """
        return np.reshape(layer.screen.scrn[np.where(pupil_footprint==1)],[self.resolution,self.resolution]) * (self.wavelength/src.wavelength)
    
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

    
