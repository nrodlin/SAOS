# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:23:18 2020

@author: cheritie

Major update on 26/03/2025
@author: nrodlin
"""
import logging

import numpy as np
from .LoggingHelper import get_logger

"""
Telescope Module
=================

This module contains the `Telescope` class, used for modeling the telescope pupil and sampling in adaptive optics simulations.
"""

class Telescope:
    
    def __init__(self,resolution:float,
                 diameter:float,
                 samplingTime:float=0.001,
                 centralObstruction:float = 0,
                 fov:float = 0,
                 pupil:bool=None,
                 pupilReflectivity:float=1,
                 logger=None):
        """
        Initialize a Telescope object for adaptive optics simulations.

        Parameters
        ----------
        resolution : float
            Resolution of the pupil mask (number of pixels across diameter).
        diameter : float
            Physical diameter of the telescope in meters.
        samplingTime : float, optional
            Time step of the AO loop, used for updating atmospheric phase screens.
        centralObstruction : float, optional
            Central obstruction diameter as a fraction of telescope diameter (0–1).
        fov : float, optional
            Field of view in arcseconds.
        pupil : bool or np.ndarray, optional
            User-defined pupil mask. If None, generates a circular pupil with optional obstruction.
        pupilReflectivity : float or np.ndarray, optional
            Pupil reflectivity map or scalar. Defaults to 1.
        logger : logging.Logger, optional
            Logger instance for output. If None, a default logger is created.
        """

        self.logger = logger if logger is not None else get_logger()

        self.isInitialized               = False                        # Resolution of the telescope
        self.resolution                  = resolution                   # Resolution of the telescope
        self.D                           = diameter                     # Diameter in m
        self.pixelSize                   = self.D/self.resolution       # size of the pixels in m
        self.centralObstruction          = centralObstruction           # central obstruction
        self.fov                         = fov                          # Field of View in arcsec converted in radian
        self.fov_rad                     = fov/206265                   # Field of View in arcsec converted in radian
        self.samplingTime                = samplingTime                 # AO loop speed
        self.user_defined_pupil          = pupil                        # input user-defined pupil
        self.pupilReflectivity           = pupilReflectivity            # Pupil Reflectivity <=> amplitude map
        
        self.tag                         = 'telescope'                  # tag of the object

        self.set_pupil()                                                # set the pupil

        self.area_eff                    = (self.pupil.sum() / (self.resolution**2)) * (self.D**2) # Collecting area in meters

        self.isInitialized               = True

    def set_pupil(self):
        """
        Generate the pupil mask either from user input or by creating a circular aperture
        with optional central obstruction.

        Updates the following:
        - self.pupil
        - self.pupilReflectivity
        - self.pixelArea
        - self.pupilLogical

        Returns
        -------
        None
        """
        # Case where the pupil is not input: circular pupil with central obstruction    
        if self.user_defined_pupil is None:
            D           = self.resolution+1
            x           = np.linspace(-self.resolution/2,self.resolution/2,self.resolution)
            xx,yy       = np.meshgrid(x,x)
            circle      = xx**2 + yy**2
            obs         = circle >= (self.centralObstruction*D/2)**2
            self.pupil  = circle < (D/2)**2 
            self.pupil  = self.pupil*obs
        else:
            self.logger.info('Telescope::set_pupil - User-defined pupil, the central obstruction will not be taken into account.')
            self.pupil  = self.user_defined_pupil.copy()        
            
        self.pupilReflectivity           = self.pupil.astype(float)*self.pupilReflectivity                    # A non uniform reflectivity can be input by the user
        self.pixelArea                   = np.sum(self.pupil)                                                 # Total number of pixels in the pupil area
        self.pupilLogical                = np.where(np.reshape(self.pupil,self.resolution*self.resolution)>0) # index of valid pixels in the pupil

                
    def apply_spiders(self,angle,thickness_spider,offset_X = None, offset_Y=None):
        """
        Add spiders to the telescope pupil mask.

        Parameters
        ----------
        angle : list of float
            Direction of every arm [deg], counter-clockwise from +x (columns) with y = +row, the frame of the source
            azimuths. Each arm is a half-strip of width thickness_spider from the centre to the pupil edge.
        thickness_spider : float
            Width of the spider arms in meters.
        offset_X : list of float, optional
            X offsets per spider. Same length as angle.
        offset_Y : list of float, optional
            Y offsets per spider. Same length as angle.

        Returns
        -------
        None
        """
        # Reset the pupil to the default one before adding the spiders
        self.set_pupil()

        if thickness_spider > 0:           
            pup = np.copy(self.pupil)

            max_offset = self.centralObstruction*self.D/2 - thickness_spider/2

            if offset_X is None:
                offset_X = np.zeros(len(angle))
                
            if offset_Y is None:
                offset_Y = np.zeros(len(angle))
                        
            if np.max(np.abs(offset_X))>=max_offset or np.max(np.abs(offset_Y))>max_offset:
                self.logger.warning('Telescope::apply_spiders - The spider offsets are too large! Weird things could happen!')

            # Pixel centres of the pupil grid, (i - (N-1)/2) D/N, with x = +column and y = +row (frame of the source azimuths)
            x = (np.arange(self.resolution) - (self.resolution - 1) / 2) * self.D / self.resolution
            [X0, Y0] = np.meshgrid(x, x)
            for i in range(len(angle)):
                # Arm i: half-strip of width thickness_spider starting at the centre (displaced by the offsets) towards the
                # angle [deg], counter-clockwise from +x
                th = np.deg2rad(angle[i])
                X = X0 + offset_X[i]
                Y = Y0 + offset_Y[i]
                along = X*np.cos(th) + Y*np.sin(th)
                perp = -X*np.sin(th) + Y*np.cos(th)
                pup = pup * ~((along > 0) & (np.abs(perp) < thickness_spider/2))

            self.pupil = pup.copy()
            self.pupilReflectivity = self.pupil.astype(float) * self.pupilReflectivity
            self.pixelArea = np.sum(self.pupil)
            self.pupilLogical = np.where(np.reshape(self.pupil, self.resolution*self.resolution) > 0)
            
        else:
            self.logger.warning('Telescope::apply_spider - Thickness is <=0, returning default pupil.')
        
        self.area_eff                    = (self.pupil.sum() / (self.resolution**2)) * (self.D**2) # Collecting area in meters
        
        return 
        
    def print_properties(self):
        """
        Log and print key telescope properties: diameter, resolution, pixel size,
        surface area, central obstruction, and FOV.

        Returns
        -------
        None
        """        
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('{: ^18s}'.format('Diameter')                     + '{: ^18s}'.format(str(self.D))                                        +'{: ^18s}'.format('[m]'   ))
        self.logger.info('{: ^18s}'.format('Resolution')                   + '{: ^18s}'.format(str(self.resolution))                               +'{: ^18s}'.format('[pixels]'   ))
        self.logger.info('{: ^18s}'.format('Pixel Size')                   + '{: ^18s}'.format(str(np.round(self.pixelSize,2)))                    +'{: ^18s}'.format('[m]'   ))
        self.logger.info('{: ^18s}'.format('Surface')                      + '{: ^18s}'.format(str(np.round(self.pixelArea*self.pixelSize**2)))    +'{: ^18s}'.format('[m2]'  ))
        self.logger.info('{: ^18s}'.format('Central Obstruction')          + '{: ^18s}'.format(str(100*self.centralObstruction))                   +'{: ^18s}'.format('[% of diameter]' ))
        self.logger.info('{: ^18s}'.format('Pixels in the pupil')          + '{: ^18s}'.format(str(self.pixelArea))                                +'{: ^18s}'.format('[pixels]' ))
        self.logger.info('{: ^18s}'.format('Field of View')                + '{: ^18s}'.format(str(self.fov))                                      +'{: ^18s}'.format('[arcsec]' ))
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
