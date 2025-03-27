# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie

Major update on March 24 2025
@author: nrodlin
"""


import numpy as np
import torch
import cv2
from joblib import Parallel, delayed

import logging
import logging.handlers
from queue import Queue

from .MisRegistration import MisRegistration
from .tools.interpolateGeometricalTransformation import interpolate_cube
from .tools.tools import pol2cart


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class dmLayerClass():
    def __init__(self):
        self.altitude = None
        self.D_fov                  = None # diameter of the DM projected into the altitude layer [meters]
        self.D_px                   = None # size of the DM in [pixels]
        self.telescope_D            = None # Telescope diamter in [meters]
        self.telescope_D_px         = None # Telescope diameter in [px] using the DM resolution
        self.telescope_resolution   = None # Telescope diameter in [px] using the original telescope resolution
        self.center                 = None # center coordinates of the DM [pixels]
        self.pupil_footprint        = None # 2D telescope pupil projected to the altitude layer
        self.OPD                    = None # stores the layer OPD without projection to any source (full pupil/metapupil)
        
"""
Deformable Mirror Module
=================

This module contains the `DeformableMirror` class, used for modeling a deformable mirror in adaptive optics simulations.
"""

class DeformableMirror:
    def __init__(self,
                 telescope,
                 nSubap:float,
                 mechCoupling:float = 0.35,
                 coordinates:np.ndarray = None,
                 pitch:float = None,
                 modes:np.ndarray = None,
                 misReg = None,
                 customDM = None,
                 floating_precision:int = 64,
                 altitude:float = None,
                 flip = False,
                 flip_lr = False,
                 sign = 1,
                 valid_act_thresh_outer = None, 
                 valid_act_thresh_inner = None,
                 logger = None):
        """
        Initialize a Deformable Mirror (DM) with zonal or modal influence functions.

        Parameters
        ----------
        telescope : Telescope
            Telescope associated with this DM.
        nSubap : float
            Number of subapertures across the pupil diameter.
        mechCoupling : float, optional
            Coupling factor between actuators, by default 0.35.
        coordinates : np.ndarray, optional
            Custom actuator coordinates. Overrides default grid.
        pitch : float, optional
            Actuator pitch in meters.
        modes : np.ndarray, optional
            Influence functions or modal basis.
        misReg : MisRegistration, optional
            Misregistration object for geometrical offsets.
        customDM : dict, optional
            Custom DM setup configuration.
        floating_precision : int, optional
            Use 32 or 64-bit floats, by default 64.
        altitude : float, optional
            Conjugation altitude of the DM in meters.
        flip : bool, optional
            Flip the influence functions vertically.
        flip_lr : bool, optional
            Flip the influence functions left-right.
        sign : int, optional
            Sign of actuation.
        valid_act_thresh_outer : float, optional
            Threshold for validating actuators outside pupil.
        valid_act_thresh_inner : float, optional
            Threshold for validating actuators inside central obstruction.
        logger : logging.Logger, optional
            Logger instance.
        """
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Define class attributes
        self.tag = 'deformableMirror'

        self.floating_precision = floating_precision
        self.flip_= flip
        self.flip_lr = flip_lr 
        self.sign = sign
        self.altitude = altitude

        if mechCoupling <=0:
            raise ValueError('The value of mechanical coupling should be positive.')
        else:
            self.mechCoupling          = mechCoupling

        # The DM can be customized, defining the Influence Functions
        if customDM is not None:
            self.logger.warning('DeformableMirror::__init__ - Custon DM IF is not yet supported in this new version, using default.')
            self.isCustom = False
        else:
            self.isCustom = False
        # Define the DM layer       
        self.dm_layer = self.buildLayer(telescope, altitude)
        
        # Case with no pitch specified --> Cartesian geometry
        if pitch is None:
            self.pitch             = self.dm_layer.D_fov/(nSubap)  # size of a subaperture
        else:
            self.pitch = pitch
        
        if misReg is None:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration()
        else:
            self.misReg=misReg
        
        ## DM initialization

        # If no coordinates are given --> Cartesian Geometry is assumed
        
        if coordinates is None: 
            self.logger.info('DeformableMirror::__init__ - No coordinates loaded.. taking the cartesian geometry as a default') 

            self.nAct                               = nSubap+1 # In that case corresponds to the number of actuator along the diameter            
            self.nActAlongDiameter                  = self.nAct-1 
            
            # set the coordinates of the DM object to produce a cartesian geometry
            x = np.linspace(-(self.dm_layer.D_fov)/2,(self.dm_layer.D_fov)/2,self.nAct)
            X,Y=np.meshgrid(x,x)            
            
            # compute the initial set of coordinates
            self.xIF0 = np.reshape(X,[self.nAct**2])
            self.yIF0 = np.reshape(Y,[self.nAct**2])
            
            # select valid actuators --> removing the central and outer obstruction
            r = np.sqrt(self.xIF0**2 + self.yIF0**2)
            if valid_act_thresh_outer is None: 
                valid_act_thresh_outer = self.dm_layer.D_fov/2+0.7533*self.pitch
            if valid_act_thresh_inner is None:
                valid_act_thresh_inner = telescope.centralObstruction*self.dm_layer.D_fov/2-0.5*self.pitch
            
            validActInner = r > valid_act_thresh_inner
            validActOuter = r <= valid_act_thresh_outer
    
            self.validAct = validActInner*validActOuter
            self.validAct_2D = self.validAct.reshape(self.nAct,self.nAct)
            self.nValidAct = sum(self.validAct) 
            
        # If the coordinates are specified   
        else:
            if np.shape(coordinates)[1] !=2:
                raise AttributeError('DeformableMirror::__init__ - Wrong size for the DM coordinates, \
                                     the (x,y) coordinates should be input as a 2D array of dimension [nAct,2]')
                
            self.logger.info('DeformableMirror::__init__ - Coordinates loaded.') 

            self.xIF0 = coordinates[:,0]
            self.yIF0 = coordinates[:,1]
            self.nAct = len(self.xIF0) # In that case corresponds to the total number of actuators
            self.nActAlongDiameter = (self.dm_layer.D_fov)/self.pitch
            
            validAct=(np.arange(0,self.nAct)) # In that case assumed that all the Influence Functions provided are controlled actuators
            
            self.validAct = validAct.astype(int)         
            self.nValidAct = self.nAct 
            
        #  INFLUENCE FUNCTIONS COMPUTATION
        #  initial coordinates
        xIF0 = self.xIF0[self.validAct]
        yIF0 = self.yIF0[self.validAct]

        # anamorphosis
        xIF3, yIF3 = self.anamorphosis(xIF0, yIF0, self.misReg.anamorphosisAngle*np.pi/180, self.misReg.tangentialScaling, self.misReg.radialScaling)
        
        # rotation
        xIF4, yIF4 = self.rotateDM(xIF3, yIF3, self.misReg.rotationAngle*np.pi/180)
        
        # shifts
        xIF = xIF4 - self.misReg.shiftX
        yIF = yIF4 - self.misReg.shiftY
        
        self.xIF = xIF
        self.yIF = yIF

        # corresponding coordinates on the pixel grid
        u0x      = self.dm_layer.D_px /2 + xIF*self.dm_layer.D_px /self.dm_layer.D_fov
        u0y      = self.dm_layer.D_px /2 + yIF*self.dm_layer.D_px /self.dm_layer.D_fov      
        self.nIF = len(xIF)
        # store the coordinates
        self.coordinates        = np.zeros([self.nIF, 2])
        self.coordinates[:,0]   = xIF
        self.coordinates[:,1]   = yIF

        # If the Influence Functions are not provided, the DM must generate them:
        if self.isCustom==False:
            self.logger.info('DeformableMirror::__init__ - Generating Deformable Mirror modes.')
            if np.ndim(modes)==0:
                self.logger.info('DeformableMirror::__init__ - Computing the 2D zonal modes..')
                # FWHM of the gaussian depends on the anamorphosis
                def joblib_construction():
                    Q=Parallel(n_jobs=8,prefer='threads')(delayed(self.modesComputation)(i,j) for i,j in zip(u0x,u0y))
                    return Q 
                self.modes = np.ascontiguousarray(np.squeeze(np.moveaxis(np.asarray(joblib_construction()),0,-1))).reshape(self.dm_layer.D_px,
                                                                                                                           self.dm_layer.D_px,
                                                                                                                           self.nValidAct)
                self.modes_torch = torch.tensor(self.modes).contiguous()
            else:
                self.logger.info('DeformableMirror::__init__ - Loading the 2D zonal modes..')
                self.modes = modes
                self.nValidAct = self.modes.shape[1]
            self.logger.info('DeformableMirror::__init__ - Done!')
        else:
            self.logger.info('DeformableMirror::__init__ - Using Custom Influence Functions.')
        
        # Setting the precision for the actuation commands

        if floating_precision==32:            
            self.coefs = np.zeros(self.nValidAct,dtype=np.float32)
        else:
            self.coefs = np.zeros(self.nValidAct,dtype=np.float64)
    
    # The DM can be considered as an atmospheric layers with discrete points actuated, which are then connected with their influence functions, 
    # shaping a continuous 2D surface. 
    def buildLayer(self, telescope, altitude):
        """
        Construct and configure the DM layer at a given conjugation altitude.

        Parameters
        ----------
        telescope : Telescope
            Telescope providing aperture and resolution information.
        altitude : float
            Altitude in meters to conjugate the DM layer.

        Returns
        -------
        dmLayerClass
            Configured DM layer with geometric and aperture metadata.
        """
        self.logger.debug('DeformableMirror::buildLayer')
        # initialize layer object
        layer                   = dmLayerClass()
       
        # gather properties of the atmosphere
        if altitude is None:
            layer.altitude          = 0
        else:
            layer.altitude          = altitude
                
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov             = telescope.D + 2*np.tan(telescope.fov/(206624*2))*layer.altitude # in [m]
        layer.D_px              = int(np.ceil((telescope.resolution/telescope.D)*layer.D_fov)) # Diameter in [px]
        layer.center            = layer.D_px//2

        layer.OPD               = np.zeros([layer.D_px,layer.D_px]) # stores the layer OPD without projection to any source (full pupil/metapupil)

        layer.telescope_D          = telescope.D # Telescope diameter in [m]
        layer.telescope_D_px       = int(np.ceil(telescope.D * (layer.D_px / layer.D_fov))) # Telescope diameter in [px] using the DM resolution
        layer.telescope_resolution = telescope.resolution # Telescope diameter in [px] using the original telescope resolution
        
        return layer
    # When the DM is located at an altitude layer, the phase of the DMs affect differently the sources dpending on their coordinates in sky. 
    # Before return the phase of the DM, we need to select the correct region of the DM affecting the source, which is done by masking the pupil
    def get_dm_pupil(self, src):
        """
        Compute pupil mask seen by a source at the DM altitude.

        Parameters
        ----------
        src : Source
            Source object with angular position.

        Returns
        -------
        np.ndarray
            Binary pupil mask (1s where pupil is seen by the source).
        """
        self.logger.debug('DeformableMirror::get_dm_pupil')
        
        # Source coordinates are [angle_fov["], zenith_angle[rad]]. Hence, to obtain the location of the object at the DM altitude plane:
        # 1) Compute the projection: altitude * tan(angle_fov[rad]) -> location in meters
        # 2) From meters to pixels: result_1 * (D_px/metapupil_D)
        # 3) From polar to cartesian: (result_2[px], zenith_angle[rad]) -> (x_z, y_z) [px]
        [x_z, y_z] = pol2cart(self.dm_layer.altitude * np.tan(src.coordinates[0]/206265)*(self.dm_layer.D_px/self.dm_layer.D_fov), 
                             np.deg2rad(src.coordinates[1]))

        # Matriz origin is placed at the left-top corner, whereas the telescope origin is at the optical axis.
        # We add an offset to translate the origins.
        center_x = int(y_z) + self.dm_layer.D_px//2
        center_y = int(x_z) + self.dm_layer.D_px//2
    
        # Finally, we mask the region that sees the source. This region is centered at the location computed in 3) 
        # and its shape equals the telescope pupil with the DM layer diameter in [px]
        pupil_footprint = np.zeros([self.dm_layer.D_px, self.dm_layer.D_px])
        # Define square limits to take the region of the metapupil affecting the source
        left_corner_x = center_x-self.dm_layer.telescope_D_px//2
        left_corner_y = center_y-self.dm_layer.telescope_D_px//2
        # Mask the region
        pupil_footprint[left_corner_x:left_corner_x + self.dm_layer.telescope_D_px,
                        left_corner_y:left_corner_y + self.dm_layer.telescope_D_px] = 1 
        
        return pupil_footprint

    # The OPD is computed for a given source - for altitude layers, it depends on its location in sky
    # Returns the OPD [m] and the phase [rad], for which the wavelength of the input source is used. 
    # The shape of the output is [telescope.resolution, telescope.resolution] [px]
    def get_dm_opd(self, source):
        """
        Compute the Optical Path Difference (OPD) and phase from the DM for a given source.

        Parameters
        ----------
        source : Source
            Source object defining wavelength and position.

        Returns
        -------
        tuple of np.ndarray
            OPD in meters and phase in radians.
        """
        self.logger.debug('DeformableMirror::get_dm_opd')
        # Get the pupil for the object. For the case of the sun, only the central subdir is considered.
        pupil = self.get_dm_pupil(source)
        # Select only the region of the DM that is affecting to the source.
        OPD = self.dm_layer.OPD[pupil==1].reshape((self.dm_layer.telescope_D_px, self.dm_layer.telescope_D_px))
        # Depending on the source type, certain action may differ
        if source.tag == 'LGS':
            # This code considers the impact of having an object at a finite altitude (typ. LGS). 
            sub_im = np.atleast_3d(OPD)
                
            alpha_cone = np.arctan(self.dm_layer.telescope_D/2/source.altitude)
            h = source.altitude-self.dm_layer.altitude

            if np.isinf(h):
                r = self.dm_layer.telescope_D/2
            else:
                r = h*np.tan(alpha_cone)

            ratio = self.dm_layer.telescope_D/r/2
            
            cube_in = sub_im.T
            pixel_size_in   = self.dm_layer.D_fov/self.dm_layer.D_px
            pixel_size_out  = pixel_size_in/ratio
            
            output_OPD = np.asarray(np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, self.dm_layer.telescope_resolution)))
        
        else: # NGS and Sun types can be handled equally. The sun is simplified, only considering the projection of the centrar subdir
            output_OPD = cv2.resize(OPD, (self.dm_layer.telescope_resolution, self.dm_layer.telescope_resolution), interpolation=cv2.INTER_AREA)

        output_phase = output_OPD * (2*np.pi / source.wavelength)       

        return output_OPD, output_phase                     

    # The shape of the mirror is controlled through a set of modes that by default are zonal --> defining a typical DM. 
    # If a modal DM is defined, then the coefficients correspond to those of the modal basis.
    # Please notice that in this context, the modes do not refer to the AO control modal base but the intrinsic mechanical behaviour of the DM.
    # The shape of the mirror is computed as the matricial product of modes x coeffs -> modes [dm_layer.D_px, nValidActs], coefs [nValidActs, 1]    
    def updateDMShape(self, val):
        """
        Update the OPD map from the current coefficients or 2D grid.

        Parameters
        ----------
        val : np.ndarray
            Either a coefficient vector or a 2D shape map.

        Returns
        -------
        bool
            True if update was successful.
        """
        self.logger.debug('DeformableMirror::updateDMShape') 

        if isinstance(val, np.ndarray):
            if (val.shape[0] == self.nAct) and (val.shape[1] == self.nAct):
                # The command received is a 2D matrix, take only the valid actuators!
                val = val.flatten()[self.validAct]
        
        if self.floating_precision==32:            
            self._coefs = np.float32(val)
        else:
            self._coefs = val

        if len(val)==self.nValidAct:
            temp = torch.matmul(self.modes_torch, torch.tensor(self._coefs))
            self.dm_layer.OPD = temp.view(self.dm_layer.D_px,self.dm_layer.D_px).double().numpy()
        else:
            self.logger.error('DeformableMirror::updateDMShape - Wrong value for the coefficients, only a 1D vector or a valid 2D matrix is expected.')    
            raise ValueError('DeformableMirror::updateDMShape - Dimensions do not match to the number of valid actuators.')

        return True
    
    def rotateDM(self,x,y,angle):
        """
        Rotate coordinates of the DM actuators.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        tuple
            Rotated x and y arrays.
        """
        self.logger.debug('DeformableMirror::rotateDM')
        xOut =   x*np.cos(angle)-y*np.sin(angle)
        yOut =   y*np.cos(angle)+x*np.sin(angle)
        return xOut,yOut

    def anamorphosis(self,x,y,angle,mRad,mNorm):
        """
        Apply anamorphic transformation to actuator coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        angle : float
            Rotation angle in radians.
        mRad : float
            Radial scaling.
        mNorm : float
            Tangential scaling.

        Returns
        -------
        tuple
            Transformed x and y coordinates.
        """
        self.logger.debug('DeformableMirror::anamorphosis')
        mRad  += 1
        mNorm += 1
        xOut   = x * (mRad*np.cos(angle)**2  + mNorm* np.sin(angle)**2)  +  y * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
        yOut   = y * (mRad*np.sin(angle)**2  + mNorm* np.cos(angle)**2)  +  x * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
    
        return xOut,yOut
        
    def modesComputation(self,i,j):
        """
        Compute Gaussian influence function at a given location.

        Parameters
        ----------
        i : float
            X center of actuator.
        j : float
            Y center of actuator.

        Returns
        -------
        np.ndarray
            Influence function as flattened array.
        """
        self.logger.debug('DeformableMirror::modesComputation')
        x0 = i
        y0 = j
        cx = (1+self.misReg.radialScaling)*(self.dm_layer.D_px /self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))
        cy = (1+self.misReg.tangentialScaling)*(self.dm_layer.D_px /self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))

        # Radial direction of the anamorphosis
        theta  = self.misReg.anamorphosisAngle*np.pi/180
        x      = np.linspace(0,1,self.dm_layer.D_px )*self.dm_layer.D_px 
        X,Y    = np.meshgrid(x,x)
    
        # Compute the 2D Gaussian coefficients
        a = np.cos(theta)**2/(2*cx**2)  +  np.sin(theta)**2/(2*cy**2)
        b = -np.sin(2*theta)/(4*cx**2)   +  np.sin(2*theta)/(4*cy**2)
        c = np.sin(theta)**2/(2*cx**2)  +  np.cos(theta)**2/(2*cy**2)
    
        G = self.sign * np.exp(-(a*(X-x0)**2 +2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
        
        if self.flip_lr:
            G = np.fliplr(G)

        if self.flip_:
            G = np.flip(G)
            
        output = np.reshape(G,[1,self.dm_layer.D_px **2])
        
        output[output < 1e-10] = 0

        if self.floating_precision == 32:
            output = np.float32(output)
           
        return output
    
    def print_properties(self):
        """
        Print a summary of the DM configuration.

        Returns
        -------
        None
        """
        self.logger.info('DeformableMirror::print_properties')
        self.logger.info('DeformableMirror::print_properties')
        self.logger.info('{: ^21s}'.format('Controlled Actuators')                     + '{: ^18s}'.format(str(self.nValidAct)))
        self.logger.info('{: ^21s}'.format('CustomDM')                                 + '{: ^18s}'.format(str(self.isCustom)))
        self.logger.info('{: ^21s}'.format('Pitch')                                    + '{: ^18s}'.format(str(self.pitch))                    +'{: ^18s}'.format('[m]'))
        self.logger.info('{: ^21s}'.format('Mechanical Coupling')                      + '{: ^18s}'.format(str(self.mechCoupling))             +'{: ^18s}'.format('[%]' ))
        self.logger.info('Mis-registration:')
        self.misReg.print_properties()

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
#        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    # Returns the coefficients for the modes of the DM.
    @property
    def coefs(self):
        return self._coefs
    # A setter is defined to avoid unprocessed variation of the coefficient vector!
    @coefs.setter
    def coefs(self,val):
        self.logger.debug('DeformableMirror::coefs')
        self.updateDMShape(val)

