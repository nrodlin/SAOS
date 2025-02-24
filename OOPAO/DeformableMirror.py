# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie
"""

import inspect
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import logging
import logging.handlers
from queue import Queue

from .MisRegistration import MisRegistration
from .tools.interpolateGeometricalTransformation import interpolate_cube, interpolate_image
from .tools.tools import emptyClass, pol2cart, print_


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class dmLayerClass():
    def __init__(self):
        self.altitude = None
        self.D_fov = None # diameter of the DM projected into the altitude layer [meters]
        self.D_px = None # size of the DM in [pixels]
        self.telescope_D = None # Telescope diamter in [meters]
        self.telescope_D_px = None # Telescope diameter in [px] using the DM resolution
        self.telescope_resolution = None # Telescope diameter in [px] using the original telescope resolution
        self.center = None # center coordinates of the DM [pixels]
        self.pupil_footprint = None # 2D telescope pupil projected to the altitude layer
        

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
                 nJobs:int = 30,
                 nThreads:int = 20,
                 print_dm_properties:bool = True,
                 floating_precision:int = 64,
                 altitude:float = None,
                 flip = False,
                 flip_lr = False,
                 sign = 1,
                 valid_act_thresh_outer = None, 
                 valid_act_thresh_inner = None,
                 logger = None):
        """DEFORMABLE MIRROR
        A Deformable Mirror object consists in defining the 2D maps of influence functions of the actuators. 
        By default, the actuator grid is cartesian in a Fried Geometry with respect to the nSubap parameter. 
        The Deformable Mirror is considered to to be in a pupil plane.
        By default, the influence functions are 2D Gaussian functions normalized to 1 [m]. 
        IMPORTANT: The deformable mirror is considered to be transmissive instead of reflective. 
        This is to prevent any confusion with an eventual factor 2 in OPD due to the reflection.

        Parameters
        ----------
        telescope : Telescope
            the telescope object associated. 
            In case no coordinates are provided, the selection of the valid actuator is based on the radius
            of the telescope (assumed to be circular) and the central obstruction value (assumed to be circular).
            The telescope spiders are not considered in the selection of the valid actuators. 
            For more complex selection of actuators, specify the coordinates of the actuators using
            the optional parameter "coordinates" (see below).
        nSubap : float
            This parameter is used when no user-defined coordinates / modes are specified. This is used to compute
            the DM actuator influence functions in a fried geometry with respect to nSubap subapertures along the telescope 
            diameter. 
            If the optional parameter "pitch" is not specified, the Deformable Mirror pitch property is computed 
            as the ratio of the Telescope Diameter with the number of subaperture nSubap. 
            This impacts how the DM influence functions mechanical coupling is computed. .
        mechCoupling : float, optional
            This parameter defines the mechanical coupling between the influence functions. 
            A value of 0.35 which means that if an actuator is pushed to an arbitrary value 1, 
            the mechanical deformation at a circular distance of "pitch" from this actuator is equal to 0.35. 
            By default, "pitch" is the inter-actuator distance when the Fried Geometry is considered.   
            If the parameter "modes" is used, this parameter is ignored. The default is 0.35.
        coordinates : np.ndarray, optional
            User defined coordinates for the DM actuators. Be careful to specify the pitch parameter associated, 
            otherwise the pitch is computed using its default value (see pitch parameter). 
            If this parameter is specified, all the actuators are considered to be valid 
            (no selection based on the telescope pupil).
            The default is None.
        pitch : float, optional
            pitch considered to compute the Gaussian Influence Functions, associated to the mechanical coupling. 
            If no pitch is specified, the pitch is computed to match a Fried geometry according to the nSubap parameter.  
            The default is None.
        modes : np.ndarray, optional
            User defined influence functions or modes (modal DM) can be input to the Deformable Mirror. 
            They must match the telescope resolution and be input as a 2D matrix, where the 2D maps are 
            reshaped as a 1D vector of size n_pix*n_pix : size = [n_pix**2,n_modes].
            The default is None.
        misReg : TYPE, optional
            A Mis-Registration object (See the Mis-Registration class) can be input to apply some geometrical transformations
            to the Deformable Mirror. When using user-defined influence functions, this parameter is ignored.
            Consider to use the function applyMisRegistration in OOPAO/mis_registration_identification_algorithm/ to perform interpolations.
            The default is None.
        customDM : Parameter File, optional
            Parameter File for M4 computation. The default is None.
        nJobs : int, optional
            Number of jobs for the joblib multi-threading. The default is 30.
        nThreads : int, optional
            Number of threads for the joblib multi-threading. The default is 20.
        print_dm_properties : bool, optional
            Boolean to print the dm properties. The default is True.
        floating_precision : int, optional
            If set to 32, uses float32 precision to save memory. The default is 64.
        altitude : float, optional
            Altitude to which the DM is conjugated. The default is None and corresponds to a DM conjugated to the ground.
        valid_act_thresh_inner : float, optional
            Maximum distance from one actuator to the edge of the mirror in the obscuration inner circle so that it is consider valid.
        valid_act_thresh_inner : float, optional
            Maximum distance from one actuator to the edge of the mirror in the outer circle so that it is consider valid.

        Returns
        -------
        None.

        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Deformable Mirror object are listed here: 
        _ dm.coefs             : dm coefficients in units of dm.modes, if using the defauly gaussian influence functions, in [m].
        _ dm.OPD               : the 2D map of the optical path difference in [m]
        _ dm.modes             : matrix of size: [n_pix**2,n_modes]. 2D maps of the dm influence functions (or modes for a modal dm) where the 2D maps are reshaped as a 1D vector of size n_pix*n_pix.
        _ dm.nValidAct         : number of valid actuators
        _ dm.nAct              : Total number of actuator along the diameter (valid only for the default case using cartesian fried geometry).
                                 Otherwise nAct = dm.nValidAct.
        _ dm.coordinates       : coordinates in [m] of the dm actuators (should be input as a 2D array of dimension [nAct,2])
        _ dm.pitch             : pitch used to compute the gaussian influence functions
        _ dm.misReg            : MisRegistration object associated to the dm object
        
        The main properties of the object can be displayed using :
            dm.print_properties()

        ************************** PROPAGATING THE LIGHT THROUGH THE DEFORMABLE MIRROR **************************
        The light can be propagated from a telescope tel through the Deformable Mirror dm using: 
            tel*dm
        Two situations are possible:
            * Free-space propagation: The telescope is not paired to an atmosphere object (tel.isPaired = False). 
                In that case tel.OPD is overwritten by dm.OPD: tel.OPD = dm.OPD
            
            * Propagation through the atmosphere: The telescope is paired to an atmosphere object (tel.isPaired = True). 
                In that case tel.OPD is summed with dm.OPD: tel.OPD = tel.OPD + dm.OPD

        ************************** CHANGING THE OPD OF THE MIRROR **************************
        np.squeeze(interpolate_image(layer.phase, pixel_size_in, pixel_size_out, resolution_out, shift_x =layer.extra_sx, shift_y= layer.extra_sy))
        * The dm.OPD can be reseted to 0 by setting the dm.coefs property to 0:
            dm.coefs = 0

        * The dm.OPD can be updated by setting the dm.coefs property using a 1D vector vector_command of length dm.nValidAct: 
            
            dm.coefs = vector_command
        
        The resulting OPD is a 2D map obtained computing the matricial product dm.modes@dm.coefs and reshaped in 2D. 
        
        * It is possible to compute a cube of 2D OPD using a 2D matrix, matrix_command of size [dm.nValidAct, n_opd]: 
            
            dm.coefs = matrix_command
        
        The resulting OPD is a 3D map [n_pix,n_pix,n_opd] obtained computing the matricial product dm.modes@dm.coefs and reshaped in 2D. 
        This can be useful to parallelize the measurements, typically when measuring interaction matrices. This is compatible with tel*dm operation. 
        
        WARNING: At the moment, setting the value of a single (or subset) actuator will not update the dm.OPD property if done like this: 
            dm.coefs[given_index] = value
        It requires to re-assign dm.coefs to itself so the change can be detected using:
            dm.coefs = dm.coefs
        
                 
        ************************** EXEMPLE **************************
        
        1) Create an 8-m diameter circular telescope with a central obstruction of 15% and the pupil sampled with 100 pixels along the diameter. 
        tel = Telescope(resolution = 100, diameter = 8, centralObstruction = 0.15)
        
        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8) 
        
        3) Create a Deformable Mirror object with 21 actuators along the diameters (20 in the pupil) and influence functions with a coupling of 45 %.
        dm = DeformableMirror(telescope = tel, nSubap = 20, mechCoupling = 0.45)
        
        4) Assign a random vector for the coefficients and propagate the light
        dm. coefs = numpy.random.randn(dm.nValidAct)
        src*tel*dm
        
        5) To visualize the influence function as seen by the telescope: 
        dm. coefs = numpy.eye(dm.nValidAct)
        src*tel*dm
        
        tel.OPD contains a cube of 2D maps for each actuator

        """
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        # Define class attributes
        self.print_dm_properties = print_dm_properties
        self.floating_precision = floating_precision
        self.flip_= flip
        self.flip_lr = flip_lr 
        self.sign = sign
        if customDM is not None:
            print_('Building the set of influence functions for the custom DM...', print_dm_properties)
            # generate the M4 influence functions            

            pup = telescope.pupil
            filename = customDM['m4_filename']
            nAct = customDM['nActuator']
            
            a = time.time()
            # compute M4 influence functions
            # TODO: implement makeInfluenceFunctions for custom DMs
            try:
                coordinates_M4 = makeM4influenceFunctions(pup                   = pup,\
                                                            filename              = filename,\
                                                            misReg                = misReg,\
                                                            dm                    = self,\
                                                            nAct                  = nAct,\
                                                            nJobs                 = nJobs,\
                                                            nThreads              = nThreads,\
                                                            order                 = customDM['order_m4_interpolation'],\
                                                            floating_precision    = floating_precision)
            except:
                coordinates_M4 = makeM4influenceFunctions(pup                   = pup,\
                                                            filename              = filename,\
                                                            misReg                = misReg,\
                                                            dm                    = self,\
                                                            nAct                  = nAct,\
                                                            nJobs                 = nJobs,\
                                                            nThreads              = nThreads,\
                                                            floating_precision    = floating_precision)

#            selection of the valid M4 actuators
            if customDM['validActCriteria']!=0:
                IF_STD = np.std(np.squeeze(self.modes[telescope.pupilLogical,:]), axis=0)
                ACTXPC=np.where(IF_STD >= np.mean(IF_STD) * customDM['validActCriteria'])
                self.modes         = self.modes[:,ACTXPC[0]]
            
                coordinates = coordinates_M4[ACTXPC[0],:]
            else:
                coordinates = coordinates_M4
            # normalize coordinates 
            coordinates   = (coordinates/telescope.resolution - 0.5)*40
            self.M4_param = M4_param
            self.isM4 = True
            print_ ('Done!',print_dm_properties)
            b = time.time()

            print_('Done! M4 influence functions computed in ' + str(b-a) + ' s!',print_dm_properties)
            self.isCustom = True
        else:
            self.isCustom = False
        self.telescope             = telescope # Remove?
        self.altitude = altitude
        if mechCoupling <=0:
            raise ValueError('The value of mechanical coupling should be positive.')
        
        self.dm_layer = self.buildLayer(self.telescope, altitude)
        self.mechCoupling          = mechCoupling
        self.tag                   = 'deformableMirror'
        
        # case with no pitch specified (Cartesian geometry)
        if pitch is None:
            self.pitch             = self.dm_layer.D_fov/(nSubap)  # size of a subaperture
        else:
            self.pitch = pitch
        
        if misReg is None:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration()
        else:
            self.misReg=misReg
        
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# If no coordinates are given, the DM is in a Cartesian Geometry
        
        if coordinates is None:  
            print_('No coordinates loaded.. taking the cartesian geometry as a default',print_dm_properties)
            self.nAct                               = nSubap+1                            # In that case corresponds to the number of actuator along the diameter            
            self.nActAlongDiameter                  = self.nAct-1
            
            # set the coordinates of the DM object to produce a cartesian geometry
            x = np.linspace(-(self.dm_layer.D_fov)/2,(self.dm_layer.D_fov)/2,self.nAct)
            X,Y=np.meshgrid(x,x)            
            
            # compute the initial set of coordinates
            self.xIF0 = np.reshape(X,[self.nAct**2])
            self.yIF0 = np.reshape(Y,[self.nAct**2])
            
            # select valid actuators (central and outer obstruction)
            r = np.sqrt(self.xIF0**2 + self.yIF0**2)
            if valid_act_thresh_outer is None: 
                valid_act_thresh_outer = self.dm_layer.D_fov/2+0.7533*self.pitch
            if valid_act_thresh_inner is None:
                valid_act_thresh_inner = telescope.centralObstruction*self.dm_layer.D_fov/2-0.5*self.pitch
            
            validActInner = r > valid_act_thresh_inner
            validActOuter = r <= valid_act_thresh_outer
    
            self.validAct = validActInner*validActOuter
            self.nValidAct = sum(self.validAct) 
            
        # If the coordinates are specified
            
        else:
            if np.shape(coordinates)[1] !=2:
                raise AttributeError('Wrong size for the DM coordinates, the (x,y) coordinates should be input as a 2D array of dimension [nAct,2]')
                
            print_('Coordinates loaded...',print_dm_properties)

            self.xIF0 = coordinates[:,0]
            self.yIF0 = coordinates[:,1]
            self.nAct = len(self.xIF0)                            # In that case corresponds to the total number of actuators
            self.nActAlongDiameter = (self.dm_layer.D_fov)/self.pitch
            
            validAct=(np.arange(0,self.nAct))                     # In that case assumed that all the Influence Functions provided are controlled actuators
            
            self.validAct = validAct.astype(int)         
            self.nValidAct = self.nAct 
            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INFLUENCE FUNCTIONS COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        #  initial coordinates
        xIF0 = self.xIF0[self.validAct]
        yIF0 = self.yIF0[self.validAct]

        # anamorphosis
        xIF3, yIF3 = self.anamorphosis(xIF0,yIF0,self.misReg.anamorphosisAngle*np.pi/180,self.misReg.tangentialScaling,self.misReg.radialScaling)
        
        # rotation
        xIF4, yIF4 = self.rotateDM(xIF3,yIF3,self.misReg.rotationAngle*np.pi/180)
        
        # shifts
        xIF = xIF4-self.misReg.shiftX
        yIF = yIF4-self.misReg.shiftY
        
        self.xIF = xIF
        self.yIF = yIF

        
        # corresponding coordinates on the pixel grid
        u0x      = self.dm_layer.resolution/2+xIF*self.dm_layer.resolution/self.dm_layer.D_fov
        u0y      = self.dm_layer.resolution/2+yIF*self.dm_layer.resolution/self.dm_layer.D_fov      
        self.nIF = len(xIF)
        # store the coordinates
        self.coordinates        = np.zeros([self.nIF,2])
        self.coordinates[:,0]   = xIF
        self.coordinates[:,1]   = yIF
 
        if self.isCustom==False:
            print_('Generating a Deformable Mirror: ',print_dm_properties)
            if np.ndim(modes)==0:
                print_('Computing the 2D zonal modes...',print_dm_properties)
    #                FWHM of the gaussian depends on the anamorphosis
                def joblib_construction():
                    Q=Parallel(n_jobs=8,prefer='threads')(delayed(self.modesComputation)(i,j) for i,j in zip(u0x,u0y))
                    return Q 
                self.modes=np.squeeze(np.moveaxis(np.asarray(joblib_construction()),0,-1))
                    
            else:
                print_('Loading the 2D zonal modes...',print_dm_properties)
                self.modes = modes
                self.nValidAct = self.modes.shape[1]
                print_('Done!',print_dm_properties)

        else:
            print_('Using M4 Influence Functions',print_dm_properties)
        if floating_precision==32:            
            self.coefs = np.zeros(self.nValidAct,dtype=np.float32)
        else:
            self.coefs = np.zeros(self.nValidAct,dtype=np.float64)
        self.current_coefs = self.coefs.copy()
        if self.print_dm_properties:
            self.print_properties()
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GEOMETRICAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    # The DM can be considered as an atmospheric layers with discrete points actuated, which are then connected with their influence functions, 
    # shaping a continuous 2D surface. 
    def buildLayer(self, telescope, altitude):
        self.logger.debug('DeformableMirror::buildLayer')
        # initialize layer object
        layer                   = dmLayerClass()
       
        # gather properties of the atmosphere
        layer.altitude          = altitude       
                
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov             = telescope.D + 2*np.tan(telescope.fov/2)*layer.altitude # in [m]
        layer.D_px              = int(np.ceil((telescope.resolution/telescope.D)*layer.D_fov)) # Diameter in [px]
        layer.center            = layer.D_px//2

        layer.phase = np.zeros([layer.D_px,layer.D_px])
        layer.telescope_D          = telescope.D # Telescope diameter in [m]
        layer.telescope_D_px       = telescope.D * (layer.D_px / layer.D_fov) # Telescope diameter in [px] using the DM resolution
        layer.telescope_resolution = telescope.resolution # Telescope diameter in [px] using the original telescope resolution
        
        return layer
    # When the DM is located at an altitude layer, the phase of the DMs affect differently the sources dpending on their coordinates in sky. 
    # Before return the phase of the DM, we need to select the correct region of the DM affecting the source, which is done by masking the pupil
    def get_dm_pupil(self, src):
        self.logger.debug('DeformableMirror::get_dm_pupil')
        
        # Source coordinates are [angle_fov["], zenith_angle[rad]]. Hence, to obtain the location of the object at the DM altitude plane:
        # 1) Compute the projection: altitude * tan(angle_fov[rad]) -> location in meters
        # 2) From meters to pixels: result_1 * (D_px/metapupil_D)
        # 3) From polar to cartesian: (result_2[px], zenith_angle[rad]) -> (x_z, y_z) [px]
        [x_z, y_z] = pol2cart((self.dm_layer.altitude * np.tan(src.coordinates[0])/206265)*(self.dm_layer.D_px/self.dm_layer.D_fov), 
                             np.deg2rad(src.coordinates[1]))

        # Matriz origin is placed at the left-top corner, whereas the telescope origin is at the optical axis.
        # We add an offset to translate the origins.
        center_x = int(y_z) + self.dm_layer.D_px//2
        center_y = int(x_z) + self.dm_layer.D_px//2
    
        # Finally, we mask the region that sees the source. This region is centered at the location computed in 3) 
        # and its shape equals the telescope pupil with the DM layer diameter in [px]
        pupil_footprint = np.zeros([self.dm_layer.D_px, self.dm_layer.D_px])
        pupil_footprint[center_x-self.dm_layer.telescope_D_px//2:center_x+self.dm_layer.telescope_D_px//2,
                        center_y-self.dm_layer.telescope_D_px//2:center_y+self.dm_layer.telescope_D_px//2] = 1 
        
        return pupil_footprint

    # The OPD is computed for a given source - for altitude layers, it depends on its location in sky
    # Returns the OPD [m] and the phase [rad], for which the wavelength of the input source is used. 
    # The shape of the output is [telescope.resolution, telescope.resolution] [px]
    def get_dm_opd(self, source):
        self.logger.debug('DeformableMirror::get_dm_opd')
        # Set the OPD dimensions
        result_OPD = np.zeros((self.dm_layer.telescope_resolution, self.dm_layer.telescope_resolution))
        # Get the pupil for the object. For the case of the sun, only the central subdir is considered.
        pupil = self.get_dm_pupil(source)
        # Select only the region of the DM that is affecting to the source.
        OPD = np.reshape(self.OPD[np.where(pupil==1)],[self.dm_layer.telescope_D_px,self.dm_layer.telescope_D_px]) # reshape is necessary because indexing ravels the original 2D matrix

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
            output_OPD = np.squeeze(interpolate_image(OPD, 1, 1, self.dm_layer.telescope_resolution))

        output_phase = output_OPD * (2*np.pi / source.wavelength)       

        return output_OPD, output_phase                     
        
    # The shape of the mirror is controlled through a set of modes that by default are zonal --> defining a typical DM. 
    # If a modal DM is defined, then the coefficients correspond to those of the modal basis.
    # Please notice that in this context, the modes do not refer to the AO control modal base but the intrinsic mechanical behaviour of the DM.
    # The shape of the mirror is computed as the matricial product of modes x coeffs -> modes [dm_layer.D_px, nValidActs], coefs [nValidActs, 1]    
    def updateDMShape(self, val):
        self.logger.debug('DeformableMirror::updateDMShape')  
        if self.floating_precision==32:            
            self._coefs = np.float32(val)
        else:
            self._coefs = val

        if np.isscalar(val):
            if val==0:
                self.logger.info('DeformableMirror::updateDMShape - Setting the DM to zero.')  
                self._coefs = np.zeros(self.nValidAct,dtype=np.float64)
                try:
                    self.OPD = np.float64(np.reshape(np.matmul(self.modes,self._coefs),[self.dm_layer.resolution,self.dm_layer.resolution]))
                except:
                    self.OPD = np.float64(np.reshape(self.modes@self._coefs,[self.dm_layer.resolution,self.dm_layer.resolution]))

            else:
                self.logger.error('DeformableMirror::updateDMShape - Wrong value for the coefficients, cannot be scale unless it is zero.')
                return False    
        else:                
            if len(val)==self.nValidAct:
                # One array of coefficients is given
                if np.ndim(val)==1:
                    try:
                        self.OPD = np.float64(np.reshape(np.matmul(self.modes,self._coefs),[self.dm_layer.resolution,self.dm_layer.resolution]))
                    except:
                        self.OPD = np.float64(np.reshape(self.modes@self._coefs,[self.dm_layer.resolution,self.dm_layer.resolution]))
            else:
                self.logger.error('DeformableMirror::updateDMShape - Wrong value for the coefficients, only a 1D vector is expected.')    
                raise ValueError('DeformableMirror::updateDMShape - Dimensions do not match to the number of valid actuators.')
        return True
    
    def rotateDM(self,x,y,angle):
        self.logger.debug('DeformableMirror::rotateDM')
        xOut =   x*np.cos(angle)-y*np.sin(angle)
        yOut =   y*np.cos(angle)+x*np.sin(angle)
        return xOut,yOut

    def anamorphosis(self,x,y,angle,mRad,mNorm):
        self.logger.debug('DeformableMirror::anamorphosis')
        mRad  += 1
        mNorm += 1
        xOut   = x * (mRad*np.cos(angle)**2  + mNorm* np.sin(angle)**2)  +  y * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
        yOut   = y * (mRad*np.sin(angle)**2  + mNorm* np.cos(angle)**2)  +  x * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
    
        return xOut,yOut
        
    def modesComputation(self,i,j):
        self.logger.debug('DeformableMirror::modesComputation')
        x0 = i
        y0 = j
        cx = (1+self.misReg.radialScaling)*(self.dm_layer.resolution/self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))
        cy = (1+self.misReg.tangentialScaling)*(self.dm_layer.resolution/self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))

#                    Radial direction of the anamorphosis
        theta  = self.misReg.anamorphosisAngle*np.pi/180
        x      = np.linspace(0,1,self.dm_layer.resolution)*self.dm_layer.resolution
        X,Y    = np.meshgrid(x,x)
    
#                Compute the 2D Gaussian coefficients
        a = np.cos(theta)**2/(2*cx**2)  +  np.sin(theta)**2/(2*cy**2)
        b = -np.sin(2*theta)/(4*cx**2)   +  np.sin(2*theta)/(4*cy**2)
        c = np.sin(theta)**2/(2*cx**2)  +  np.cos(theta)**2/(2*cy**2)
    
        G= self.sign * np.exp(-(a*(X-x0)**2 +2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))    
        
        if self.flip_lr:
            G = np.fliplr(G)

        if self.flip_:
            G = np.flip(G)
            
        output = np.reshape(G,[1,self.dm_layer.resolution**2])
        if self.floating_precision == 32:
            output = np.float32(output)
            
        return output
    
    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFORMABLE MIRROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^21s}'.format('Controlled Actuators')                     + '{: ^18s}'.format(str(self.nValidAct)))
        print('{: ^21s}'.format('M4')                   + '{: ^18s}'.format(str(self.isM4)))
        print('{: ^21s}'.format('Pitch')                                    + '{: ^18s}'.format(str(self.pitch))                    +'{: ^18s}'.format('[m]'))
        print('{: ^21s}'.format('Mechanical Coupling')                      + '{: ^18s}'.format(str(self.mechCoupling))             +'{: ^18s}'.format('[%]' ))
        print('-------------------------------------------------------------------------------')
        print('Mis-registration:')
        self.misReg.print_()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

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
    
    def __del__(self):
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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       