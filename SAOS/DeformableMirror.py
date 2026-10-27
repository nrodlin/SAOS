# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie

Major update on March 24 2025
@author: nrodlin
"""


import numpy as np
import torch



import h5py

import logging

from .MisRegistration import MisRegistration
from .tools.tools import pol2cart
from .LoggingHelper import get_logger


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class dmLayerClass():
    def __init__(self):
        self.altitude = None
        self.D_fov                  = None # diameter of the DM projected into the altitude layer (metapupil) [meters]
        self.D_px                   = None # size of the DM grid in [pixels]: telescope resolution + symmetric whole-pixel margin
        self.pixel_size             = None # grid step [meters]: the telescope pixel D / resolution, samples at the cell centres
        self.telescope_D            = None # Telescope diamter in [meters]
        self.telescope_resolution   = None # Telescope diameter in [px] using the original telescope resolution
        self.center                 = None # center coordinates of the DM [pixels]
        self.metapupil              = None # 2D telescope pupil at the DM altitude [circular mask], at 0km equals the pupil without spider nor central obs
        self.pupil                  = None # 2D telescope pupil [binary mask]
        self.OPD                    = None # stores the layer OPD without projection to any source (full pupil/metapupil)
        self.cmd_1D                 = None # stores the 1D DM command, including valid and invalid actuators
        
"""
Deformable Mirror Module
=================

This module contains the `DeformableMirror` class, used for modeling a deformable mirror in adaptive optics simulations.
"""

class DeformableMirror:
    def __init__(self,
                 telescope,
                 nActs:float,
                 mechCoupling:float = 0.60,
                 coordinates:np.ndarray = None,
                 pitch:float = None,
                 modes:np.ndarray = None,
                 misReg = None,
                 typeDM:str = 'cartesian',
                 altitude:float = None,
                 valid_act_thresh_outer = None,
                 logger = None,
                 **kwargs):
        """
        Initialize a Deformable Mirror (DM) with zonal or modal influence functions.

        Parameters
        ----------
        telescope : Telescope
            Telescope associated with this DM.
        nActs : float
            Number of actuators in the horizontal axis of the pupil.
        mechCoupling : float, optional
            Coupling factor between actuators, by default 0.60.
        coordinates : np.ndarray, optional
            Custom actuator coordinates.
        pitch : float, optional
            Actuator pitch in meters.
        modes : np.ndarray, optional
            Influence functions or modal basis.
        misReg : MisRegistration, optional
            Misregistration object for geometrical offsets.
        typeDM : str, optional
            Type of the DM: {cartesian, radial, custom}. By default, cartesian. If 'custom', the `coordinates` parameter must be provided.
        altitude : float, optional
            Conjugation altitude of the DM in meters.
        valid_act_thresh_outer : float, optional
            Threshold for validating actuators outside pupil.
        logger : logging.Logger, optional
            Logger instance.
        **kwargs : dict, optional
            Additional keyword arguments.
            
            validActThreshpercentage : float, optional
                Parameter to select a percentage of the actuator pitch to consider it valid o not.
            maxStrokePtV : float, optional
                Maximum mechanical stroke peak-to-valley in [m]. By default 100e-6 [m].
            dynamicModel : str, optional
                Path to the h5 file containing the state-space model of the Deformable Mirror.
            normalize_IFs : bool, optional
                Whether to apply Partition of Unity normalization to custom influence functions. By default True.
        """
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        self.logger = logger if logger is not None else get_logger()

        # Define class attributes
        self.tag = 'deformableMirror'

        self.altitude = altitude
        self.nActs = nActs

        if mechCoupling <=0:
            raise ValueError('The value of mechanical coupling should be >= 0.')
        else:
            self.mechCoupling          = mechCoupling
        
        # Define the DM layer       
        self.dm_layer = self.buildLayer(telescope, altitude)

        if pitch is None:
            self.pitch = self.dm_layer.D_fov/(self.nActs-1)  # size of a subaperture
        else:
            self.pitch = pitch
        
        if misReg is None:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration(0,0,0,1,telescope=telescope, logger=self.logger)
        else:
            self.misReg=misReg            

        self.valid_act_thresh_outer = valid_act_thresh_outer
        self.validActThreshpercentage = kwargs.get('validActThreshpercentage', 0.0) # Dasp uses 0.7533, but the border are not seen well, which inestabilizes the loop.
        self.maxStrokePtV = kwargs.get('maxStrokePtV', 100e-6) # [m]
        self.dynamic_model_path = kwargs.get('dynamicModel', '')

        self.device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Depending on the type of the DM, the coordinates are generated differently
        if typeDM == 'cartesian':
            # Define the coordinates
            self.coordinates, self.validAct, self.nValidAct = self.generate_cartesian_dm(self.nActs)
        elif typeDM == 'radial':
            self.coordinates, self.validAct, self.nValidAct = self.generate_radial_dm()
        elif typeDM == 'custom':
            self.coordinates, self.validAct, self.nValidAct = self.generate_custom_dm(coordinates)
        else:
            self.logger.error('DeformableMirror::__init__ - Unrecognized DM type, using default. Implemented are: [cartesian, radial, custom]')
            raise ValueError('Unrecognized DM type, using default.')
        
        self.typeDM = typeDM
        
        # Compute scaling (epsilon) for the Gaussian influence functions based on mechanical coupling
        self.epsilon = np.sqrt(-1*np.log(self.mechCoupling))/self.pitch

        # High resolution meshgrid: the telescope pixel grid (cell centres, step D / resolution), as the pupil, the atmosphere and
        # the WFS sample the phase; the actuators span the metapupil D_fov
        x    = (np.arange(self.dm_layer.D_px) - (self.dm_layer.D_px - 1)/2) * self.dm_layer.pixel_size
        X, Y = np.meshgrid(x,x)

        self.high_res_coords = np.array([X.flatten(), Y.flatten()]).T
        
        self.custom_IFs = modes is not None
        if modes is not None:
            self.logger.info('DeformableMirror::__init__ - Using custom influence functions for shape generation.')
            custom_IF = torch.as_tensor(modes, device=self.device, dtype=torch.float64)
            if custom_IF.ndim == 3:
                custom_IF = custom_IF.reshape(-1, custom_IF.shape[-1])
            
            # Normalize custom influence functions using Partition of Unity if requested
            if kwargs.get('normalize_IFs', True):
                norm_factor = custom_IF.sum(dim=1, keepdim=True)
                self.influence_functions = custom_IF / (norm_factor + 1e-12)
            else:
                self.influence_functions = custom_IF
        else:
            # Compute the normalized influence functions using Partition of Unity
            self.influence_functions = self.precomputeInfluenceFunctions(self.coordinates[self.validAct], self.high_res_coords, self.epsilon)

        # Load dynamic model, if specified
        if self.dynamic_model_path != '':
            self.dyn_A, self.dyn_B, self.dyn_C, self.dyn_D = self.load_dynamic_model(self.dynamic_model_path, telescope.samplingTime)
        else:
            self.dyn_A = None
            self.dyn_B = None
            self.dyn_C = None
            self.dyn_D = None
    
    # Generation of the cartesian coordinates and mask of valid actuators using the outer pupil limit
    
    def generate_cartesian_dm(self, nActs):
        """
        Generates a distribution of cartesian points and a logic mask 
        filtering the points that are within the limits of the external
        pupil diameter.

        Parameters
        ----------
        nActs : int
            Number of actuators in the square side

        Returns
        -------
        coordinates : numpy.ndarray
            X and Y coordinates aranged as [nActs**2,2]
        validAct : numpy.ndarray
            Logic mask of valid actuators
        nValidAct : int
            Number of valid actuators
        """
        # First, we need to generate the coordinates for the cartesian grid, centered in the optical axis.

        x    = np.linspace(-(self.dm_layer.D_fov)/2,(self.dm_layer.D_fov)/2, nActs)
        X, Y = np.meshgrid(x,x)

        coordinates = np.array([X.flatten(), Y.flatten()]).T

        # Second, define mask to obtain the valid actuators

        r = np.sqrt(X**2 + Y**2)
        
        if self.valid_act_thresh_outer is None:
            self.valid_act_thresh_outer = self.dm_layer.D_fov/2 + self.validActThreshpercentage*self.pitch
        
        validAct  = r <= self.valid_act_thresh_outer
        nValidAct = np.sum(validAct)

        return coordinates, validAct.flatten(), nValidAct
    
    def generate_radial_dm(self):
        """
        Generate a distribution of radial actuator points approximated by a hexagonal grid,
        with a logic mask filtering the points within the external pupil diameter.

        Returns
        -------
        coordinates : numpy.ndarray
            X and Y coordinates arranged as [nActs, 2].
        validAct : numpy.ndarray
            Boolean mask of valid actuators.
        nValidAct : int
            Number of valid actuators.
        """
        # Define the vertical spacing for the actuators --> pitch applies for the 

        dy = self.pitch * (np.sqrt(3) / 2) # form equilateral triangles

        n_rows = int(np.ceil(2 * self.dm_layer.D_fov/2 / dy)) + 1

        ys = (np.arange(n_rows) - (n_rows - 1) / 2) * dy

        # Generate the coordinates following a triangular pattern
        coordinates = []

        for j, y in enumerate(ys):
            x_offset = 0.5 * self.pitch if (j % 2) else 0.0
            xs = np.arange(-self.dm_layer.D_fov/2 - self.pitch, self.dm_layer.D_fov/2 + self.pitch + 1e-12, self.pitch) + x_offset
            for x in xs:
                coordinates.append((x, y))

        coordinates = np.array(coordinates, dtype=float)

        # Second, define mask to obtain the valid actuators

        r = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        
        if self.valid_act_thresh_outer is None:
            self.valid_act_thresh_outer = self.dm_layer.D_fov/2 + self.validActThreshpercentage*self.pitch
        
        validAct  = r <= self.valid_act_thresh_outer
        nValidAct = np.sum(validAct)    

        return coordinates, validAct.flatten(), nValidAct

    def generate_custom_dm(self, custom_coordinates):
        """
        Generate a custom distribution of actuator points based on provided coordinates,
        with a logic mask filtering the points within the external pupil diameter.

        Parameters
        ----------
        custom_coordinates : numpy.ndarray
            X and Y coordinates arranged as [N, 2].

        Returns
        -------
        coordinates : numpy.ndarray
            X and Y coordinates arranged as [N, 2].
        validAct : numpy.ndarray
            Boolean mask of valid actuators.
        nValidAct : int
            Number of valid actuators.
        """
        if custom_coordinates is None:
            self.logger.error('DeformableMirror::generate_custom_dm - Custom coordinates must be provided for custom DM type.')
            raise ValueError('Custom coordinates not provided.')

        coordinates = np.array(custom_coordinates, dtype=float)

        # define mask to obtain the valid actuators
        r = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        
        if self.valid_act_thresh_outer is None:
            self.valid_act_thresh_outer = self.dm_layer.D_fov/2 + self.validActThreshpercentage*self.pitch
        
        validAct  = r <= self.valid_act_thresh_outer
        nValidAct = np.sum(validAct)    

        return coordinates, validAct.flatten(), nValidAct

    def precomputeInfluenceFunctions(self, input_points, output_points, epsilon):
        """
        Precompute normalized Gaussian influence functions using Partition of Unity.

        Parameters
        ----------
        input_points : np.ndarray
            2D coordinates of valid mirror actuators [nValidAct, 2].
        output_points : np.ndarray
            2D coordinates of the high-resolution output grid [D_px^2, 2].
        epsilon : float
            Radial scaling factor for the Gaussian fitting derived from pitch and mechanical coupling.

        Returns
        -------
        influence_functions : torch.Tensor
            Normalized influence function matrix [D_px^2, nValidAct] with partition of unity.
        """
        input_points_torch  = torch.as_tensor(input_points,  device=self.device, dtype=torch.float64)
        output_points_torch = torch.as_tensor(output_points, device=self.device, dtype=torch.float64)

        r = torch.cdist(output_points_torch, input_points_torch)

        influence_functions = torch.exp(-(epsilon * r)**2)

        # Partition of unity normalization across actuators for each pixel
        norm_factor = influence_functions.sum(dim=1, keepdim=True)
        influence_functions.div_(norm_factor + 1e-12)

        return influence_functions

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
                
        # Diameter of the layer seen by the Field Of View (metapupil) and its grid: the telescope pixel D / resolution, with the
        # margin around the pupil rounded up to whole pixels and symmetric (D_px - resolution even), so that the on-axis footprint
        # is centred and every altitude samples the same physical pixel as the pupil and the atmosphere
        layer.D_fov             = telescope.D + 2*np.tan(np.deg2rad(telescope.fov/3600)/2)*layer.altitude # in [m]
        layer.pixel_size        = telescope.D/telescope.resolution # in [m]
        layer.D_px              = telescope.resolution + 2*int(np.ceil((layer.D_fov - telescope.D)/(2*layer.pixel_size) - 1e-9)) # in [px]
        layer.center            = layer.D_px//2

        layer.OPD               = np.zeros([layer.D_px,layer.D_px]) # stores the layer OPD without projection to any source (full pupil/metapupil)
        layer.cmd_1D            = None # stores the 1D DM command, including valid and invalid actuators

        layer.telescope_D          = telescope.D # Telescope diameter in [m]
        layer.telescope_resolution = telescope.resolution # Telescope diameter in [px] using the original telescope resolution

        # Circular metapupil (pixel centres inside D_fov)
        x = (np.arange(layer.D_px) - (layer.D_px - 1)/2) * layer.pixel_size
        xx, yy = np.meshgrid(x, x)
        layer.metapupil = xx**2 + yy**2 <= (layer.D_fov/2)**2
        layer.pupil                 = telescope.pupil.copy()
        
        return layer
    # When the DM is located at an altitude layer, the phase of the DMs affect differently the sources depending on their coordinates in sky. 
    # Before return the phase of the DM, we need to select the correct region of the DM affecting the source, which is done by masking an square area
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
            Binary square mask (1s where the source is affected).
        """
        self.logger.debug('DeformableMirror::get_dm_pupil')
        
        # Source coordinates are [angle_fov["], zenith_angle[rad]]. Hence, to obtain the location of the object at the DM altitude plane:
        # 1) Compute the projection: altitude * tan(angle_fov[rad]) -> location in meters
        # 2) From meters to pixels: result_1 / pixel_size
        # 3) From polar to cartesian: (result_2[px], zenith_angle[rad]) -> (x_z, y_z) [px]
        [x_z, y_z] = pol2cart(self.dm_layer.altitude * np.tan(src.coordinates[0]/206265)/self.dm_layer.pixel_size, 
                             np.deg2rad(src.coordinates[1]))

        # Matriz origin is placed at the left-top corner, whereas the telescope origin is at the optical axis.
        # We add an offset to translate the origins. The offset is rounded to the nearest pixel (integer-pixel model, as the
        # atmosphere footprint): position error <= 0.5 px per axis, unbiased.
        center_x = int(np.round(y_z)) + self.dm_layer.D_px//2
        center_y = int(np.round(x_z)) + self.dm_layer.D_px//2
    
        # Finally, we mask the region that sees the source. This region is centered at the location computed in 3) 
        # and its shape equals the telescope pupil with the DM layer diameter in [px]
        square_mask = np.zeros([self.dm_layer.D_px, self.dm_layer.D_px])
        # Define square limits to take the region of the metapupil affecting the source
        left_corner_x = center_x-self.dm_layer.telescope_resolution//2
        left_corner_y = center_y-self.dm_layer.telescope_resolution//2
        # Mask the region
        square_mask[left_corner_x:left_corner_x + self.dm_layer.telescope_resolution,
                    left_corner_y:left_corner_y + self.dm_layer.telescope_resolution] = 1
        
        return square_mask

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
        # Apply mis-registration
        opd_misregistered = self.misReg.apply_misreg(self.dm_layer.OPD)
        # Select only the region of the DM that is affecting to the source.
        OPD = np.zeros([self.dm_layer.telescope_resolution, self.dm_layer.telescope_resolution])
        OPD = opd_misregistered[pupil==1].reshape(self.dm_layer.telescope_resolution, self.dm_layer.telescope_resolution)
        # Natural sources and the sun (only the projection of the central sub-direction is considered)
        output_OPD = OPD * self.dm_layer.pupil

        output_phase = output_OPD * (2*np.pi / source.wavelength)       

        return output_OPD, output_phase

    # Saturate the actuation of the deformable mirror

    def saturateShape(self, cmd):
        """
        Saturate the command of the mirror.

        Parameters
        ----------
        cmd : np.ndarray
            Command required from the mirror.

        Returns
        -------
        cmd_saturated
            Command executed by the mirror.
        """
        self.logger.debug('DeformableMirror::saturateShape') 

        # The OPD is treated in the mulator as wavefront --> the PtV maximum is equivalent to the wavefront value
        cmd_saturated = np.clip(cmd, a_min=-self.maxStrokePtV, a_max=self.maxStrokePtV)
        return cmd_saturated
    
    def load_dynamic_model(self, filename, samplingTime):
        """
        Load the state-space model of the deformable mirror from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the H5 file containing the discrete state-space matrices.
        samplingTime : float
            Sampling time used to discretize the model [s].

        Returns
        -------
        A : torch.Tensor
            Discrete state-transition matrix.
        B : torch.Tensor
            Discrete input-state matrix.
        C : torch.Tensor
            Discrete state-output matrix.
        D : torch.Tensor
            Discrete feedthrough matrix.
        """        

        self.logger.debug('DeformableMirror::load_dynamic_model')


        if not filename.endswith(".h5"):
            filename += ".h5"        

        with h5py.File(filename, 'r') as f:

            A = torch.atleast_2d(torch.as_tensor(f['A']['data'][()], dtype=torch.float64, device=self.device))
            B = torch.atleast_2d(torch.as_tensor(f['B']['data'][()], dtype=torch.float64, device=self.device))
            B = B.T # so that it is a column vector, loaded by default as row

            C = torch.atleast_2d(torch.as_tensor(f['C']['data'][()], dtype=torch.float64, device=self.device))
            D = torch.atleast_2d(torch.as_tensor(f['D']['data'][()], dtype=torch.float64, device=self.device))

            Ts = f['A'].attrs['Ts']

            if Ts != samplingTime:
                self.logger.error('DeformableMirror::load_dynamic_model - Sampling time of the state-space does not match the simulation\'s.')
                raise ValueError('Sampling time of the state-space does not match the simulation\'s')

        # Generate the null initial conditions for the state of each valid actuator
        self.curr_state = torch.zeros((self.nValidAct, A.shape[0]), dtype=torch.float64, device=self.device)

        return A, B, C, D
    
    def applyDynamics(self, cmd):
        """
        Apply a state-space to the command of the mirror to obtain the temporal response of the mirror

        Parameters
        ----------
        cmd : torch.Tensor
            Command required from the mirror.

        Returns
        -------
        dyn_cmd : torch.Tensor
            Temporal command executed by the mirror.
        """
        self.logger.debug('DeformableMirror::applyDynamics') 

        dyn_cmd = torch.zeros_like(cmd, dtype=torch.float64, device=self.device)

        x_next  = self.curr_state@self.dyn_A.T + cmd@self.dyn_B.T
        dyn_cmd = self.curr_state@self.dyn_C.T + cmd@self.dyn_D.T

        # Update state for the next temporal iteration
        self.curr_state.copy_(x_next)

        return dyn_cmd 

    # The shape of the mirror is computed as the matrix product of influence functions x commands -> influence_functions [D_px^2, nValidActs], coefs [nValidActs, 1]
    def updateDMShape(self, val, dynamicResponse=True):
        """
        Update the mirror OPD map from the actuator command vector.

        Parameters
        ----------
        val : torch.Tensor, np.ndarray, or list
            1D actuator command vector of length nValidAct or nActs^2.
        dynamicResponse : bool, optional
            Whether to apply the DM dynamic model (if configured). Default is True.

        Returns
        -------
        bool
            True if update was successful.
        """
        self.logger.debug('DeformableMirror::updateDMShape') 

        if isinstance(val, (np.ndarray, list)):
            val = torch.as_tensor(val, device=self.device, dtype=torch.float64)
        elif isinstance(val, torch.Tensor):
            val = val.to(device=self.device, dtype=torch.float64)
        else:
            raise TypeError('Expected Torch Tensor or numpy array.')

        if val.squeeze().ndim > 1:
            self.logger.error(f'DeformableMirror::updateDMShape - Shape of the command is not supported. Expected 1D array.')                
            raise ValueError('Shape of the command is not supported. Expected 1D array.')
        if val.shape[0] == self.validAct.shape[0]:
            # Command received is 1D, without filtering the unused actuators
            val = val[self.validAct]
        elif val.shape[0] == self.nValidAct:
            # Command received is 1D, only valid actuators
            val = val
        else:
            self.logger.error(f'DeformableMirror::updateDMShape - Size of the command is not correct: {val.shape}.')
            raise ValueError('Size of the command is not correct.')
        
        # Ensure dimensions nValidAct,1
        if val.ndim == 1:
            val = val.unsqueeze(1)
        # Fill the layer 1D command
        temp = np.zeros_like(self.validAct, dtype=np.float32)
        temp[self.validAct] = val.detach().cpu().numpy().squeeze()
        
        self.dm_layer.cmd_1D = temp.copy()

        # Compute the shape of the mirror using the influence functions and applying the dynamics, if specified
        if (self.dyn_A is not None) and (dynamicResponse is True):
            coefs_torch = self.applyDynamics(val)
        else:
            coefs_torch           = val

        opd_highres = (self.influence_functions @ coefs_torch).squeeze(1)

        self.dm_layer.OPD     = opd_highres.cpu().numpy().reshape(self.dm_layer.D_px, self.dm_layer.D_px)

        # Saturate the actuation
        self.dm_layer.OPD = self.saturateShape(self.dm_layer.OPD)

        return True
    
    def set_actuator_misregistration(self, shiftX=0.0, shiftY=0.0, rotation=0.0, radialScaling=1.0):
        """
        Exact static misregistration of a DM with analytic (Gaussian, Partition of Unity) influence functions: the actuator coordinates
        are transformed, x' = s R(rotation) x + (shiftX, shiftY), the Gaussian width is scaled by s, and the influence functions are
        recomputed on them, so the DM shape is generated already misregistered (no interpolation of the OPD map). Use it instead of misReg.set_params; the OPD-map
        misregistration (misReg) must stay null. The new shape applies from the next updateDMShape.

        Parameters
        ----------
        shiftX, shiftY : float
            Shift [m] in the DM plane.
        rotation : float
            Rotation [deg] around the centre, same sense as misReg (clockwise for a positive angle with x right and y up).
        radialScaling : float
            Magnification (1 = none).

        Notes
        -----
        The shifts are in metres; the DM grid step is the telescope pixel D/resolution, the same one misReg uses.
        """
        if getattr(self, 'custom_IFs', False):
            raise NotImplementedError('DeformableMirror::set_actuator_misregistration - only for analytic influence functions.')
        # Same sense of rotation as the OPD-map misregistration (MisRegistration.apply_misreg, scipy.ndimage.rotate): a positive angle
        # turns the DM clockwise in the (x right, y up) DM coordinates
        th = -np.deg2rad(rotation)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        coords = radialScaling * (self.coordinates[self.validAct] @ rot.T) + np.array([shiftX, shiftY])
        # A magnification of the DM image scales the influence functions too (their width ~ 1/epsilon), not only the actuator positions:
        # the shape is then exactly the unmisregistered one evaluated at T^-1(x), including the Partition of Unity normalisation
        self.influence_functions = self.precomputeInfluenceFunctions(coords, self.high_res_coords, self.epsilon / radialScaling)
        self.logger.info(f'DeformableMirror::set_actuator_misregistration - shift ({shiftX:.3e}, {shiftY:.3e}) m, rotation {rotation} deg, '
                         f'scaling {radialScaling}: influence functions recomputed.')

    def updateMisreg(self, elapsedTime):
        """
        Update the mis-registration params by the temporal factor.

        Parameters
        ----------
        elapsedTime : float
            Time elapsed in seconds.

        Returns
        -------
        bool
            True if updated successfully.
        """
        self.misReg.update_params(elapsedTime)

        return True
            
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
        self.logger.info('{: ^21s}'.format('Pitch')                                    + '{: ^18s}'.format(str(self.pitch))                    +'{: ^18s}'.format('[m]'))
        self.logger.info('{: ^21s}'.format('Mechanical Coupling')                      + '{: ^18s}'.format(str(self.mechCoupling))             +'{: ^18s}'.format('[%]' ))
        self.logger.info('Mis-registration:')
        self.misReg.print_properties()

    
