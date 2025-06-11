import numpy as np
from numpy.random import RandomState

"""
Layer Module
=================

This module contains the `LayerClass` class, used as the base atmosphere layer class to build a full atmosphere in adaptive optics simulations.
"""

class LayerClass:
    def __init__(self):
        """
        Initialize a LayerClass object representing a single turbulent atmospheric layer.

        Attributes
        ----------
        id : int
            Layer identifier.
        seed : int
            Random number generator seed.
        D_fov : float
            Field of view diameter [m].
        spatial_res : float
            Spatial resolution of the layer [m/px].
        npix : int
            Number of pixels in the layer.
        fractionalR0 : float
            Contribution of this layer to the overall r0.            
        screen : Von Karman infinite screen object
            The screen object representing the turbulence in this layer.
        windSpeed : float
            Wind speed at this layer [m/s].
        windDirection : float
            Wind direction at this layer [degrees].
        altitude : float
            Altitude of the layer [m].
        displ_buffer_x : float
            Buffer storing the cummulative displacement in X-axis
        displ_buffer_y : float
            Buffer storing the cummulative displacement in Y-axis            
        """
        # The Layer class defines the main variables that are required to constitute a layer of turbulence.
        # The atmosphere class is made of several layer objects.

        # Scalar variables 
        self.id = 0
        self.seed = 0
        self.D_fov = 0
        self.spatial_res = 0
        self.npix = 0
        self.fractionalR0 = 0
        self.screen = None
        self.windSpeed = 0
        self.windDirection = 0
        self.altitude = 0

        self.displ_buffer_x = 0
        self.displ_buffer_y = 0
