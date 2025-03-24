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
        D : float
            Diameter of the pupil [m].
        D_fov : float
            Field of view diameter [m].
        altitude : float
            Altitude of the layer [m].
        d0 : float
            Characteristic scale [m].
        direction : float
            Wind direction in radians or degrees.
        extra_sx, extra_sy : int
            Offsets due to sampling shifts in x and y.
        nExtra : int
            Number of extra pixels around the layer.
        nPixel : int
            Number of pixels across the layer.
        notDoneOnce : bool
            Initialization flag.
        resolution : int
            Resolution of the simulation grid.
        resolution_fov : int
            Resolution for the full field-of-view.
        seed : int
            Random number generator seed.
        vX, vY : float
            Wind velocity components.
        windSpeed : float
            Wind speed at this layer [m/s].
        fractionalR0 : float
            Contribution of this layer to the overall r0.

        Covariance matrices and phase screen arrays:
        A, B, BBt, XXt, XXt_r0, ZXt, ZXt_r0, ZZt, ZZt_inv,
        ZZt_inv_r0, ZZt_r0, initialPhase, mapShift : np.ndarray
            Intermediate matrices for phase screen generation.
        innerMask, outerMask : np.ndarray
            Masks to define valid pixels inside/outside the pupil.
        innerZ, outerZ : np.ndarray
            Zonal representation inside/outside.
        ratio : np.ndarray
            Ratio of various sampling metrics.
        randomState : np.random.RandomState
            Random state used for reproducibility.
        """
        # The Layer class defines the main variables that are required to constitute a layer of turbulence.
        # The atmosphere class is made of several layer objects.

        # Scalar variables 
        self.id = 0
        self.D = 0
        self.D_fov = 0
        self.altitude = 0
        self.d0 = 0
        self.direction = 0
        self.extra_sx = 0
        self.extra_sy = 0
        self.nExtra = 0
        self.nPixel = 0
        self.notDoneOnce = False
        self.resolution = 0
        self.resolution_fov = 0
        self.seed = 0
        self.vX = 0
        self.vY = 0
        self.windSpeed = 0
        self.fractionalR0 = 0
        # Arrays 
        self.A = np.array([])
        self.B = np.array([])
        self.BBt = np.array([])
        self.XXt = np.array([])
        self.XXt_r0 = np.array([])
        self.ZXt = np.array([])
        self.ZXt_r0 = np.array([])
        self.ZZt = np.array([])
        self.ZZt_inv = np.array([])
        self.ZZt_inv_r0 = np.array([])
        self.ZZt_r0 = np.array([])
        self.initialPhase = np.array([])
        self.innerMask = np.array([])
        self.innerZ = np.array([])
        self.mapShift = np.array([])
        self.outerMask = np.array([])
        self.outerZ = np.array([])
        self.ratio = np.zeros(2)
        self.randomState = RandomState(0)