import numpy as np

class LayerClass:
    def __init__(self):
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