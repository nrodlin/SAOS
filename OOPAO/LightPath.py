import numpy as np
import torch
from joblib import Parallel, delayed
import time
import logging


class LightPath:
    def __init__(self):
        # Process variables: updated per iteration
        # Optical Path Difference: is the difference in optical path length (OPL) between two rays of light [m]
        # IMPORTANT: DM is considered transmissive, instead of reflective, so there is no need to multiply by 2
        # the physical displacemente of the actuators to take into account the go-and-return path.
        self.OPD = None
        self.OPD_no_pupil = None # removes the pupil, including the spider if there is one.
        # Phase: difference in phase at each point of the space [rad]. Computed as OPD * 2pi/lambda.
        self.phase = None
        self.phase_no_pupil = None # removes the pupil, including the spider if there is one.

        self.slopes_1D = None
        self.slopes_2D = None

        self.sci_frame = None
        self.wfs_frame = None
        # Optical Path components
        self.optical_path =  []
        self.atm = None
        self.wfs = None
        self.dm = None
        self.tel = None
        self.wfs_cam = None
        self.sci_cam = None
        self.ncpa = None
        # Miscelaneous variables
        self.timing = False # Used for debugging purposes, to show the timing of the execution of each step. Printed at the end of each method.
        logging.basicConfig(level=logging.INFO)

    def propagate(self):
        
        # UpdateOPD with the DM and the atmosphere
        tasks = []
        tasks.append(delayed(self.tel.updateOPD)(self.atm, self.src, self.dm))
        tasks.append(delayed(self.dm.dm_propagate)(self.tel))
        if self.ncpa is not None:
            tasks.append(delayed(self.ncpa)(self.tel))

        resuls = Parallel(n_jobs=len(tasks))(tasks)

        # Update Light Path local variables
        self.OPD = self.tel.OPD

        # Compute WFS slopes, WFS PSF, SCI PSF
        tasks = []
        tasks.append(delayed(self.wfs.measure)())
        if (self.wfs_cam is not None):
            tasks.append(delayed(self.wfs_cam.measure)()) 
        
        if (self.sci_cam is not None):
            tasks.append(delayed(self.sci_cam.measure)())
        
        results = Parallel(n_jobs=len(tasks))(tasks)

        # Update Light Path local variables
        return True

    def create_optical_path(self, src, atm, tel, dm, wfs, ngs_cam=None, sci_cam=None, ncpa=None, timing=False):

        # Creating optical path
        logging.info("LightPath::create_optical_path - Creating optical path.")
        
        timing and (t0 := time.time()) # alternative syntax, t0 records the time if timing is True
        self.optical_path = []
        tmp_obj_list = [src, atm, tel, dm, wfs, ngs_cam, ncpa, sci_cam]

        logging.info("LightPath::create_optical_path - Checking compatibility between the components.")

        if src is None or tel is None:
            logging.error("LightPath::create_optical_path - At minimum, source and telescope objects must be correctly defined.")
            return False
        
        compatibility = False

        if src.tag == 'source' or src.tag == 'asterism':
            if wfs.tag == 'pyramid' or wfs.tag == 'shackHartmann' or wfs.tag == 'double_wfs':
                compatibility = True
        elif src.tag == 'sun':
            if wfs.tag == 'correlatingShackHartmann':
                compatibility = True
        
        if compatibility == False:
            logging.error("LightPath::create_optical_path - Incompatible components.")
            return False
        logging.info("LightPath::create_optical_path - Compatible components. Creating path.")
        # Make description of the valid objects (!= None)
        for obj in tmp_obj_list:
            if obj is not None:
                self.optical_path.append([obj.tag, id(obj)])
        
        # Assigning components to the LightPath, if they are not None. 
        # During assignment, the dimensions between the elements are fit. 
        # For instance, an Asterism requires several OPDs, one per star. This is done in here.

        logging.debug("LightPath::create_optical_path - Redimensioning OPD.")

        if src.tag == 'asterism':
            self.OPD = [tel.pupil.astype(float) for i in range(src.n_source)]
            self.OPD_no_pupil = [tel.pupil.astype(float)*0 +1 for i in range(src.n_source)]
        if src.tag == 'sun':
            self.OPD = [tel.pupil.astype(float) for i in range(self.src.sun_subDir_ast.n_source)]
            self.OPD_no_pupil = [tel.pupil.astype(float)*0 +1 for i in range(src.sun_subDir_ast.n_source)]
        else:
            self.OPD = 0*tel.pupil.astype(float)
            self.OPD_no_pupil = 0*tel.pupil.astype(float)
        

        timing and logging.warning(f"LightPath::create_optical_path - Time elapsed: {time.time()-t0} s")
        return True
        
