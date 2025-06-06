"""
Created + major update on March 24 2025
@author: nrodlin
"""

from SAOS.Source import Source
from SAOS.Asterism import Asterism

import numpy as np
from importlib.resources import files
from astropy.io import fits

import logging
import logging.handlers
from queue import Queue

"""
ExtendedSource Module
=================

This module contains the `ExtendedSource` class, used for modeling an extended source in adaptive optics simulations. Typical case is a sun patch.
"""

class ExtendedSource(Source):
    def __init__(self, optBand:str,
                 coordinates:list = [0,0],
                 fov=10,
                 altitude:float = np.inf,
                 img_path="",
                 img_PS=1/60,
                 nSubDirs=1,
                 maxnSubDirs=7,
                 patch_padding=5,
                 subDir_margin=1.0,
                 logger=None):
        """
        Initialize an extended source (e.g., the Sun) for use in AO simulations.

        Parameters
        ----------
        optBand : str
            Optical band identifier.
        coordinates : list, optional
            Sky position [zenith, azimuth] in [arcsec, degrees], by default [0, 0].
        fov : float, optional
            Field of view in arcseconds, by default 10.
        altitude : float, optional
            Altitude of the source in meters. Default is infinity.
        img_path : str, optional
            Path to the FITS image of the source (e.g., solar image).
        img_PS : float, optional
            Plate scale in arcsec/pixel, by default 1/60.
        nSubDirs : int, optional
            Number of sub-directions across the extended source.
        maxnSubDirs : int, optional
            Maximum number of sub-directions allowed for processing limits.        
        patch_padding : float, optional
            Padding beyond FOV for sub-region extraction.
        subDir_margin : float, optional
            Margin added to avoid boundary effects.
        logger : logging.Logger, optional
            Optional logger instance.
        """        
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger
        
        self.is_initialized = False
        tmp = self.photometry(optBand)
        
        tmp             = self.photometry(optBand)              # get the photometry properties
        self.optBand    = optBand                               # optical band
        self.wavelength = tmp[0]                                # wavelength in m
        self.bandwidth  = tmp[1]                                # optical bandwidth
        self.nPhoton    =  tmp[2]                               # photon per m2 per s
        self.fluxMap    = []                                    # 2D flux map of the source
        self.tag        = 'sun'                                 # tag of the object
        self.altitude = altitude                                # altitude of the source object in m    
        self.coordinates = coordinates                          # polar coordinates [r,theta] 
        self.fov = fov                                          # Field of View in arcsec of the patch selected
        self.img_path = img_path                                # Sun pattern that shall be uploaded, expected FITS file.
        self.img_PS = img_PS                                    # Plate Scale in "/px of the input image, default matches the default img_path PS
        self.nSubDirs = nSubDirs                                # Number of sub-directions taken from the sun image to build the inner aberration of the pupil in the image
        self.patch_padding = patch_padding                      # Padding outside the subaperture FoV in arcsec.
        self.subDir_margin = subDir_margin                                # Extra margin to the subDirs size to avoid border effects [arcsec]

        self.type     = 'SUN'

        if self.img_path == "":
            images_dir = files('SAOS.images')
            self.img_path = images_dir / 'imsol.fits'
           
        self.sun_nopad, self.sun_padded = self.load_sun_img()   # Take sun patch, pad + no pad

        if self.nSubDirs > maxnSubDirs:
            raise ValueError("ExtendedSource::__init__ - Too many subdirections, processing will not be feasible")
        
        self.subDirs_coordinates, self.subDirs_sun = self.get_subDirs() # Coordinates are in polar [r, theta(radians)] + FoV [arcsec]
                                                                        # The origin is the telescope axis.

        # Finally, create an Asterism using the coordinates of the subdirections to monitor the atmosphere in those lines of sight

        subDirs_stars = []

        for dirX in range(self.nSubDirs):
            for dirY in range(self.nSubDirs):
                subDirs_stars.append(Source(optBand=self.optBand, 
                                            magnitude=1, 
                                            coordinates=[self.subDirs_coordinates[0,dirX,dirY], self.subDirs_coordinates[1,dirX,dirY]],
                                            logger=self.logger))
                subDirs_stars[-1].nPhoton = np.round(self.nPhoton/(self.nSubDirs*self.nSubDirs))
        
        self.sun_subDir_ast = Asterism(subDirs_stars)

        # Last step, define the 2D filter that will be used to combine the subDirs. 
        # Taken from the WideField module of DASP (Durham Adaptive Optics Simulator, Alaister Basedn et al.)
        filt_width = np.round((self.subDirs_coordinates[2,0,0])/self.img_PS).astype(int)
        self.filter_2D = np.zeros((filt_width, filt_width, self.nSubDirs, self.nSubDirs))

        lin = 1 - np.abs(np.arange(filt_width) - (filt_width - 1) / 2) / ((filt_width - 1) / 2)
        filter_2D_template = np.tile(lin, (filt_width, 1))
        filter_2D_template = filter_2D_template * lin[:, np.newaxis]
        
        subDir_size = filt_width

        for dirX in range(self.nSubDirs):
            for dirY in range(self.nSubDirs):
                self.filter_2D[:,:,dirX,dirY] = np.copy(filter_2D_template)

                if dirX == 0: # top, add top left and right to match the external contributions to the filter
                    # top
                    self.filter_2D[0:subDir_size//2,:,dirX,dirY] += filter_2D_template[-(subDir_size//2):,:] # add bottom
                if dirX == (self.nSubDirs-1): # bottom, add top to match the external contributions to the filter
                    # bottom
                    self.filter_2D[-(subDir_size//2):,:,dirX,dirY] += filter_2D_template[0:subDir_size//2,:] # add top
                if dirY == 0: # left, add right match the external contributions to the filter
                    # left
                    self.filter_2D[:,0:subDir_size//2,dirX,dirY] += filter_2D_template[:,-(subDir_size//2):] # add right
                if dirY == (self.nSubDirs-1): # right, add left match the external contributions to the filter
                    # right
                    self.filter_2D[:,-(subDir_size//2):,dirX,dirY] += filter_2D_template[:,0:subDir_size//2] # add left

                # Corner cases:

                if (dirX == 0 and dirY == 0): # top-left
                    self.filter_2D[0:subDir_size//2,0:subDir_size//2,dirX,dirY] = np.max(filter_2D_template)
                if (dirX == 0 and dirY == (self.nSubDirs-1)): # top-right
                    self.filter_2D[0:subDir_size//2,-(subDir_size//2):,dirX,dirY] = np.max(filter_2D_template)
                if (dirX == (self.nSubDirs-1) and dirY == 0): # bottom-left
                    self.filter_2D[-(subDir_size//2):,0:subDir_size//2,dirX,dirY] = np.max(filter_2D_template)
                if (dirX == (self.nSubDirs-1) and dirY == (self.nSubDirs-1)): # bottom-right
                    self.filter_2D[-(subDir_size//2):,-(subDir_size//2):,dirX,dirY] = np.max(filter_2D_template)
                       
        self.is_initialized = True
        
    def photometry(self,arg):
        """
        Return photometric properties specific to solar sources.

        Parameters
        ----------
        arg : str
            Photometric band identifier.

        Returns
        -------
        list
            [wavelength, bandwidth, zero point flux] or -1 if not found.
        """
        self.logger.debug('ExtendedSource::photometry')
        # photometry object [wavelength, bandwidth, zeroPoint]
        class phot:
            pass
        
        # Source: Serpone, Nick & Horikoshi, Satoshi & Emeline, Alexei. (2010). 
        # Microwaves in advanced oxidation processes for environmental applications. A brief review. 
        # Journal of Photochemistry and Photobiology C-photochemistry Reviews - J PHOTOCHEM PHOTOBIOL C-PHOTO. 11. 114-131. 10.1016/j.jphotochemrev.2010.07.003. 
        phot.V     =  [0.500e-6, 0.0, 3.31e12]#[0.500e-6, 0.0, 3.5e20]
        phot.R     =  [0.700e-6, 0.0, 2.25e20]
        phot.IR    =  [1.300e-6, 0.0, 0.5e20]
         
        if isinstance(arg,str):
            if hasattr(phot,arg):
                return getattr(phot,arg)
            else:
                self.logger.error('ExtendedSource::photometry - Wrong name for the photometry object.')
                raise ValueError('Wrong name for the photometry object.')
        else:
            self.logger.error('ExtendedSource::photometry - The photometry object takes a scalar as an input.')
            raise ValueError('The photometry object takes a scalar as an input.')

    def print_properties(self):
        """
        Print key attributes of the extended source.

        Returns
        -------
        None
        """
        self.logger.info('ExtendedSource::print_properties')
        self.logger.info('{: ^8s}'.format('Source') +'{: ^10s}'.format('Wavelength')+ '{: ^8s}'.format('Zenith')+ '{: ^10s}'.format('Azimuth')+ '{: ^10s}'.format('Altitude')+ '{: ^10s}'.format('Flux') )
        self.logger.info('{: ^8s}'.format('') +'{: ^10s}'.format('[m]')+ '{: ^8s}'.format('[arcsec]')+ '{: ^10s}'.format('[deg]')+ '{: ^10s}'.format('[m]')+ '{: ^10s}'.format('[phot/m2/s]') )

        self.logger.info('-------------------------------------------------------------------')        
        self.logger.info('{: ^8s}'.format(self.type) +'{: ^10s}'.format(str(self.wavelength))+ '{: ^8s}'.format(str(self.coordinates[0]))+ '{: ^10s}'.format(str(self.coordinates[1]))+'{: ^10s}'.format(str(np.round(self.altitude,2)))+'{: ^10s}'.format(str(np.round(self.nPhoton,1))) )

    def load_sun_img(self):
        """
        Load the solar image patch and extract both padded and unpadded versions.

        Returns
        -------
        tuple of np.ndarray
            (unpadded image patch, padded image patch)
        """
        self.logger.debug('ExtendedSource::load_sun_img')
        try:
            tmp_sun = fits.open(self.img_path)[0].data.astype('<f4')
        except:
            self.logger.error('ExtendedSource::load_sun_img - Error loading the FITS file')
            raise FileExistsError('Error loading the FITS file')
        cnt_x_arc = self.coordinates[0] * np.cos(np.deg2rad(self.coordinates[1]))
        cnt_y_arc = self.coordinates[0] * np.sin(np.deg2rad(self.coordinates[1]))

        cnt_x = (cnt_x_arc / self.img_PS) + tmp_sun.shape[0]//2
        cnt_y = (cnt_y_arc / self.img_PS) + tmp_sun.shape[1]//2

        width_subap_nopad = np.round(self.fov / self.img_PS).astype(int)
        width_subap_pad = np.round((self.fov+self.patch_padding+self.subDir_margin) / self.img_PS).astype(int)

        cx_nopad = np.round(cnt_x - width_subap_nopad/2).astype(int)
        cy_nopad = np.round(cnt_y - width_subap_nopad/2).astype(int)

        cx_pad = np.round(cnt_x - width_subap_pad/2).astype(int)
        cy_pad = np.round(cnt_y - width_subap_pad/2).astype(int)

        if cx_pad < 0 or cy_pad < 0 or (cx_pad+width_subap_nopad) > tmp_sun.shape[0] or (cy_pad+width_subap_pad) > tmp_sun.shape[1]:
            raise ValueError(f'The selected patch is out of the image bounds. Image size= {tmp_sun.shape[0]*self.img_PS} arcsec. \
                             Required = {2*np.maximum(np.abs(cnt_x_arc),np.abs(cnt_y_arc)) + (self.fov+self.patch_padding+self.subDir_margin)} arcsec.')

        sun_nopad = tmp_sun[cx_nopad:cx_nopad+width_subap_nopad,cy_nopad:cy_nopad+width_subap_nopad]

        sun_pad = tmp_sun[cx_pad:cx_pad+width_subap_pad,cy_pad:cy_pad+width_subap_pad]

        return sun_nopad, sun_pad
    
    def get_subDirs(self):
        """
        Compute the sub-direction coordinates and extract subregions from the image.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Coordinates [r, theta] and subregion image cubes.
        """
        self.logger.debug('ExtendedSource::get_subDirs')
        subDir_loc = np.zeros((3,self.nSubDirs, self.nSubDirs))
        if self.nSubDirs > 1:
            subDir_size = (2*(self.fov+self.patch_padding)) / (self.nSubDirs+1) # This guarantees a superposition of the 50% of the subdirs
        else:
            subDir_size = self.fov+self.patch_padding
        
        subDir_width = np.round((subDir_size+self.subDir_margin)/self.img_PS).astype(int)
        subDir_imgs = np.zeros((subDir_width, subDir_width, self.nSubDirs, self.nSubDirs))
        
        # Coordinates of the star w.r.t the telescope axis (optical axis)
        tel_x = self.coordinates[0] * np.cos(np.deg2rad(self.coordinates[1]))
        tel_y = self.coordinates[0] * np.sin(np.deg2rad(self.coordinates[1]))
        
        # Patch width 
        patch_width = self.fov+self.patch_padding # in arcsec

        for dirX in range(self.nSubDirs):
            for dirY in range(self.nSubDirs):
                # Crop the subDirs regions
                # Centroid of the subDir w.r.t the top-left corner of the square of size: fov + patch_padding
                cx = ((dirX +1)*(patch_width))/(self.nSubDirs + 1)
                cy = ((dirY +1)*(patch_width))/(self.nSubDirs + 1)

                # Centroid of the subDir w.r.t the top-left corner of the sun_padded image 
                cx = cx - subDir_size/2
                cy = cy - subDir_size/2
                # Change the origin to the sun_padded image of size: fov+patch_padding+subDir_margin
                corner_x = cx + self.subDir_margin/2
                corner_y = cy + self.subDir_margin/2
                # Then, we want to select the cropping region with the margin, so (corner-margin/2):(corner-margin/2 + width)
                # This seems stupid, because the offset in the origin was added and now is removed, but it is a consequence of 
                # the patches selections, so it would be difficult to trace later if we don't keep the full maths in here
                corner_x = corner_x - self.subDir_margin/2
                corner_y = corner_y - self.subDir_margin/2
                # Convert to pixel coordinates
                corner_x = np.round(corner_x/self.img_PS).astype(int)
                corner_y = np.round(corner_y/self.img_PS).astype(int)

                subDir_imgs[:,:,dirX,dirY] = self.sun_padded[corner_x : corner_x+subDir_width, corner_y : corner_y+subDir_width]
                # Store the coordinates on-sky of the subDirs w.r.t optical axis
                # Define subDir origin in the mid of the subDir + displace to iterate through the subDirs + 
                # (-Translation) to define the global axis in the centre of the subaperture (in arcsecs)
                dirx_c = (subDir_size/2) + dirX * (subDir_size/2) - (self.fov+self.patch_padding)/2
                diry_c = (subDir_size/2) + dirY * (subDir_size/2) - (self.fov+self.patch_padding)/2

                # Now, change to polar, moving the origin towards the telescope axis

                subDir_loc[0, dirX, dirY] = np.sqrt((dirx_c+tel_x)**2 + (diry_c+tel_y)**2)
                subDir_loc[1, dirX, dirY] = np.rad2deg(np.arctan2((diry_c+tel_y), (dirx_c+tel_x)))
                subDir_loc[2, dirX, dirY] = subDir_size
        
        return subDir_loc, subDir_imgs

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