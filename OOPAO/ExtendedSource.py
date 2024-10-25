from OOPAO.Source import Source
from OOPAO.Asterism import Asterism
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

class ExtendedSource(Source):
    def __init__(self, optBand:str,
                 coordinates:list = [0,0],
                 fov=10,
                 altitude:float = np.inf, 
                 display_properties:bool=True,
                 chromatic_shift:list = None,
                 img_path="OOPAO/OOPAO/images/imsol.fits",
                 img_PS=1/60,
                 nSubDirs=1,
                 patch_padding=2,
                 ):
        """SOURCE
        A extended source object shares the implementation of the Source class, but the photometry varies for the Sun. 
        The LGS and Magnitude parameters are omitted in this definition

        Parameters
        ----------
        optBand : str
            The optical band of the source (see the method photometry)
            ex, 'V0' corresponds to a wavelength of 500 nm
        coordinates : list, optional
            DESCRIPTION. The default is [0,0], in polar format [r, theta] in which theta is in degrees.
        fov : float
            DESCRIPTION. Field of View in arcsec of the patch selected
        altitude : float, optional
            DESCRIPTION. The default is np.inf.
        display_properties : bool, optional
            DESCRIPTION. The default is True.
        chromatic_shift : list, optional
            DESCRIPTION. The default is None.
        img_path: str, optional
            DESCRIPTION. Sun pattern that shall be uploaded, expected FITS file.
        img_PS: str, optional
            DESCRIPTION. Plate Scale in "/px of the input image, default matches the default img_path PS
        nSubDirs: int, optional
            DESCRIPTION. Number of sub-directions taken from the sun image to build the inner aberration of the pupil in the image
        patch_padding: float, optional
            DESCRIPTION. Padding outside the subaperture FoV in arcsec.
        Returns
        -------
        None.

        """
        """
        ************************** REQUIRED PARAMETERS **************************
        
        A Source object is characterised by two parameter:
        _ optBand               : the optical band of the source (see the method photometry)
                            
        ************************** COUPLING A SOURCE OBJECT **************************
        
        Once generated, a Source object "src" can be coupled to a Telescope "tel" that contains the OPD.
        _ This is achieved using the * operator     : src*tel
        _ It can be accessed using                  : tel.src       

    
        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Source object are listed here: 
        _ src.phase     : 2D map of the phase scaled to the src wavelength corresponding to tel.OPD
        _ src.type      : SUN  

        _ src.nPhoton   : number of photons per m2 per s, can be updated online. 
        _ src.fluxMap   : 2D map of the number of photons per pixel per frame (depends on the loop frequency defined by tel.samplingTime)  
        _ src.display_properties : display the properties of the src object
        _ src.chromatic_shift : list of shift in arcesc to be applied to the pupil footprint at each layer of the atmosphere object. 
        
        The main properties of the object can be displayed using :
            src.print_properties()
            
        ************************** OPTIONAL PROPERTIES **************************
        _ altitude              : altitude of the source. Default is inf (SUN) 
        ************************** EXEMPLE **************************

        Creates a sun object in V band and combine it to the telescope
        src = ExtendedSource(opticalBand = 'V0') 
        src*tel
       
        """
        self.is_initialized = False
        tmp = self.photometry(optBand)
        
        self.display_properties = display_properties
        tmp             = self.photometry(optBand)              # get the photometry properties
        self.optBand    = optBand                               # optical band
        self.wavelength = tmp[0]                                # wavelength in m
        self.bandwidth  = tmp[1]                                # optical bandwidth
        self.nPhoton    =  tmp[2]                               # photon per m2 per s
        self.phase      = []                                    # phase of the source 
        self.phase_no_pupil      = []                           # phase of the source (no pupil)
        self.fluxMap    = []                                    # 2D flux map of the source
        self.tag        = 'sun'                                 # tag of the object
        self.altitude = altitude                                # altitude of the source object in m    
        self.coordinates = coordinates                          # polar coordinates [r,theta] 
        self.chromatic_shift = chromatic_shift                  # shift in arcsec to be applied to the atmospheric phase screens (one value for each layer) to simulate a chromatic effect
        self.fov = fov                                          # Field of View in arcsec of the patch selected
        self.img_path = img_path                                # Sun pattern that shall be uploaded, expected FITS file.
        self.img_PS = img_PS                                    # Plate Scale in "/px of the input image, default matches the default img_path PS
        self.nSubDirs = nSubDirs                                # Number of sub-directions taken from the sun image to build the inner aberration of the pupil in the image
        self.patch_padding = patch_padding                      # Padding outside the subaperture FoV in arcsec.

        self.type     = 'SUN'

        if self.display_properties:
            self.print_properties()
            
        self.sun_nopad, self.sun_padded = self.load_sun_img()   # Take sun patch, pad + no pad

        if self.nSubDirs > 5:
            raise BaseException("Too many directions, processing will not be feasible")
        
        self.subDirs_coordinates, self.subDirs_sun = self.get_subDirs() # Coordinates are in polar [r, theta(radians)] + FoV [arcsec]
                                                                        # The origin is the telescope axis.

        # Finally, create an Asterism using the coordinates of the subdirections to monitor the atmosphere in those lines of sight

        subDirs_stars = []

        for dirX in range(self.nSubDirs):
            for dirY in range(self.nSubDirs):
                subDirs_stars.append(Source(optBand=self.optBand, 
                                            magnitude=1, 
                                            coordinates=[self.subDirs_coordinates[0,dirX,dirY], self.subDirs_coordinates[1,dirX,dirY]], 
                                            display_properties=False))
        
        self.sun_subDir_ast = Asterism(subDirs_stars)

        self.is_initialized = True
        
    def photometry(self,arg):
        # photometry object [wavelength, bandwidth, zeroPoint]
        class phot:
            pass
        
        # Source: Serpone, Nick & Horikoshi, Satoshi & Emeline, Alexei. (2010). 
        # Microwaves in advanced oxidation processes for environmental applications. A brief review. 
        # Journal of Photochemistry and Photobiology C-photochemistry Reviews - J PHOTOCHEM PHOTOBIOL C-PHOTO. 11. 114-131. 10.1016/j.jphotochemrev.2010.07.003. 
        phot.V =  [0.500e-6, 0.0, 3.5e20]
        phot.R =  [0.700e-6, 0.0, 2.25e20]
        phot.IR = [1.300e-6, 0.0, 0.5e20]
        
        if isinstance(arg,str):
            if hasattr(phot,arg):
                return getattr(phot,arg)
            else:
                print('Error: Wrong name for the photometry object')
                return -1
        else:
            print('Error: The photometry object takes a scalar as an input')
            return -1   
    @property
    def nPhoton(self):
        return self._nPhoton       
    @nPhoton.setter
    def nPhoton(self,val):
        self._nPhoton  = val
        if self.is_initialized:

            print('NGS flux updated!')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Wavelength \t'+str(round(self.wavelength*1e6,3)) + ' \t [microns]') 
            print('Optical Band \t'+self.optBand) 
            print('Flux \t\t'+ str(np.round(self.nPhoton)) + str('\t [photons/m2/s]'))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^8s}'.format('Source') +'{: ^10s}'.format('Wavelength')+ '{: ^8s}'.format('Zenith')+ '{: ^10s}'.format('Azimuth')+ '{: ^10s}'.format('Altitude')+ '{: ^10s}'.format('Flux') )
        print('{: ^8s}'.format('') +'{: ^10s}'.format('[m]')+ '{: ^8s}'.format('[arcsec]')+ '{: ^10s}'.format('[deg]')+ '{: ^10s}'.format('[m]')+ '{: ^10s}'.format('[phot/m2/s]') )

        print('-------------------------------------------------------------------')        
        print('{: ^8s}'.format(self.type) +'{: ^10s}'.format(str(self.wavelength))+ '{: ^8s}'.format(str(self.coordinates[0]))+ '{: ^10s}'.format(str(self.coordinates[1]))+'{: ^10s}'.format(str(np.round(self.altitude,2)))+'{: ^10s}'.format(str(np.round(self.nPhoton,1))) )
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def load_sun_img(self):
        tmp_sun = fits.open(self.img_path)[0].data

        cx = (self.coordinates[0] * np.cos(np.deg2rad(self.coordinates[1])) / self.img_PS) + tmp_sun.shape[0]//2
        cy = (self.coordinates[0] * np.sin(np.deg2rad(self.coordinates[1])) / self.img_PS) + tmp_sun.shape[1]//2

        width_subap_nopad = self.fov / self.img_PS
        width_subap_pad = (self.fov+self.patch_padding) / self.img_PS

        sun_nopad = tmp_sun[np.round(cx-width_subap_nopad/2).astype(int):np.round(cx+width_subap_nopad/2).astype(int),
                            np.round(cy-width_subap_nopad/2).astype(int):np.round(cy+width_subap_nopad/2).astype(int)]
        sun_pad = tmp_sun[np.round(cx-width_subap_pad/2).astype(int):np.round(cx+width_subap_pad/2).astype(int),
                          np.round(cy-width_subap_pad/2).astype(int):np.round(cy+width_subap_pad/2).astype(int)]

        return sun_nopad, sun_pad
    
    def get_subDirs(self):
        subDir_loc = np.zeros((3,self.nSubDirs, self.nSubDirs))
        if self.nSubDirs > 1:
            subDir_size = (2*(self.fov+self.patch_padding)) / (self.nSubDirs+1) # This guarantees a superposition of the 50% of the subdirs
        else:
            subDir_size = self.fov+self.patch_padding
        
        subDir_imgs = np.zeros((np.ceil(subDir_size/self.img_PS).astype(int), np.ceil(subDir_size/self.img_PS).astype(int), self.nSubDirs, self.nSubDirs))
        
        tel_x = self.coordinates[0] * np.cos(np.deg2rad(self.coordinates[1]))
        tel_y = self.coordinates[0] * np.sin(np.deg2rad(self.coordinates[1]))
        
        for dirX in range(self.nSubDirs):
            for dirY in range(self.nSubDirs):
                # Define subDir origin in the mid of the subDir + displace to iterate through the subDirs + 
                # (-Translation) to define the global axis in the centre of the subaperture
                dirx_c = (subDir_size/2) + dirX * (subDir_size/2) +  - (self.fov+self.patch_padding)/2
                diry_c = (subDir_size/2) + dirY * (subDir_size/2) +  - (self.fov+self.patch_padding)/2

                # Now, change to polar, moving the origin towards the telescope axis

                subDir_loc[0, dirX, dirY] = np.sqrt((dirx_c+tel_x)**2 + (diry_c+tel_y)**2)
                subDir_loc[1, dirX, dirY] = np.rad2deg(np.arctan2((diry_c+tel_y), (dirx_c+tel_x)))
                subDir_loc[2, dirX, dirY] = subDir_size

                # Finally, crop the region

                cx = (dirx_c / self.img_PS) + self.sun_padded.shape[0]//2
                cy = (diry_c / self.img_PS) + self.sun_padded.shape[1]//2               

                subDir_imgs[:,:,dirX,dirY] = self.sun_padded[np.round(cx-subDir_size/(2*self.img_PS)).astype(int):np.round(cx+subDir_size/(2*self.img_PS)).astype(int),
                                                             np.round(cy-subDir_size/(2*self.img_PS)).astype(int):np.round(cy+subDir_size/(2*self.img_PS)).astype(int)]       
        
        """         fig, axs = plt.subplots(self.nSubDirs, self.nSubDirs, squeeze=False)
        print(axs.shape)
        for dirX in range(self.nSubDirs):
            for dirY in range(self.nSubDirs):
                axs[dirX, dirY].imshow(subDir_imgs[:,:,dirX,dirY])
        plt.show() """
        
        return subDir_loc, subDir_imgs

    def __mul__(self,obj):
        print("Here I am")
        if obj.tag =='telescope':
            obj.src   = self
            if type(obj.OPD) is list:
                obj.resetOPD()
            
            if np.ndim(obj.OPD) ==3:
                obj.resetOPD()
    
            obj.OPD = obj.OPD*obj.pupil # here to ensure that a new pupil is taken into account
    
            # update the phase of the source
            self.phase      = obj.OPD*2*np.pi/self.wavelength
            self.phase_no_pupil      = obj.OPD_no_pupil*2*np.pi/self.wavelength
    
            # compute the variance in the pupil
            self.var        = np.var(self.phase[np.where(obj.pupil==1)])
            # assign the source object to the obj object
    
            self.fluxMap    = obj.pupilReflectivity*self.nPhoton*obj.samplingTime*(obj.D/obj.resolution)**2

            for i in range(self.sun_subDir_ast.n_source):
                self.sun_subDir_ast.src[i].fluxMap = self.fluxMap
            if obj.optical_path is None:
                obj.optical_path = []
                obj.optical_path.append([self.type + '('+self.optBand+')',id(self)])
                obj.optical_path.append([obj.tag,id(obj)])
            else:
                obj.optical_path[0] =[self.type + '('+self.optBand+')',id(self)]
                
            return obj
        else:
            raise AttributeError('The Source can only be paired to a Telescope!')