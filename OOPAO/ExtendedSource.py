from OOPAO.Source import Source
import numpy as np
from astropy.io import fits

class ExtendedSource(Source):
    def __init__(self, optBand:str,
                 coordinates:list = [0,0],
                 altitude:float = np.inf, 
                 display_properties:bool=True,
                 chromatic_shift:list = None,
                 img_path="C:/Users/nlinares/Documents/06.Repositorios/simestao/imsol.fits"):
        """SOURCE
        A extended source object shares the implementation of the Source class, but the photometry varies for the Sun. 
        The LGS and Magnitude parameters are omitted in this definition

        Parameters
        ----------
        optBand : str
            The optical band of the source (see the method photometry)
            ex, 'V0' corresponds to a wavelength of 500 nm
        coordinates : list, optional
            DESCRIPTION. The default is [0,0].
        altitude : float, optional
            DESCRIPTION. The default is np.inf.
        display_properties : bool, optional
            DESCRIPTION. The default is True.
        chromatic_shift : list, optional
            DESCRIPTION. The default is None.

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
        self.tag        = 'source'                              # tag of the object
        self.altitude = altitude                                # altitude of the source object in m    
        self.coordinates = coordinates                          # polar coordinates [r,theta] 
        self.chromatic_shift = chromatic_shift                  # shift in arcsec to be applied to the atmospheric phase screens (one value for each layer) to simulate a chromatic effect

        self.type     = 'SUN'
        self.img_path = img_path

        if self.display_properties:
            self.print_properties()
            
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

    def load_sun_img(self, img_path):
        sun = fits.open(img_path)[0].data