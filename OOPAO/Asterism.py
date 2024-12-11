# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:35:32 2022

@author: cheritie
"""

import numpy as np
import inspect
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class Asterism:    
    def __init__(self,list_src):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        An Asterism object is an asterism of Source objects. It requires the following parameters: 
            
        _ list_src              : a list of Source objects that can combine NGS and LGS types
                            
        ************************** COUPLING AN ASTERISM OBJECT **************************
        
        Once generated, an asterism  object "ast" can be coupled to a Telescope "tel" that contains the OPD.
        _ This is achieved using the * operator     : ast*tel
        _ It can be accessed using                  : tel.src       

    
        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ ast.coordinates   : coordinates of the source objects
        _ ast.altitude      : altitude of the source objects
        _ ast.nPhoton       : nPhoton property of the source objects  
                 
        ************************** EXEMPLE **************************

        Create a list of source object in H band with a magnitude 8 and combine it to the telescope
        
        src_1 = Source(opticalBand = 'H', magnitude = 8, coordinates=[0,0]) 
        src_2 = Source(opticalBand = 'H', magnitude = 8, coordinates=[60,0]) 
        src_3 = Source(opticalBand = 'H', magnitude = 8, coordinates=[60,120])
        src_4 = Source(opticalBand = 'H', magnitude = 8, coordinates=[60,240])
        
        ast = Asterism([src_1,src_2,src_3, src_4]

        
        """
        self.n_source = len(list_src)
        self.src = list_src
        self.coordinates = [] # polar, [r, theta] where theta is in [deg]

        self.altitude = []
        self.nPhoton = 0
        self.chromatic_shift = None
        
        self.print_properties()


    def print_properties(self):
        for i in range(self.n_source):
            self.coordinates.append(self.src[i].coordinates)
            self.altitude.append(self.src[i].altitude)
            self.nPhoton += self.src[i].nPhoton/self.n_source
        self.tag = 'asterism'
        self.type = 'asterism'
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ASTERISM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^18s}'.format('Source') +'{: ^18s}'.format('Wavelength')+ '{: ^18s}'.format('Zenith [arcsec]')+ '{: ^18s}'.format('Azimuth [deg]')+ '{: ^18s}'.format('Altitude [m]')+ '{: ^18s}'.format('Magnitude') )
        print('------------------------------------------------------------------------------------------------------------')
        
        for i in range(self.n_source):
            print('{: ^18s}'.format(str(i+1)+' -- '+self.src[i].type) +'{: ^18s}'.format(str(self.src[i].wavelength))+ '{: ^18s}'.format(str(self.src[i].coordinates[0]))+ '{: ^18s}'.format(str(self.src[i].coordinates[1]))+'{: ^18s}'.format(str(np.round(self.src[i].altitude,2)))+ '{: ^18s}'.format(str(self.src[i].magnitude)) )
            print('------------------------------------------------------------------------------------------------------------')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE INTERACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
    def __mul__(self,obj):
        if obj.tag =='telescope': 
            if type(obj.OPD) is not list:
                tmp_OPD = obj.OPD.copy()
                obj.OPD = [tmp_OPD for i in range(self.n_source)]
                
                tmp_OPD = obj.OPD_no_pupil.copy()
                obj.OPD_no_pupil = [tmp_OPD for i in range(self.n_source)]
            for i in range(self.n_source):
                obj.OPD = obj.OPD*obj.pupil # here to ensure that a new pupil is taken into account

                # update the phase of the source
                self.src[i].phase = obj.OPD[i]*2*np.pi/self.src[i].wavelength
                self.src[i].phase_no_pupil = obj.OPD_no_pupil[i]*2*np.pi/self.src[i].wavelength
        
                # compute the variance in the pupil
                self.src[i].var        = np.var(self.src[i].phase[np.where(obj.pupil==1)])
                # assign the source object to the obj object
        
                self.src[i].fluxMap    = obj.pupilReflectivity*self.src[i].nPhoton*obj.samplingTime*(obj.D/obj.resolution)**2
                if obj.optical_path is None:
                    obj.optical_path = []
                    obj.optical_path.append([self.src[i].type + '('+self.src[i].optBand+')',id(self)])
                    obj.optical_path.append([obj.tag,id(obj)])
                else:
                    obj.optical_path[0] =[self.type + '('+self.src[i].optBand+')',id(self)]           
            # assign the source object to the telescope object
            obj.src   = self
            
            return obj
        elif obj.tag == 'atmosphere':
            obj*self
        else:
             raise AttributeError('The Source can only be paired to a Telescope!')
    
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