# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:01:10 2020

@author: cheritie

Minor update Marh 24 2025
@author: nrodlin
"""

import inspect

"""
Misregistration Module
=================

This module contains the `Misresgistration` class, used for modeling the DM misregistration in adaptive optics simulations.
"""

class MisRegistration:
    def __init__(self,param=None):
        """
        Initialize the MisRegistration object, which defines misalignments of an optical element.

        Parameters
        ----------
        param : dict or MisRegistration, optional
            Dictionary or another MisRegistration object defining the initial misregistration values.
        """        
        self.tag                  = 'misRegistration' 
        self.isInitialized = False  

        if param is None:
            self.rotationAngle        = 0                    # rotation angle in degrees
            self.shiftX               = 0                           # shift X in m
            self.shiftY               = 0                           # shift Y in m
            self.anamorphosisAngle    = 0                # amamorphosis angle in degrees
            self.tangentialScaling    = 0                # normal scaling in % of diameter
            self.radialScaling        = 0                    # radial scaling in % of diameter
        else:              
            if isinstance(param,dict):
                # print('MisRegistration object created from a ditionnary')
                self.rotationAngle        = param['rotationAngle']                    # rotation angle in degrees
                self.shiftX               = param['shiftX']                           # shift X in m
                self.shiftY               = param['shiftY']                           # shift Y in m
                self.anamorphosisAngle    = param['anamorphosisAngle']                # amamorphosis angle in degrees
                self.tangentialScaling    = param['tangentialScaling']                # normal scaling in % of diameter
                self.radialScaling        = param['radialScaling']                    # radial scaling in % of diameter
            else:
                if  inspect.isclass(type(param)):
                    if param.tag == 'misRegistration':
                        # print('MisRegistration object created from an existing Misregistration object')

                        self.rotationAngle        = param.rotationAngle                   # rotation angle in degrees
                        self.shiftX               = param.shiftX                           # shift X in m
                        self.shiftY               = param.shiftY                           # shift Y in m
                        self.anamorphosisAngle    = param.anamorphosisAngle                # amamorphosis angle in degrees
                        self.tangentialScaling    = param.tangentialScaling                # normal scaling in % of diameter
                        self.radialScaling        = param.radialScaling  
                    else:
                        print('wrong type of object passed to a MisRegistration object')
                else:
                    print('wrong type of object passed to a MisRegistration object')
                    
                
                
        if self.radialScaling ==0 and self.tangentialScaling ==0:
            self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                              'sX_'         + str('%.2f' %(self.shiftX))                 +'_m_'\
                              'sY_'         + str('%.2f' %(self.shiftY))                 +'_m_'\
                              'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                              'mR_'    + str('%.2f' %(self.radialScaling+1.))       +'_'\
                              'mT_'   + str('%.2f' %(self.tangentialScaling+1.)) 
        else:
             self.misRegName = 'rot_'       + str('%.2f' %self.rotationAngle)            +'_'\
                          'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                          'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                          'anam_'      + str('%.2f' %self.anamorphosisAngle)        +'_'\
                          'mR_'        + str('%.4f' %(self.radialScaling+1.))       +'_'\
                          'mT_'       + str('%.4f' %(self.tangentialScaling+1.)) 
                            
        self.isInitialized = True

#        mis-registrations can be added or sub-tracted using + and -       
    def __add__(self,misRegObject):
        """
        Combine two MisRegistration objects by summing their parameters.

        Parameters
        ----------
        misRegObject : MisRegistration
            Another misregistration instance.

        Returns
        -------
        MisRegistration
            Resulting combined misregistration.
        """
        if misRegObject.tag == 'misRegistration':
            tmp = MisRegistration()
            tmp.rotationAngle        = self.rotationAngle       + misRegObject.rotationAngle
            tmp.shiftX               = self.shiftX              + misRegObject.shiftX
            tmp.shiftY               = self.shiftY              + misRegObject.shiftY
            tmp.anamorphosisAngle    = self.anamorphosisAngle   + misRegObject.anamorphosisAngle
            tmp.tangentialScaling    = self.tangentialScaling   + misRegObject.tangentialScaling
            tmp.radialScaling        = self.radialScaling       + misRegObject.radialScaling       
        else:
            print('Error you are trying to combine a MisRegistration Object with the wrong type of object')
        return tmp
    
    def __sub__(self,misRegObject):
        """
        Subtract the misregistration parameters of another object.

        Parameters
        ----------
        misRegObject : MisRegistration
            Another misregistration instance.

        Returns
        -------
        MisRegistration
            Resulting differential misregistration.
        """
        if misRegObject.tag == 'misRegistration':
            tmp = MisRegistration()
            
            tmp.rotationAngle        = self.rotationAngle       - misRegObject.rotationAngle
            tmp.shiftX               = self.shiftX              - misRegObject.shiftX
            tmp.shiftY               = self.shiftY              - misRegObject.shiftY
            tmp.anamorphosisAngle    = self.anamorphosisAngle   - misRegObject.anamorphosisAngle
            tmp.tangentialScaling    = self.tangentialScaling   - misRegObject.tangentialScaling
            tmp.radialScaling        = self.radialScaling       - misRegObject.radialScaling
        else:
            print('Error you are trying to combine a MisRegistration Object with the wrong type of object')
        return tmp
    
    def __eq__(self, other):
        """
        Check equality between two MisRegistration objects.

        Parameters
        ----------
        other : MisRegistration
            Another instance.

        Returns
        -------
        bool
            True if all parameters match.
        """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        """
        Check inequality between two MisRegistration objects.

        Parameters
        ----------
        other : MisRegistration
            Another instance.

        Returns
        -------
        bool
            True if any parameter differs.
        """
        return not self.__eq__(other)
# ----------------------------------------- Properties  -----------------------------------------
    @property
    def rotationAngle(self):
        return self._rotationAngle
        
    @rotationAngle.setter
    def rotationAngle(self,val):
        self._rotationAngle  = val
        if self.isInitialized: 
            if self.radialScaling ==0 and self.tangentialScaling ==0:
                self.misRegName        = 'rot_'        + str('%.2f' %val)            +'_'\
                          'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                          'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                          'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                          'mR_'      + str('%.2f' %(self.radialScaling+1.))       +'_'\
                          'mT_'  + str('%.2f' %(self.tangentialScaling+1.))     
            else:
                self.misRegName        = 'rot_'        + str('%.2f' %val)            +'_'\
                          'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                          'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                          'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                          'mR_'      + str('%.4f' %(self.radialScaling+1.))       +'_'\
                          'mT_'  + str('%.4f' %(self.tangentialScaling+1.))  
        
    @property
    def shiftX(self):
        return self._shiftX
        
    @shiftX.setter
    def shiftX(self,val):
        self._shiftX  = val
        if self.isInitialized:  
            if self.radialScaling ==0 and self.tangentialScaling ==0:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)    +'_'\
                      'sX_'             + str('%.2f' %(val))                         +'_m_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mR_'      + str('%.2f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.2f' %(self.tangentialScaling+1.))     
            else:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)    +'_'\
                      'sX_'             + str('%.2f' %(val))                         +'_m_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mR_'      + str('%.4f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.4f' %(self.tangentialScaling+1.))             
    
    
    
    @property
    def shiftY(self):
        return self._shiftY
        
    @shiftY.setter
    def shiftY(self,val):
        self._shiftY  = val
        if self.isInitialized:  
            if self.radialScaling ==0 and self.tangentialScaling ==0:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)    +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                      'sY_'             + str('%.2f' %(val))                         +'_m_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mR_'      + str('%.2f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.2f' %(self.tangentialScaling+1.))   
            else:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)    +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                      'sY_'             + str('%.2f' %(val))                         +'_m_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mR_'      + str('%.4f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.4f' %(self.tangentialScaling+1.))   
                      
    @property
    def anamorphosisAngle(self):
        return self._anamorphosisAngle
        
    @anamorphosisAngle.setter
    def anamorphosisAngle(self,val):
        self._anamorphosisAngle  = val
        if self.isInitialized:  
            if self.radialScaling ==0 and self.tangentialScaling ==0:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                      'anam_'  + str('%.2f' %val)        +'_'\
                      'mR_'      + str('%.2f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.2f' %(self.tangentialScaling+1.))   
            else:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                      'anam_'  + str('%.2f' %val)        +'_'\
                      'mR_'      + str('%.4f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.4f' %(self.tangentialScaling+1.))   
                  
                  
    @property
    def radialScaling(self):
        return self._radialScaling
        
    @radialScaling.setter
    def radialScaling(self,val):
        self._radialScaling  = val
        if self.isInitialized:  
            if self.radialScaling ==0 and self.tangentialScaling ==0:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mR_'      + str('%.2f' %(val+1.))       +'_'\
                      'mT_'  + str('%.2f' %(self.tangentialScaling+1.))   
            else:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_m_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_m_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mR_'      + str('%.4f' %(val+1.))       +'_'\
                      'mT_'  + str('%.4f' %(self.tangentialScaling+1.))   
                  
                  
    @property
    def tangentialScaling(self):
        return self._tangentialScaling
        
    @tangentialScaling.setter
    def tangentialScaling(self,val):
        self._tangentialScaling  = val
        if self.isInitialized:  
            if self.radialScaling ==0 and self.tangentialScaling ==0:

                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                      'X_'             + str('%.2f' %(self.shiftX))                 +'_'\
                      'Y_'             + str('%.2f' %(self.shiftY))                 +'_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mN_'      + str('%.2f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.2f' %(val+1.))   
            else:
                self.misRegName = 'rot_'        + str('%.2f' %self.rotationAngle)            +'_'\
                      'sX_'             + str('%.2f' %(self.shiftX))                 +'_'\
                      'sY_'             + str('%.2f' %(self.shiftY))                 +'_'\
                      'anam_'  + str('%.2f' %self.anamorphosisAngle)        +'_'\
                      'mN_'      + str('%.4f' %(self.radialScaling+1.))       +'_'\
                      'mT_'  + str('%.4f' %(val+1.))      
                  
                  
                  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def print_(self):
        print('{: ^14s}'.format('Rotation [deg]') + '\t' + '{: ^11s}'.format('Shift X [m]')+ '\t' + '{: ^11s}'.format('Shift Y [m]')+ '\t' + '{: ^18s}'.format('Radial Scaling [%]')+ '\t' + '{: ^22s}'.format('Tangential Scaling [%]'))
        print("{: ^14s}".format(str(self.rotationAngle) )  + '\t' +'{: ^11s}'.format(str(self.shiftX))+'\t' + '{: ^11s}'.format(str(self.shiftY)) +'\t' +'{: ^18s}'.format(str(self.radialScaling))+'\t' + '{: ^22s}'.format(str(self.tangentialScaling)))
     
