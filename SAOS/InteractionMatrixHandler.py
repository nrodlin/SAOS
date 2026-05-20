import time

import numpy as np
from joblib import Parallel, delayed
import torch

import h5py
import os

from SAOS.modalBasis.zonalModes import generate_zonal_modes
from SAOS.modalBasis.zernikeModes import generate_zernike_modes
from SAOS.modalBasis.karhunenLoeveModes import generate_kl_modes
from SAOS.modalBasis.discHarmonicModes import generate_dh_modes
from SAOS.modalBasis.hadamardModes import generate_hadamard_modes

import logging
import logging.handlers
from queue import Queue

"""
Interaction Matrix Module
=================

This module contains the `InteractionMatrixHandler` class, used for helping wit the measurement of the interaction 
matrix and the modal basis generation in adaptive optics simulations.
"""

class InteractionMatrixHandler:
    def __init__(self, logger):
        """
        Initialize the InteractionMatrixHandler for managing IM acquisition.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance to record operations.
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Class attributes that will be fill during execution        
        self.dm_scanned_list = None
        self.light_path_list = None
        self.im_boolean_matrix = None
        self.interaction_matrix_warehouse = None
        self.modal_basis = []
        

    # Checks the Light Paths defined to knwo the interaction matrices that are necessary and prepares the measurement procedure

    def initialize_im_class(self, light_path_list):
        """
        Analyze the light path setup and prepare the IM scanning structure.

        Parameters
        ----------
        light_path_list : list
            List of LightPath objects.

        Returns
        -------
        bool
            True if initialization succeeds.
        """
        self.logger.debug('interactionMatrixHandler::initialize_im_class')

        # Check if the light path is a list:

        if not isinstance(light_path_list, list):
            light_path_list = [light_path_list]

        # List of modes avaiable:
        self.modal_list = ['zonal', 'zernike', 'kl', 'hadamard', 'dh']
        
        # For efficiency, we need to detect the different DMs of the optical configuration. This will enable this class
        # to move each DM once and measure all the WFS affected, avoiding repeating any movement and saving time.
        im_scan_plan = [] # Stores in each row a list of the DM indeces affecting the light Path. Each row is a different LP, if empty, an IM is not needed.
        self.dm_scanned_list = [] # Stores a link to the DMs objects that are different, so as to recover their properties and command them

        for i in range(len(light_path_list)):
            tmp_dm_light_path_relation = []
            if hasattr(light_path_list[i], 'dm') and (light_path_list[i].wfs is not None):
                if len(self.dm_scanned_list) == 0:
                    for j in range(len(light_path_list[i].dm)):
                        # There are no DMs defined, so add a new one to our list!
                        self.dm_scanned_list.append(light_path_list[i].dm[j])
                        # Register the relation of this DM to the light path
                        tmp_dm_light_path_relation.append(len(self.dm_scanned_list)-1)                        
                else:
                    # We already have DMs defined, so before adding a new one to our list, we need to check whether it is listed or not.
                    dm_already_listed = False

                    for j in range(len(light_path_list[i].dm)):
                        # Check if the DM is in the list
                        for k in range(len(self.dm_scanned_list)):
                            # Compare the properties of this DM with the ones already listed
                            if ((self.dm_scanned_list[k].nActs        == light_path_list[i].dm[j].nActs)     and
                                (self.dm_scanned_list[k].altitude     == light_path_list[i].dm[j].altitude)  and
                                (self.dm_scanned_list[k].nValidAct    == light_path_list[i].dm[j].nValidAct) and
                                (self.dm_scanned_list[k].mechCoupling == light_path_list[i].dm[j].mechCoupling)):
                                
                                dm_already_listed = True
                                # The DM is listed, so we will simply register the relation of this DM to the Light Path analysed.
                                tmp_dm_light_path_relation.append(k)
                                break
                        # If it is not, append it
                        if dm_already_listed is False:
                            # Entering here implies that the DM was not listed, so we will have to add it
                            self.dm_scanned_list.append(light_path_list[i].dm[j])
                            # Register the relation of the DM to the light path 
                            tmp_dm_light_path_relation.append(len(self.dm_scanned_list)-1)
                        # Reset for next DM
                        dm_already_listed = False
            # Save the DMs affecting the light path into the main matrix
            im_scan_plan.append(tmp_dm_light_path_relation)


        # Create a boolean matrix of size: nLightPaths x nDMs in which True implies that there is a relation between a DM and a WFs in the specified light path
        # and an interaction matrix must be defined

        self.im_boolean_matrix = np.zeros((len(light_path_list), len(self.dm_scanned_list)), dtype=bool)

        for i in range(len(im_scan_plan)):
            if len(im_scan_plan[i]):
                for j in range(len(im_scan_plan[i])):
                        self.im_boolean_matrix[i, im_scan_plan[i][j]] = True

        # Finally, store the list of light paths to have it available for the measurement process.
        self.light_path_list = light_path_list
        
        return True
    
    def generate_modal_basis(self):
        """
        Generate all modal bases (zonal, Zernike, KL, Hadamard, DH) for each DM.

        Returns
        -------
        bool
            True when all bases are generated.
        """
        # Finally, generate the modal basis for each DM to speed up the measuring later
        self.modal_basis = []
        self.logger.info('InteractionMatrixHandler::generate_modal_basis - Generating modal basis')
        
        tasks = []

        for i in range(len(self.dm_scanned_list)):
            tasks.append(delayed(generate_zonal_modes)(self.dm_scanned_list[i]))
            tasks.append(delayed(generate_zernike_modes)(self.dm_scanned_list[i]))
            tasks.append(delayed(generate_kl_modes)(self.dm_scanned_list[i]))
            tasks.append(delayed(generate_hadamard_modes)(self.dm_scanned_list[i]))
            tasks.append(delayed(generate_dh_modes)(self.dm_scanned_list[i]))

        t0 = time.time()
        modes = Parallel(n_jobs=len(tasks), prefer="threads")(tasks)
        self.logger.info(f'InteractionMatrixHandler::generate_modal_basis - Modal basis generated, took {time.time()-t0} [s]')
               
        modal_basis_dict = dict.fromkeys(self.modal_list, None)

        for i in range(len(self.dm_scanned_list)):
            tmp = modes[i*len(self.modal_list):(i+1)*len(self.modal_list)]
            self.modal_basis.append(modal_basis_dict.copy())
            for j in range(len(tmp)):
                self.modal_basis[-1][self.modal_list[j]] = tmp[j]

        return True

    # This method measures the interaction matrix according to the im_boolean_matrix generated during initialization.
    # After measuring the IM, computes the reconstruction matrix as well using pseudo-inversion. 
    # modal_basis_list can be a string or a list of length equal to the number of DMs, enabling the definition of a modal basis for all the DMs or specifying one modal basis for each DM
    # stroke is in [m] and can be a scalar or a list, to let the user set a common stroke for all the DMs or define it per DM.
    # nModes is by default None, which will use all the modes of the DMs. If define, it shall be a list specifying the number of modes per DM
    def measure(self, modal_basis, stroke, nModes=None):
        """
        Measure interaction matrices for each WFS-DM pair defined in the system.

        Parameters
        ----------
        modal_basis : str or list
            Modal basis to use ('zonal', 'zernike', etc.).
        stroke : float or list
            Stroke amplitude in meters.
        nModes : list or None
            Number of modes per DM, or None to use all.

        Returns
        -------
        bool
            True if all IMs are successfully measured.
        """
        # Check modal_basis parameter
        modal_basis_per_DM = []
        if isinstance(modal_basis, list):
            if len(modal_basis) == self.im_boolean_matrix.shape[1]:
                modal_basis_per_DM = modal_basis
            else:
                raise ValueError('InteractionMatrixHandler::measure - If the modal basis are specify per DM, the length shall be equal to the number of DMs. \
                                 Use a string otherwise.')
        else:
            if isinstance(modal_basis, str):
                modal_basis_per_DM = [modal_basis for _ in range(self.im_boolean_matrix.shape[1])]
            else:
                raise TypeError('InteractionMatrixHandler::measure - String or list were expected.')

        # Check stroke parameter
        self.stroke_per_DM = []
        if isinstance(stroke, list):
            if len(stroke) == self.im_boolean_matrix.shape[1]:
                self.stroke_per_DM = stroke
            else:
                raise ValueError('InteractionMatrixHandler::measure - If the stroke is specify per DM, the length shall be equal to the number of DMs. \
                                 Use a scalar otherwise.')
        else:
            if isinstance(stroke, float):
                self.stroke_per_DM = [stroke for _ in range(self.im_boolean_matrix.shape[1])]
            else:
                raise TypeError('InteractionMatrixHandler::measure - Float or list were expected.')

        # Check nModes parameter
        nModes_per_DM = []
        if isinstance(nModes, list):
            if len(nModes) == self.im_boolean_matrix.shape[1]:
                nModes_per_DM = nModes
            else:
                raise ValueError('InteractionMatrixHandler::measure - If the number of modes are specify per DM, the length shall be equal to the number of DMs. \
                                 Use a None otherwise.')
        else:
            if nModes is None:
                nModes_per_DM = [self.dm_scanned_list[i].nValidAct for i in range(self.im_boolean_matrix.shape[1])]
            else:
                raise TypeError('InteractionMatrixHandler::measure - None or list were expected.')
        if modal_basis not in self.modal_list:
            self.logger.error(f'InteractionMatrixHandler::measure Modal basis ({modal_basis}) unrecognised, supported types are: {self.modal_list}')
            raise ValueError('InteractionMatrixHandler::measure - Unsupported modal basis.')
        # Before measuring the IMs, we need to have all the modal basis available
        if self.modal_basis == []:
            self.generate_modal_basis()
        # Once the input parameters are defined, we proceed to measure the IM
        # Prepare the variable to store the different IMs
        im_dict = {'modalBasis':None, 'IM':None, 'slopes_units':'px'}
        # nLps x mDms
        self.interaction_matrix_warehouse = [[im_dict.copy() for i in range(self.im_boolean_matrix.shape[0])] for j in range(self.im_boolean_matrix.shape[1])]
        # Prepare the LightPaths to be parallelized
        tasks = []

        for i in range(len(self.light_path_list)):
            tasks.append(delayed(self.light_path_list[i].propagate)(temporal_tick=False, interaction_matrix=True))

        for i in range(len(self.dm_scanned_list)):
            self.logger.info(f'InteractionMatrixHandler::measure - DM {i}')
            # Get the modal basis
            modes = self.modal_basis[i][modal_basis_per_DM[i]][:, :nModes_per_DM[i]]
            # Check if the DM is at ground layer or altitude to discard TT
            discarded_modes = 0
            if self.dm_scanned_list[i].altitude > 0 and modal_basis_per_DM[i] in ['zernike', 'dh', 'kl']:
                self.logger.warning(f'InteractionMatrixHandler::measure - Be advised that TT is discarded in altitude DMs for basis {modal_basis_per_DM[i]}, the number of modes specified is reduced by 2.')
                modes = modes[:,2:]
                discarded_modes = 2
            # Initialize the IMs that will be measured
            tmp_IM_list = []
                
            for k in range(len(self.light_path_list)):
                if self.im_boolean_matrix[k, i]: # If True, then an IM shall be measured
                    tmp_IM_list.append(im_dict.copy())
                    # Fill the metadata of the matrix
                    tmp_IM_list[-1]['modalBasis'] = modal_basis_per_DM[i]
                    tmp_IM_list[-1]['discarded_modes'] = discarded_modes
                    tmp_IM_list[-1]['IM'] = np.zeros((self.light_path_list[k].wfs.nSignal, modes.shape[1]))
                    tmp_IM_list[-1]['slopes_units'] = 'rad' if self.light_path_list[k].wfs.unit_in_rad else 'px'
                else:
                    tmp_IM_list.append(im_dict.copy())
            # Now, loop over each mode to measure the interaction matrix
            for j in range(modes.shape[1]):
                if (j % 50) == 0:
                    self.logger.info(f'InteractionMatrixHandler::measure - Mode {j} out of {modes.shape[1]}')
                # Apply the modal command to the DM
                cmd = self.stroke_per_DM[i] * modes[:,j]
                self.dm_scanned_list[i].updateDMShape(torch.as_tensor(cmd, dtype=torch.float64, device=self.device).unsqueeze(1), dynamicResponse=False)
                # Propagate
                Parallel(n_jobs=2, prefer="threads")(tasks)
                # Measure the WFS slopes at the Light Path affected
                for k in range(len(self.light_path_list)):
                    if self.im_boolean_matrix[k, i]:
                        tmp_IM_list[k]['IM'][:, j] = self.light_path_list[k].slopes_1D / self.stroke_per_DM[i]

            self.interaction_matrix_warehouse[i] = tmp_IM_list.copy()
            # Make sure that the DM is set to zero before commanding the next one
            cmd = 0 * modes[:,0]
            self.dm_scanned_list[i].updateDMShape(torch.as_tensor(cmd, dtype=torch.float64, device=self.device).unsqueeze(1), dynamicResponse=False)      

        # Save the maximum displacements into a variable to provide feedback to the user
        self.max_displacement = np.zeros((len(self.dm_scanned_list), len(self.light_path_list))) # nDms x nLPs

        for i in range(self.max_displacement.shape[0]):
            for j in range(self.max_displacement.shape[1]):
                if self.interaction_matrix_warehouse[i][j]['IM'] is not None:
                    self.max_displacement[i, j] = np.max(np.abs(self.interaction_matrix_warehouse[i][j]['IM'])) * self.stroke_per_DM[i]

        self.logger.info(f'Max. displacement info: DM x LP: {self.max_displacement}')
        return True
    
    def save_IM(self, filename=None):
        """
        Save the interaction matrix warehouse to a self.im_boolean_matrix[j, i]H5 file.

        Parameters
        ----------
        filename : str
            Path to save the IM file.

        Returns
        -------
        bool
            True if save is successful.
        """
        self.logger.debug('InteractionMatrixHandler::save_IM')

        if self.interaction_matrix_warehouse is None:
            self.logger.error('InteractionMatrixHandler::save_IM - IM warehouse not initialized.')
            return False
        
        self.logger.info('InteractionMatrixHandler::save_IM - Writting...') 

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not filename.endswith(".h5"):
            filename += ".h5"
            
        with h5py.File(filename, 'a') as f:
            f.attrs['nLPS'] = len(self.light_path_list)
            f.attrs['nDMs'] = len(self.dm_scanned_list)

            for i in range(len(self.light_path_list)):
                if np.sum(self.im_boolean_matrix[i,:]) > 0: # There is an IM defined for this LP, so there is WFS
                    # Append LP
                    lp_group = f.create_group('LP' + str(i))
                    # Set main attributes of this LP
                    lp_group.attrs['wavelength'] = self.light_path_list[i].src.wavelength
                    lp_group.attrs['zenith']     = np.round(self.light_path_list[i].src.coordinates[0], 2)
                    lp_group.attrs['azimuth']    = np.round(self.light_path_list[i].src.coordinates[1], 2)
                    lp_group.attrs['nSignal']    = self.light_path_list[i].wfs.nSignal
                    # Create the IMs
                    for j in range(self.im_boolean_matrix.shape[1]):
                        if self.im_boolean_matrix[i,j]: # There is an interaction with DM j
                            im_subgroup = lp_group.create_group('IM' + str(j))
                            # Append DMs attributes
                            im_subgroup.attrs['nValidAct']         = self.dm_scanned_list[j].nValidAct
                            im_subgroup.attrs['nAct']              = self.dm_scanned_list[j].nActs
                            im_subgroup.attrs['altitude']          = self.dm_scanned_list[j].altitude
                            im_subgroup.attrs['mechCoupling']      = self.dm_scanned_list[j].mechCoupling
                            im_subgroup.attrs['modalBasis']        = self.interaction_matrix_warehouse[j][i]['modalBasis']
                            im_subgroup.attrs['maxDisplacement']   = self.max_displacement[j, i]
                            im_subgroup.attrs['discarded_modes']   = self.interaction_matrix_warehouse[j][i].get('discarded_modes', 0)
                            # Append IM
                            im_subgroup.create_dataset('data', data=self.interaction_matrix_warehouse[j][i]['IM'])
                        
        self.logger.info('InteractionMatrixHandler::save_IM - Saved.')
    
    def save_modalBasis(self, filename):
        """
        Save the generated modal bases to a H5 file.

        Parameters
        ----------
        filename : str
            Output filename (without extension).

        Returns
        -------
        bool
            True if saved correctly.
        """
        self.logger.debug('InteractionMatrixHandler::save_modalBasis')

        if self.modal_basis is None:
            self.logger.error('InteractionMatrixHandler::save_modalBasis - Modal Basis not initialized.')
            return False

        self.logger.info('InteractionMatrixHandler::save_IM - Writting...')

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not filename.endswith(".h5"):
            filename += ".h5"
            
        with h5py.File(filename, 'a') as f:
            for i in range(len(self.modal_list)):
                # Create one group per modal basis
                modal_group = f.create_group(self.modal_list[i])
                # For each modal basis, create a subgroup per DM --> The name is the number of valid acts
                for j in range(len(self.dm_scanned_list)):
                    dm_subgroup = modal_group.create_group('DM' + str(j))
                    # Then, we add the modal base and the metadata
                    dm_subgroup.create_dataset('data', data=self.modal_basis[j][self.modal_list[i]])
                    dm_subgroup.attrs['nAct'] = self.dm_scanned_list[j].nActs
                    dm_subgroup.attrs['nValidAct'] = self.dm_scanned_list[j].nValidAct
       
        self.logger.info('InteractionMatrixHandler::save_IM - Saved.')        

    def load_IM(self, filename):
        """
        Load a previously saved interaction matrix warehouse.

        Parameters
        ----------
        filename : str
            File path with/without .h5 extension.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        # To load the IM warehouse, we need to know the LightPath and DMs properties first to check if the warehouse is valid for the current setup.
        if self.im_boolean_matrix is None:
            self.logger.error('InteractionMatrixHandler::load_IM - The class has not been initialize, the im_boolean_matrix is None.')
            raise ValueError('The class has not been initialize, the im_boolean_matrix is None.')
        
        if not filename.endswith(".h5"):
            filename += ".h5"

        # Initialize the warehouse
        im_dict = {'modalBasis':None, 'IM':None, 'slopes_units':'px'}
        self.interaction_matrix_warehouse = [[im_dict.copy() for i in range(self.im_boolean_matrix.shape[0])] for j in range(self.im_boolean_matrix.shape[1])]
        self.max_displacement = np.zeros_like(self.im_boolean_matrix, dtype=np.float32).T # nDms x nLPs

        with h5py.File(filename, 'r') as f:
            # Loop over the light paths
            for i in range(len(self.light_path_list)):
                if np.sum(self.im_boolean_matrix[i,:]) > 0: # This LP has a WFS
                    # Find the match in the file
                    lp_match_key = None
                    for j in range(len(f.keys())):
                        if (f['LP' + str(j)].attrs['wavelength'] == self.light_path_list[i].src.wavelength) and \
                           (f['LP' + str(j)].attrs['zenith']       == np.round(self.light_path_list[i].src.coordinates[0], 2)) and \
                           (f['LP' + str(j)].attrs['azimuth']      == np.round(self.light_path_list[i].src.coordinates[1], 2)) and \
                           (f['LP' + str(j)].attrs['nSignal']      == self.light_path_list[i].wfs.nSignal):
                           lp_match_key = 'LP' + str(j)
                           break
                    if lp_match_key is None:
                        self.logger.error('InteractionMatrixHandler::load_IM - There is no match for the curent LP in the IM file.')
                        raise ValueError('There is no match for the curent LP in the IM file.')
                    # Then, access the IMs and find the DM match
                    for j in range(len(f[lp_match_key].keys())):
                        dm_match_idx = None
                        for k in range(len(self.dm_scanned_list)):
                            if self.im_boolean_matrix[i,k]: # DM with an interaction with the current LP
                                # Check the parameters
                                if (f[lp_match_key]['IM' + str(j)].attrs['nValidAct'] == self.dm_scanned_list[k].nValidAct) and \
                                   (f[lp_match_key]['IM' + str(j)].attrs['altitude']  == self.dm_scanned_list[k].altitude) and \
                                   (f[lp_match_key]['IM' + str(j)].attrs['nAct'] == self.dm_scanned_list[k].nActs):
                                    # We have a match with the current DM
                                    dm_match_idx = k
                                    # Store the IM
                                    self.interaction_matrix_warehouse[dm_match_idx][i]['modalBasis'] = f[lp_match_key]['IM' + str(j)].attrs['modalBasis']
                                    self.interaction_matrix_warehouse[dm_match_idx][i]['discarded_modes'] = f[lp_match_key]['IM' + str(j)].attrs.get('discarded_modes', 0)
                                    self.interaction_matrix_warehouse[dm_match_idx][i]['IM'] = np.array(f[lp_match_key]['IM' + str(j)]['data'])

                                    self.max_displacement[j, i] = f[lp_match_key]['IM' + str(j)].attrs['maxDisplacement']

                                    # Break to continue with the next IM                                    
                                    break                        
        
        self.logger.info(f'Max. displacement info: DM x LP: {self.max_displacement}')    
        self.logger.info('InteractionMatrixHandler::load_IM - Ended succesfully.')

        return True

    def load_modalBasis(self, filename):
        """
        Load a previously saved modal basis FITS file.

        Parameters
        ----------
        filename : str
            File path without .h5 extension.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        # We need to check if the content of the file is compatible with current simulation setup
        if self.dm_scanned_list is None:
            self.logger.error('InteractionMatrixHandler::load_modalBasis - The class has not been initialize, the dm_scanned_list is None.')
            raise ValueError('The class has not been initialize, the dm_scanned_list is None.')
        
        if not filename.endswith('.h5'):
            filename += '.h5' # add extension
        
        with h5py.File(filename, 'r') as f:
            # First, check if the modal basis is compatible with the DMs
            # Check modal basis
            if len(f.keys()) == len(self.modal_list):
                for i in self.modal_list:
                    if i not in f.keys():
                        self.logger.error(f'InteractionMatrixHandler::load_modalBasis - Modal base {i} is not in the file.')
                        raise ValueError('InteractionMatrixHandler::load_modalBasis - Modal base is not in the file.')
            else:
                self.logger.error(f'InteractionMatrixHandler::load_modalBasis - The number of modal basis {len(self.modal_list)} \
                                  is not consist with the file content {len(f.keys())}.')
                raise ValueError('InteractionMatrixHandler::load_modalBasis - Modal base is not in the file.')    
            # Create the modal basis list
            self.modal_basis = []
            # Check DMs and load!
            if len(f[self.modal_list[0]]) == len(self.dm_scanned_list):
                for i in range(len(self.dm_scanned_list)):
                    dm_name = 'DM' + str(i)

                    if dm_name in f[self.modal_list[0]].keys():
                        if f[self.modal_list[0]][dm_name].attrs['nValidAct'] != self.dm_scanned_list[i].nValidAct:
                            self.logger.error(f'InteractionMatrixHandler::load_modalBasis - The number of validActs is not consistent.')
                            raise ValueError('InteractionMatrixHandler::load_modalBasis - The number of validActs is not consistent.')

                        if f[self.modal_list[0]][dm_name].attrs['nAct'] != self.dm_scanned_list[i].nActs:
                            self.logger.error(f'InteractionMatrixHandler::load_modalBasis - The number of acts is not consistent.')
                            raise ValueError('InteractionMatrixHandler::load_modalBasis - The number of acts is not consistent.')  
                        # The data is consistent, load it into the class
                        tmp_dict = dict.fromkeys(self.modal_list)

                        for j in range(len(self.modal_list)):
                            tmp_dict[self.modal_list[j]] = np.array(f[self.modal_list[j]][dm_name]['data'])
                        
                        self.modal_basis.append(tmp_dict)

                    else:
                        self.logger.error(f'InteractionMatrixHandler::load_modalBasis - Expected DM not found in the file.')
                        raise ValueError('InteractionMatrixHandler::load_modalBasis - Expected DM not found in the file.')                           
                
        self.logger.info('InteractionMatrixHandler::load_modalBasis - Ended succesfully.')

        return True 

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
    
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()