import time

import numpy as np
import torch
from joblib import Parallel, delayed

from astropy.io import fits

from OOPAO.modalBasis.zonalModes import generate_zonal_modes
from OOPAO.modalBasis.zernikeModes import generate_zernike_modes
from OOPAO.modalBasis.karhunenLoeveModes import generate_kl_modes
from OOPAO.modalBasis.discHarmonicModes import generate_dh_modes
from OOPAO.modalBasis.hadamardModes import generate_hadamard_modes

import logging
import logging.handlers
from queue import Queue

class InteractionMatrixHandler:
    def __init__(self, logger):
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger

    # Checks the Light Paths defined to knwo the interaction matrices that are necessary and prepares the measurement procedure

    def initialize_im_class(self, light_path_list):
        self.logger.debug('interactionMatrixHandler::initialize_im_class')

        # Check if the light path is a list:

        if not isinstance(light_path_list, list):
            light_path_list = [light_path_list]
        
        # For efficiency, we need to detect the different DMs of the optical configuration. This will enable this class
        # to move each DM once and measure all the WFS affected, avoiding repeating any movement and saving time.
        im_scan_plan = [] # Stores in each row a list of the DM indeces affecting the light Path. Each row is a different LP, if empty, an IM is not needed.
        self.dm_scanned_list = [] # Stores a link to the DMs objects that are different, so as to recover their properties and command them

        for i in range(len(light_path_list)):
            tmp_dm_light_path_relation = []
            if hasattr(light_path_list[i], 'dm') and hasattr(light_path_list[i], 'wfs'):
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
                        for k in range(len(self.dm_scanned_list)):
                            # Compare the properties of this DM with the ones already listed
                            if ((self.dm_scanned_list[k].nAct         == light_path_list[i].dm[j].nAct)      and
                                (self.dm_scanned_list[k].altitude     == light_path_list[i].dm[j].altitude)  and
                                (self.dm_scanned_list[k].nValidAct    == light_path_list[i].dm[j].nValidAct) and
                                (self.dm_scanned_list[k].mechCoupling == light_path_list[i].dm[j].mechCoupling)):
                                
                                dm_already_listed = True
                                # The DM is listed, so we will simply register the relation of this DM to the Light Path analysed.
                                tmp_dm_light_path_relation.append(k)
                                break
                        if dm_already_listed is False:
                            # Entering here implies that the DM was not listed, so we will have to add it
                            self.dm_scanned_list.append(light_path_list[i].dm[j])
                            # Register the relation of the DM to the light path 
                            tmp_dm_light_path_relation.append(len(self.dm_scanned_list)-1)
            # Save the DMs affecting the light path into the main matrix
            im_scan_plan.append(tmp_dm_light_path_relation)


        # Create a boolean matrix of size: nLightPaths x nDMs in which True implies that there is a relation and an interaction matrix must be defined

        self.im_boolean_matrix = np.zeros((len(light_path_list), len(self.dm_scanned_list)), dtype=bool)

        for i in range(len(im_scan_plan)):
            if len(im_scan_plan[i]) > 0:
                for j in range(len(im_scan_plan[i])):
                    self.im_boolean_matrix[i, im_scan_plan[i][j]] = True

        # Finally, store the list of light paths to have it available for the measurement process.
        self.light_path_list = light_path_list
        
        return True
    # This method measures the interaction matrix according to the im_boolean_matrix generated during initialization.
    # After measuring the IM, computes the reconstruction matrix as well using psuedo-inversion. 
    # modal_basis_list can be a string or a list of length equal to the number of DMs, enabling the definition of a modal basis for all the DMs or specifying one modal basis for each DM
    # stroke is in [m] and can be a scalar or a list, to let the user set a common stroke for all the DMs or define it per DM.
    # nModes is by default None, which will use all the modes of the DMs. If define, it shall be a list specifying the number of modes per DM
    # rcond specifies the percentage over the maximum singular value that are filtered during pseudo-inversion.
    def measure(self, modal_basis, stroke, nModes=None, rcond=0.025):
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
        stroke_per_DM = []
        if isinstance(stroke, list):
            if len(stroke) == self.im_boolean_matrix.shape[1]:
                stroke_per_DM = stroke
            else:
                raise ValueError('InteractionMatrixHandler::measure - If the stroke is specify per DM, the length shall be equal to the number of DMs. \
                                 Use a scalar otherwise.')
        else:
            if isinstance(stroke, float):
                stroke_per_DM = [stroke for _ in range(self.im_boolean_matrix.shape[1])]
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
        
        # Once the input parameters are defined, we proceed to measure the IM
        # Prepare the variable to store the different IMs
        im_dict = {'modalBasis':None, 'IM':None, 'slopes_units':'px'}
        self.interaction_matrix_warehouse = [[im_dict for i in range(self.im_boolean_matrix.shape[0])] for j in range(self.im_boolean_matrix.shape[1])]
        # Prepare the LightPaths to be parallelized
        tasks = []

        for i in range(len(self.light_path_list)):
            tasks.append(delayed(self.light_path_list[i].propagate)(True, False, True))

        for i in range(len(self.dm_scanned_list)):
            self.logger.info(f'InteractionMatrixHandler::measure - DM {i}')
            # Get the modal basis
            if modal_basis_per_DM[i] == 'zonal' or modal_basis_per_DM[i] == 'Zonal':
                modes = generate_zonal_modes(self.dm_scanned_list[i], nModes=nModes_per_DM[i])
            elif modal_basis_per_DM[i] == 'hadamard' or modal_basis_per_DM[i] == 'Hadamard':
                modes = generate_hadamard_modes(dm=self.dm_scanned_list[i], nModes=nModes_per_DM[i])
            elif modal_basis_per_DM[i] == 'kl' or modal_basis_per_DM[i] == 'KL':
                modes = generate_kl_modes(dm=self.dm_scanned_list[i], nModes=nModes_per_DM[i])
            elif modal_basis_per_DM[i] == 'zernike' or modal_basis_per_DM[i] == 'Zernike':
                modes = generate_zernike_modes(dm=self.dm_scanned_list[i], nModes=nModes_per_DM[i])
            elif modal_basis_per_DM[i] == 'dh' or modal_basis_per_DM[i] == 'DH':
                modes = generate_dh_modes(dm=self.dm_scanned_list[i], nModes=nModes_per_DM[i])
            else:
                raise ValueError(f'InteractionMatrixHandler::measure - Unrecognize modal basis f{modal_basis_per_DM[i]}.')
            # Check if the DM is at ground layer or altitude to discard TT
            if self.dm_scanned_list[i].altitude > 0:
                self.logger.warning('InteractionMatrixHandler::measure - Be advised that TT is discarded in altitude DMs, the number of modes specified is reduced by 2.')
                modes = modes[:,:,2:]
            # Initialize the IMs that will be measured
            tmp_IM_list = []
                
            for k in range(len(self.light_path_list)):
                if self.im_boolean_matrix[k, i]: # If True, then an IM shall be measured
                    tmp_IM_list.append(im_dict.copy())
                    # Fill the metadata of the matrix
                    tmp_IM_list[-1]['modalBasis'] = modal_basis_per_DM[i]
                    tmp_IM_list[-1]['IM'] = np.zeros((self.light_path_list[k].wfs.nSignal, modes.shape[2]))
                    tmp_IM_list[-1]['slopes_units'] = 'rad' if self.light_path_list[k].wfs.unit_in_rad else 'px'
                else:
                    tmp_IM_list.append(None)
            # Now, loop over each mode to measure the interaction matrix
            for j in range(modes.shape[2]):
                if (j % 50) == 0:
                    self.logger.info(f'InteractionMatrixHandler::measure - Mode {j} out of {modes.shape[2]}')
                # Apply the modal command to the DM
                cmd = stroke_per_DM[i] * modes[:,:, j]
                self.dm_scanned_list[i].updateDMShape(cmd)
                # Propagate
                Parallel(n_jobs=len(self.light_path_list), prefer="threads")(tasks)
                # Measure the WFS slopes at the Light Path affected
                for k in range(len(self.light_path_list)):
                    if self.im_boolean_matrix[k, i]:
                        tmp_IM_list[k]['IM'][:, j] = self.light_path_list[k].slopes_1D / stroke_per_DM[i]

            self.interaction_matrix_warehouse[i] = tmp_IM_list.copy()
            # Make sure that the DM is set to zero before commanding the next one
            cmd = 0 * modes[:,:, 0]
            self.dm_scanned_list[i].updateDMShape(cmd)            
        return True
    
    def save(self, filename=None):
        self.logger.debug('InteractionMatrixHandler::save')

        if self.interaction_matrix_warehouse is None:
            self.logger.error('InteractionMatrixHandler::save - The atmosphere has not been initialized yet.')
            return False

        self.logger.info('InteractionMatrixHandler::save - Creating the HDU')
        
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['nDMs']    = len(self.dm_scanned_list)
        primary_hdu.header['nLPs']    = len(self.light_path_list)

        ps_hdu_list = []
        ps_hdu_list.append(primary_hdu)

        # Create the header entries for each DM and Light Path

        for i in range(len(self.dm_scanned_list)):
            ps_hdu_list[0].header['DM' + str(i) + 'vAct'] = self.dm_scanned_list[i].nValidAct
            ps_hdu_list[0].header['DM' + str(i) + 'alt']  = self.dm_scanned_list[i].altitude
            ps_hdu_list[0].header['DM' + str(i) + 'mech'] = self.dm_scanned_list[i].mechCoupling

        for i in range(len(self.light_path_list)):
            if np.sum(self.im_boolean_matrix[i,:]) > 0: # There is an IM defined for this LP, so there is WFS
                ps_hdu_list[0].header['LP' + str(i) + 'wl']    = self.light_path_list[i].src.wavelength
                ps_hdu_list[0].header['LP' + str(i) + 'radi']    = np.round(self.light_path_list[i].src.coordinates[0], 2)
                ps_hdu_list[0].header['LP' + str(i) + 'azim']    = np.round(self.light_path_list[i].src.coordinates[1], 2)
                ps_hdu_list[0].header['LP' + str(i) + 'sign']    = self.light_path_list[i].wfs.nSignal
            else:
                ps_hdu_list[0].header['LP' + str(i) + 'wl']      = 0

        for i in range(len(self.dm_scanned_list)):
            for j in range(len(self.light_path_list)):
                if self.im_boolean_matrix[j, i]:
                    name = str(i) + '-' + str(j) + '-' + self.interaction_matrix_warehouse[i][j]['modalBasis'][:2] + \
                            '-' + self.interaction_matrix_warehouse[i][j]['slopes_units'][:2]
                    ps_hdu_list.append(fits.ImageHDU(self.interaction_matrix_warehouse[i][j]['IM'], name=name))

       
        self.logger.info('InteractionMatrixHandler::save - Writting...')
        hdul = fits.HDUList(ps_hdu_list)
        hdul.writeto(filename + '.fits', overwrite=True)
        self.logger.info('InteractionMatrixHandler::save - Saved.')

    def load(self, filename):
        # To load the IM warehouse, we need to know the LightPath and DMs properties first to check if the warehouse is valid for the current setup.
        if self.im_boolean_matrix is None:
            self.logger.error('InteractionMatrixHandler::load - The class has not been initialize, the im_boolean_matrix is None.')
            raise ValueError('The class has not been initialize, the im_boolean_matrix is None.')
        # Read the file
        with fits.open(filename + '.fits') as hdul:
            
            # Check the parameters:
            if hdul[0].header['nDMs'] == len(self.dm_scanned_list):
                nDMs = hdul[0].header['nDMs']
            else:
                raise ValueError('InteractionMatrixHandler::load - Number of DMs of the file does not match the current setup.')
            
            if hdul[0].header['nLPs'] == len(self.light_path_list):
                nLPs = hdul[0].header['nLPs']
            else:
                raise ValueError('InteractionMatrixHandler::load - Number of LightPaths of the file does not match the current setup.')
            
            # Check the DMs parameters:

            for i in range(nDMs):
                if ((hdul[0].header['DM' + str(i) + 'vAct'] != self.dm_scanned_list[i].nValidAct) or
                    (hdul[0].header['DM' + str(i) + 'alt']  != self.dm_scanned_list[i].altitude)  or
                    (hdul[0].header['DM' + str(i) + 'mech'] != self.dm_scanned_list[i].mechCoupling)):
                    raise ValueError('InteractionMatrixHandler::load - stored DM parameters do not match current simulation.')

            # Check the LPs parameters

            for i in range(nLPs):
                if hdul[0].header['LP' + str(i) + 'wl'] == 0: # This implies that there is no IM defined for this LP, let's check the simulation config
                    if np.sum(self.im_boolean_matrix[i,:]) > 0:
                        raise ValueError('InteractionMatrixHandler::load - This LP should have IM defined, but the file does not contain any.')

                if ((hdul[0].header['LP' + str(i) + 'wl']    != self.light_path_list[i].src.wavelength)                   or
                    (hdul[0].header['LP' + str(i) + 'radi']  != np.round(self.light_path_list[i].src.coordinates[0], 2))  or
                    (hdul[0].header['LP' + str(i) + 'azim']  != np.round(self.light_path_list[i].src.coordinates[1], 2))   or
                    (hdul[0].header['LP' + str(i) + 'sign']  != self.light_path_list[i].wfs.nSignal)):
                    raise ValueError('InteractionMatrixHandler::load - stored LPs parameters do not match current simulation.')                
                
            # Verification passed, so we can use the class properties from now on
            im_dict = {'modalBasis':None, 'IM':None, 'slopes_units':'px'}
            self.interaction_matrix_warehouse = [[im_dict for i in range(self.im_boolean_matrix.shape[0])] for j in range(self.im_boolean_matrix.shape[1])]
            
            modal_map = {'zo': 'zonal', 'ZO': 'zonal', 'ze': 'zernike', 'ZE': 'zernike',
                     'kl': 'KL', 'KL': 'KL', 'dh': 'DH', 'DH': 'DH', 'ha': 'hadamard', 'HA': 'hadamard'}

            for hdu in hdul[1:]:
                name_parts = hdu.name.split('-')
                i, j, modalBasis, slopes_units = int(name_parts[0]), int(name_parts[1]), name_parts[2], name_parts[3]
                
                if isinstance(hdu.data, np.ndarray) and self.im_boolean_matrix[j, i]:

                    self.interaction_matrix_warehouse[i][j] = {
                        'IM': hdu.data,
                        'modalBasis': modal_map[modalBasis],
                        'slopes_units': slopes_units
                    }
                elif (not isinstance(hdu.data, np.ndarray) and self.im_boolean_matrix[j, i] or
                      isinstance(hdu.data, np.ndarray) and not self.im_boolean_matrix[j, i]):
                    raise ValueError(f'InteractionMatrixHandler::load - there is not an agreement between the im_boolean_matrix and the content of the IMs for card {hdu.name}.')
            
            self.logger.info('InteractionMatrixHandler::load - Ended succesfully.')

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