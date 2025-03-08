import time

import numpy as np
import torch
from joblib import Parallel, delayed

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
        dm_scanned_list = [] # Stores a link to the DMs objects that are different, so as to recover their properties and command them

        for i in range(len(light_path_list)):
            tmp_dm_light_path_relation = []
            if hasattr(light_path_list[i], 'dm') and hasattr(light_path_list[i], 'wfs'):
                if len(dm_scanned_list) == 0:
                    for j in range(len(light_path_list[i].dm)):
                        # There are no DMs defined, so add a new one to our list!
                        dm_scanned_list.append(light_path_list[i].dm[j])
                        # Register the relation of this DM to the light path
                        tmp_dm_light_path_relation.append(len(dm_scanned_list)-1)                        
                else:
                    # We already have DMs defined, so before adding a new one to our list, we need to check whether it is listed or not.
                    dm_already_listed = False

                    for j in range(len(light_path_list[i].dm)):
                        for k in range(len(dm_scanned_list)):
                            # Compare the properties of this DM with the ones already listed
                            if ((dm_scanned_list[k].nAct         == light_path_list[i].dm[j].nAct)      and
                                (dm_scanned_list[k].altitude     == light_path_list[i].dm[j].altitude)  and
                                (dm_scanned_list[k].nValidAct    == light_path_list[i].dm[j].nValidAct) and
                                (dm_scanned_list[k].mechCoupling == light_path_list[i].dm[j].mechCoupling)):
                                
                                dm_already_listed = True
                                # The DM is listed, so we will simply register the relation of this DM to the Light Path analysed.
                                tmp_dm_light_path_relation.append(k)
                                break
                        if dm_already_listed is False:
                            # Entering here implies that the DM was not listed, so we will have to add it
                            dm_scanned_list.append(light_path_list[i].dm[j])
                            # Register the relation of the DM to the light path 
                            tmp_dm_light_path_relation.append(len(dm_scanned_list)-1)
            # Save the DMs affecting the light path into the main matrix
            im_scan_plan.append(tmp_dm_light_path_relation)


        # Create a boolean matrix of size: nLightPaths x nDMs in which True implies that there is a relation and an interaction matrix must be defined

        self.im_boolean_matrix = np.zeros((len(light_path_list), len(dm_scanned_list)), dtype=bool)

        for i in range(len(im_scan_plan)):
            if len(im_scan_plan[i]) > 0:
                for j in range(len(im_scan_plan[i])):
                    self.im_boolean_matrix[i, im_scan_plan[i][j]] = True

        # Finally, store the list of light paths to have it available for the measurement process.
        self.light_path_list = light_path_list
        
        return True

    # Loads from file 
    def load_modal_basis(self, path):
        return True
    
    def load_interaction_matrix(self, path):
        return True

    def load_reconstruction_matrix(self,path):
        return True

   
    def save_modal_basis(self, path):
        return True
    
    def save_interaction_matrix(self, path):
        return True
    
    def save_reconstruction_matrix(self, path):
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