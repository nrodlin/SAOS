import numpy as np
import torch

import time

import logging
import logging.handlers
from queue import Queue

from SAOS.tomography.predictiveLearnApply import predictiveLearnApply
class Controller:
    def __init__(self,
                 telescope,
                 interactionMatrix,
                 controllerType,
                 reconstructionMethod,  
                 operationType='closed',               
                 logger = None,
                 **kwargs):
        """
        Initialize the Controller module.

        Parameters
        ----------
        telescope : Telescope instance
            Telescope instance, provides the sampling time of the simulation.
        interactionMatrix : InteractionMatrixHandler instance
            Contains the interaction matrices and modal basis for the simulation configuration.
        controllerType : String
            The type of controller that will be used, supported types are: {leaky, forwardPI, backwardPI, stateSpace}. 
        reconstructionMethod : String
            Type of reconstructor used, supported types are: {inversion, tikhonov, tomopLA}.        
        operationType : String
            The type of operation that will be used, supported types are: {open, closed, polc}. By default, closed.            
        **kwargs
            rcond : list of length equal to nDMs or float
                Percentage of the maximum singular value below witch the SV are discarded.
            beta : list of length equal to nDMs or float
                Regularisation coefficient beta for the Tikhonov Regularisation: alfa = beta * (Smax**2)
            gain : list of length equal to nDMs or float
                Proportional gain of the Leaky and PI controllers
            decay : list of length equal to nDMs or float
                Decay rate for the Leaky integrator
            ki : list of length equal to nDMs or float
                Integral gain for the PI controllers            
            A : list of 2D arrays or single 2D array
                State-transition matrix/matrices for the stateSpace controller. Shape (N_states, N_states).
            B : list of 2D arrays or single 2D array
                Input-to-state matrix/matrices for the stateSpace controller. Shape (N_states, N_inputs).
            C : list of 2D arrays or single 2D array
                State-to-output matrix/matrices for the stateSpace controller. Shape (N_outputs, N_states).
            D : list of 2D arrays or single 2D array, optional
                Feedforward matrix/matrices for the stateSpace controller. Shape (N_outputs, N_inputs).
        """        
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Define class attributes
        self.tag = 'controller'
        self.delay_input = kwargs.get('delay', 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.samplingTime = telescope.samplingTime

        if reconstructionMethod in {'inversion', 'tikhonov', 'tomopLA'}:
            self.reconstructionMethod = reconstructionMethod
        else:
            self.logger.error('Controller - Unknown reconstructor.')
            raise ValueError('Unknown controller')

        # Default will change to list of size nDMs once the IM is scanned
        self.rcond = kwargs.get('rcond', 0.025)
        self.beta = kwargs.get('beta', 1e-4) # adim, adjusted through trial-error
        
        # Mask provided by the user to select specific WFS-DM links
        self.control_mask = kwargs.get('control_mask', None)

        # Predictive Learn&Apply params
        if self.reconstructionMethod == 'tomopLA':
            self.window = kwargs.get('window', 1000)
            self.updateCycles = kwargs.get('updateCycles', 2000)
            self.lightPaths_init = kwargs.get('lightPaths', None)

        self.nModes = kwargs.get('nModes', None)

        # Run the initialization of the reconstructor
        self.reconstructor, self.modal_basis, self.mask, self.discarded_modes = self.initializeReconstructor(self.reconstructionMethod, interactionMatrix)                

        # Setup the controller

        if controllerType in {'leaky', 'forwardPI', 'backwardPI', 'stateSpace'}:
            self.controllerType = controllerType
        else:
            self.logger.error('Controller - Unknown controller.')
            raise ValueError('Unknown controller')
        
        if operationType in {'open', 'closed', 'polc'}:
            self.operationType = operationType
        else:
            raise ValueError('Unknown operation type')
        
        if self.controllerType == 'stateSpace':
            self.A = kwargs.get('A', None)
            self.B = kwargs.get('B', None)
            self.C = kwargs.get('C', None)
            self.D = kwargs.get('D', None)
            
            if self.A is None or self.B is None or self.C is None:
                raise ValueError("State-space controller requires matrices A, B, and C to be specified.")
            
            if not isinstance(self.A, list):
                self.A = [self.A]
            if not isinstance(self.B, list):
                self.B = [self.B]
            if not isinstance(self.C, list):
                self.C = [self.C]
            if self.D is not None and not isinstance(self.D, list):
                self.D = [self.D]
                
            if len(self.A) != len(self.reconstructor) or len(self.B) != len(self.reconstructor) or len(self.C) != len(self.reconstructor):
                raise ValueError("State-space matrices A, B, C lists must have length equal to the number of DMs.")
            if self.D is not None and len(self.D) != len(self.reconstructor):
                raise ValueError("State-space matrix D list must have length equal to the number of DMs.")
        else:
            self.gain = kwargs.get('gain', [0.0 for _ in range(len(self.reconstructor))])
            self.decay = kwargs.get('decay', [0.0 for _ in range(len(self.reconstructor))])
            self.ki = kwargs.get('ki', [0.0 for _ in range(len(self.reconstructor))])

            if not isinstance(self.gain, list):
                temp_gain = self.gain
                self.gain = [temp_gain for _ in range(len(self.reconstructor))]
            if not isinstance(self.decay, list):
                temp_decay = self.decay
                self.decay = [temp_decay for _ in range(len(self.reconstructor))]
            if not isinstance(self.ki, list):
                temp_ki = self.ki
                self.ki = [temp_ki for _ in range(len(self.reconstructor))]                        

            if len(self.gain) != len(self.reconstructor):
                raise ValueError('The gain should be a float or a a list of size nDMs.')
            if len(self.decay) != len(self.reconstructor):
                raise ValueError('The decay should be a float or a a list of size nDMs.')
            if len(self.ki) != len(self.reconstructor):
                raise ValueError('The ki should be a float or a a list of size nDMs.')                

        # Run the initialization of the controller
        self.initializeController(self.controllerType, self.reconstructor)
   
    def initializeReconstructor(self, reconstructionMethod, interactionMatrix):
        """
        Initialize the reconstructor matrix from the measured interaction matrices.

        Parameters
        ----------
        reconstructionMethod : str
            Type of reconstructor ('inversion' or 'tikhonov').
        interactionMatrix : InteractionMatrixHandler
            Object containing the measured interaction matrices and modal basis.

        Returns
        -------
        reconstructor : list
            List of reconstructor matrices per DM.
        modal_basis : list
            List of modal basis per DM.
        mask : np.ndarray
            Boolean mask indicating interactions between DMs and light paths.
        discarded_modes : list
            List of number of discarded modes per DM.
        """
        self.logger.info('Controller::initializeReconstructor - Computing the reconstructor.')
        t0 = time.time()

        # Define the mask that relates the DMs with the LPs
        
        nDMs = len(interactionMatrix.interaction_matrix_warehouse) # IM warehouse has: nDms x nLPs

        if nDMs < 1:
            raise ValueError('Number of DMs detected are less than 1.')
        
        nLPs = len(interactionMatrix.interaction_matrix_warehouse[0])

        if nLPs < 1:
            raise ValueError('Number of LPs detected are less than 1.')
        
        # Setup delay array
        if isinstance(self.delay_input, list):
            if len(self.delay_input) != nDMs:
                raise ValueError('Delay parameter must be an integer or a list of size nDMs.')
            self.delay = self.delay_input
        else:
            self.delay = [self.delay_input for _ in range(nDMs)]
        
        mask = np.zeros((nDMs, nLPs),dtype=bool)

        # Scan for interactions: if None, then there is not interaction.

        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j]['IM'] is not None:
                    mask[i, j] = True

        if hasattr(self, 'control_mask') and self.control_mask is not None:
            # Check dimensions
            control_mask_arr = np.array(self.control_mask, dtype=bool)
            if control_mask_arr.shape != (nDMs, nLPs):
                self.logger.error(f'Controller - control_mask shape must be ({nDMs}, {nLPs})')
                raise ValueError(f'control_mask shape mismatch. Expected ({nDMs}, {nLPs}), got {control_mask_arr.shape}')
            
            # Warn if user requests control where no IM exists
            invalid_requests = control_mask_arr & (~mask)
            if np.any(invalid_requests):
                self.logger.warning('Controller - control_mask requests control for DM/LP pairs without an interaction matrix. These will be ignored.')
                
            mask = mask & control_mask_arr

        # Check the reconstructor parameters
        if reconstructionMethod == 'inversion':
            if isinstance(self.rcond, list):
                if len(self.rcond) != nDMs:
                    raise ValueError('Rcond parameter is expected to be a list of size equal to the number of DMs.')
            else:
                # Make the list copying the values
                temp_rcond = self.rcond
                self.rcond = [temp_rcond for _ in range(nDMs)]
        elif reconstructionMethod == 'tikhonov':
            if isinstance(self.beta, list):
                if len(self.rcond) != nDMs:
                    raise ValueError('Beta parameter is expected to be a list of size equal to the number of DMs.')
            else:
                # Make the list copying the values
                temp_beta = self.beta
                self.beta = [temp_beta for _ in range(nDMs)]   
        elif reconstructionMethod == 'tomopLA':
            self.tomoReconstructor = predictiveLearnApply(self.window, self.updateCycles, lightPaths=self.lightPaths_init, logger=self.logger)
              
        # Get modal basis
        modal_basis = []
        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j]['IM'] is not None:
                    # The modal basis is common for each DM
                    modal_basis_type = interactionMatrix.interaction_matrix_warehouse[i][j]['modalBasis']
                    modal_basis.append(torch.as_tensor(interactionMatrix.modal_basis[i][modal_basis_type], dtype=torch.float64, device=self.device))
                    break
        # Get discarded modes metadata:
        discarded_modes = []
        for i in range(nDMs):
            found_discarded_modes = 0
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j]['IM'] is not None:
                    found_discarded_modes = interactionMatrix.interaction_matrix_warehouse[i][j].get('discarded_modes', 0)
                    break
            discarded_modes.append(found_discarded_modes)

        # Now, define the reconstruction matrices for each DM

        reconstructor = []
        self.im_per_dm = []
                
        for i in range(nDMs):
            interaction_matrix_per_DM = []
            for j in range(nLPs):
                if mask[i,j]:
                    # Append the IMs to shape one large matrix of size nValidAct x nSignals
                    im = interactionMatrix.interaction_matrix_warehouse[i][j]['IM']
                    if self.nModes is not None:
                        n_m = self.nModes[i] if isinstance(self.nModes, list) else self.nModes
                        im = im[:, :n_m]
                    interaction_matrix_per_DM.append(im)
            # Compute the reconstructor
            if len(interaction_matrix_per_DM) == 0:
                self.logger.warning(f'Controller - DM {i} has no associated WFS in the control mask. Setting reconstructor to zero.')
                nModes = modal_basis[i].shape[1]
                temp_reconstructor = torch.zeros((nModes, 0), dtype=torch.float64, device=self.device)
                self.im_per_dm.append(torch.zeros((0, nModes), dtype=torch.float64, device=self.device))
            else:
                interaction_matrix_per_DM = torch.as_tensor(np.vstack(interaction_matrix_per_DM), dtype=torch.float64, device=self.device).squeeze()
                if interaction_matrix_per_DM.ndim == 1:
                    interaction_matrix_per_DM = interaction_matrix_per_DM.unsqueeze(0)
                    
                self.im_per_dm.append(interaction_matrix_per_DM)

                if reconstructionMethod == 'inversion':
                    temp_reconstructor = torch.linalg.pinv(interaction_matrix_per_DM, self.rcond[i])
                elif reconstructionMethod == 'tikhonov':
                    # (D.T@D + alfa*I)@D.T --> implemented through SVD to improve the stability of the inversion and the automation of alfa
                    H = interaction_matrix_per_DM
                    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
                    alfa = self.beta[i] * torch.max(S)**2
                    S_reg = S / (S**2 + alfa)
                    temp_reconstructor = Vh.T @ torch.diag(S_reg) @ U.T
                elif reconstructionMethod == 'tomopLA':
                    # We need a dummy reconstructor to know the dimensions (n_modes) in computeControlAction
                    temp_reconstructor = torch.zeros((interaction_matrix_per_DM.shape[1], interaction_matrix_per_DM.shape[0]), dtype=torch.float64, device=self.device)
                else:
                    self.logger.error('Controller::initializeReconstructor - Unknown reconstructor')
                    raise ValueError('Unknown reconstructor method.')
            reconstructor.append(temp_reconstructor)

        self.logger.info(f'Controller::initializeReconstructor - Reconstruction took {time.time()-t0}[s]')

        return reconstructor, modal_basis, mask, discarded_modes
    
    def initializeController(self, controllerType, reconstructor):
        """
        Initialize the control state (history buffers) based on the controller type.

        Parameters
        ----------
        controllerType : str
            Type of controller ('leaky', 'forwardPI', 'backwardPI', 'stateSpace').
        reconstructor : list
            List of reconstructor matrices per DM.

        Returns
        -------
        bool
            True if initialization succeeds.
        """
        buffer_size = max(max(self.delay), 1)
        self.command_history = [
            [torch.zeros((reconstructor[i].shape[0], 1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
            for _ in range(buffer_size)
        ]
        self.slopes_res = None
        self.slopes_polc = None

        if controllerType == 'leaky':
            self.command_previous = [torch.zeros((reconstructor[i].shape[0],1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
        elif controllerType == 'forwardPI' or controllerType == 'backwardPI':
            self.command_previous = [torch.zeros((reconstructor[i].shape[0],1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
            self.error_previous = [torch.zeros((reconstructor[i].shape[0],1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
        elif controllerType == 'stateSpace':
            self.state_previous = []
            self.A_tensor = []
            self.B_tensor = []
            self.C_tensor = []
            self.D_tensor = []
            
            for i in range(len(reconstructor)):
                n_inputs = reconstructor[i].shape[0]
                n_outputs = n_inputs
                
                A_i = torch.as_tensor(self.A[i], dtype=torch.float64, device=self.device)
                B_i = torch.as_tensor(self.B[i], dtype=torch.float64, device=self.device)
                C_i = torch.as_tensor(self.C[i], dtype=torch.float64, device=self.device)
                
                if A_i.ndim != 2 or A_i.shape[0] != A_i.shape[1]:
                    raise ValueError(f"Matrix A for DM {i} must be a square 2D matrix.")
                
                n_states = A_i.shape[0]
                
                if B_i.ndim != 2 or B_i.shape[0] != n_states or B_i.shape[1] != n_inputs:
                    raise ValueError(f"Matrix B for DM {i} must have shape ({n_states}, {n_inputs}).")
                    
                if C_i.ndim != 2 or C_i.shape[0] != n_outputs or C_i.shape[1] != n_states:
                    raise ValueError(f"Matrix C for DM {i} must have shape ({n_outputs}, {n_states}).")
                
                if self.D is not None and self.D[i] is not None:
                    D_i = torch.as_tensor(self.D[i], dtype=torch.float64, device=self.device)
                    if D_i.ndim != 2 or D_i.shape[0] != n_outputs or D_i.shape[1] != n_inputs:
                        raise ValueError(f"Matrix D for DM {i} must have shape ({n_outputs}, {n_inputs}).")
                else:
                    D_i = torch.zeros((n_outputs, n_inputs), dtype=torch.float64, device=self.device)
                    
                self.A_tensor.append(A_i)
                self.B_tensor.append(B_i)
                self.C_tensor.append(C_i)
                self.D_tensor.append(D_i)
                
                self.state_previous.append(torch.zeros((n_states, 1), dtype=torch.float64, device=self.device))
        else:
            self.logger.error('Controller::initializeController - Unknown controller')
            raise ValueError('Unknown controller.')
        return True

    def computeControlAction(self, lightPaths):
        """
        Compute the control action for each DM given the wavefront error from the light paths.

        Parameters
        ----------
        lightPaths : list
            List of LightPath objects that contain the wavefront error measurements.

        Returns
        -------
        dm_cmd : list
            List of command arrays to be sent to each Deformable Mirror.
        """
        # ---------------------------------------------------------
        # Step 1: Extract residual slopes (combined_slopes)
        # ---------------------------------------------------------
        error_res = []
        for i in range(len(self.reconstructor)):
            combined_slopes = []
            for j in range(len(lightPaths)):
                if self.mask[i,j]:
                    combined_slopes.append(lightPaths[j].get_wavefront_error())
            
            # Convert to torch
            if len(combined_slopes) > 0:
                error_res.append((-1)*torch.as_tensor(np.hstack(combined_slopes).T, dtype=torch.float64, device=self.device).unsqueeze(1)) # -1 for the feedback
            else:
                error_res.append(torch.zeros((0, 1), dtype=torch.float64, device=self.device))
        
        self.slopes_res = error_res

        if self.operationType == 'open':
            dm_cmd = []
            modal_cmd = []
            for i in range(len(self.reconstructor)):
                n_modes = self.reconstructor[i].shape[0]
                modal_cmd.append(torch.zeros((n_modes, 1), dtype=torch.float64, device=self.device))
                offset = self.discarded_modes[i]
                dm_cmd.append(self.modal_basis[i][:, offset : offset + self.reconstructor[i].shape[0]] @ modal_cmd[-1])
            self.command_history.pop(0)
            self.command_history.append(modal_cmd)
            return dm_cmd

        # ---------------------------------------------------------
        # Step 2: Compute POLC slopes if requested
        # ---------------------------------------------------------
        if self.operationType == 'polc':
            error_polc = []
            for i in range(len(self.reconstructor)):
                d_i = self.delay[i]
                # command_history stores from oldest to newest. 
                # If buffer size is max(delay), index -d_i is exactly d_i steps ago.
                cmd_delayed = self.command_history[-d_i][i]
                
                # Predict slopes from delayed commands
                pred_s = self.im_per_dm[i] @ cmd_delayed
                error_polc.append(error_res[i] + pred_s)
            self.slopes_polc = error_polc
        else:
            self.slopes_polc = None

        # Determine the error to feed to the controllers
        # We use error_res for standard PI/Leaky to prevent windup.
        error = error_res

        # ---------------------------------------------------------
        # Step 3: Control Action
        # ---------------------------------------------------------
        modal_error = []
        modal_cmd = []
        state_next = []

        if self.reconstructionMethod == 'inversion' or self.reconstructionMethod == 'tikhonov':
            for i in range(len(self.reconstructor)):
                modal_error.append(self.reconstructor[i]@error[i])

                if self.controllerType == 'leaky':
                    modal_cmd.append(self.gain[i]*modal_error[i] + self.decay[i] * self.command_previous[i])
                elif self.controllerType == 'forwardPI':
                    modal_cmd.append(self.command_previous[i] + self.gain[i] * (modal_error[i]-self.error_previous[i]) + self.ki[i]*self.error_previous[i])
                elif self.controllerType == 'backwardPI':
                    modal_cmd.append(self.command_previous[i] + self.gain[i] * (modal_error[i]-self.error_previous[i]) + self.ki[i]*modal_error[i])            
                elif self.controllerType == 'stateSpace':
                    u_k = modal_error[i]
                    x_k = self.state_previous[i]
                    
                    y_k = self.C_tensor[i] @ x_k + self.D_tensor[i] @ u_k
                    modal_cmd.append(y_k)
                    
                    x_next = self.A_tensor[i] @ x_k + self.B_tensor[i] @ u_k
                    state_next.append(x_next)

            # Compute the DM command
            dm_cmd = []
            for i in range(len(self.reconstructor)):
                offset = self.discarded_modes[i]
                dm_cmd.append(self.modal_basis[i][:, offset : offset + self.reconstructor[i].shape[0]] @ modal_cmd[i])
        elif self.reconstructionMethod == 'tomopLA':
            # Use the residuals from the first DM's mask (assuming it contains all WFS)
            global_res = error_res[0]
            if self.operationType == 'polc':
                total_pred = torch.zeros_like(global_res)
                for i in range(len(self.reconstructor)):
                    d_i = self.delay[i]
                    cmd_delayed = self.command_history[-d_i][i]
                    total_pred += self.im_per_dm[i] @ cmd_delayed
                global_slopes = (global_res + total_pred).cpu().numpy()
            else:
                global_slopes = global_res.cpu().numpy()
            
            self.tomoReconstructor.feed(global_slopes)
            self.tomoReconstructor.reconstruct(global_slopes)

            dm_cmd = []
            for i in range(len(self.reconstructor)):
                n_modes = self.reconstructor[i].shape[0]
                modal_cmd.append(torch.zeros((n_modes, 1), dtype=torch.float64, device=self.device))
                offset = self.discarded_modes[i]
                dm_cmd.append(self.modal_basis[i][:, offset : offset + n_modes] @ modal_cmd[-1])

        # Update history buffers for the next iteration
        if self.controllerType == 'leaky':
            self.command_previous = modal_cmd.copy()
        elif self.controllerType == 'forwardPI' or self.controllerType == 'backwardPI':
            self.command_previous = modal_cmd.copy()
            self.error_previous = modal_error.copy()
        elif self.controllerType == 'stateSpace':
            self.state_previous = state_next.copy()

        # Update POLC command history for the next iteration
        self.command_history.pop(0)
        self.command_history.append(modal_cmd.copy())

        return dm_cmd

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