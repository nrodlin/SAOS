import numpy as np
import torch
from scipy.optimize import least_squares

import time

import logging
import logging.handlers
from queue import Queue

class predictiveLearnApply:
    def __init__(self,
                 window:int=1000, 
                 updateCycles:int=2000,            
                logger = None,
                **kwargs):
    
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Define class attributes
        self.tag = 'tomopLA'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

        # Initialize attributes
        self.slopes_buffer = None
        self.window = window
        self.updateCycles = updateCycles # number of cycles after the first matrix before updating the CovMat
        self.iteration = 0
        self.slopes_covmat = None

    def feed(self, combined_slopes):
        """
        Fill circular buffer with open-loop or pseudo-open-loop slopes and
        periodically estimate the slopes covariance matrix:

            C_s = 1/N sum_k s_k s_k.T
        """

        combined_slopes = np.asarray(combined_slopes).ravel()

        if self.slopes_buffer is None:
            self.slopes_buffer = np.zeros((self.window, combined_slopes.size))

        idx = self.iteration % self.window
        self.slopes_buffer[idx, :] = combined_slopes

        self.iteration += 1

        buffer_is_full = self.iteration >= self.window

        should_update = (
            self.iteration == self.window or
            (
                self.slopes_covmat is not None and
                (self.iteration - self.window) % self.updateCycles == 0
            )
        )

        if buffer_is_full and should_update:
            X = self.slopes_buffer
            self.slopes_covmat = (X.T @ X) / self.window

        return True
    
    # ------------------------------------------------------------------
    # Parameter transforms
    # ------------------------------------------------------------------

    @staticmethod
    def softmax(alpha):
        alpha = np.asarray(alpha, dtype=float)
        e = np.exp(alpha - np.max(alpha))
        return e / np.sum(e)

    def theta_to_physical(self, theta):
        """
        theta = [log(r0), log(L0), alpha_0, ..., alpha_n]

        Returns
        -------
        r0 : float
        L0 : float
        cn2_frac : ndarray, shape (n_layers,)
        """

        theta = np.asarray(theta, dtype=float)

        r0 = np.exp(theta[0])
        L0 = np.exp(theta[1])
        cn2_frac = self.softmax(theta[2:])

        return r0, L0, cn2_frac

    def physical_to_theta(self, r0, L0, cn2_frac):
        cn2_frac = np.asarray(cn2_frac, dtype=float)
        cn2_frac = cn2_frac / np.sum(cn2_frac)

        theta = np.zeros(2 + self.n_layers, dtype=float)

        theta[0] = np.log(r0)
        theta[1] = np.log(L0)

        # Inverse-softmax up to an arbitrary constant
        theta[2:] = np.log(cn2_frac + 1e-15)

        return theta 

    # ------------------------------------------------------------------
    # Model covariance
    # ------------------------------------------------------------------

    def compute_covariance_matrix(self, r0, L0, cn2_frac):
        """
        Build theoretical covariance matrix:

            C_s = C_s(r0, L0, Cn2 fractions)

        You must replace the body of this function with your analytical
        slope covariance model F{r_ij, rho}.

        Expected return
        ---------------
        C_model : ndarray, shape (n_slopes, n_slopes)
        """

        raise NotImplementedError(
            "Implement your analytical slope covariance model here."
        )   
    # ------------------------------------------------------------------
    # Full nonlinear least-squares
    # ------------------------------------------------------------------

    def estimate_full_lm(self, r0_init, L0_init, cn2_frac_init):
        """
        Full Levenberg-Marquardt fit using all covariance terms.

        This is expensive for large slope vectors.
        """

        if self.slopes_covmat is None:
            raise RuntimeError("slopes_covmat has not been estimated yet.")

        theta0 = self.physical_to_theta(
            r0=r0_init,
            L0=L0_init,
            cn2_frac=cn2_frac_init,
        )

        n_slopes = self.slopes_covmat.shape[0]
        ii, jj = np.triu_indices(n_slopes)

        y = self.slopes_covmat[ii, jj]

        def residual(theta):
            r0, L0, cn2_frac = self.theta_to_physical(theta)

            C_model = self.compute_covariance_matrix(
                r0=r0,
                L0=L0,
                cn2_frac=cn2_frac,
            )

            return C_model[ii, jj] - y

        result = least_squares(
            residual,
            x0=theta0,
            method="lm",
            x_scale="jac",
        )

        self.theta = result.x
        self.r0_hat, self.L0_hat, self.cn2_frac_hat = self.theta_to_physical(
            self.theta
        )

        return self.r0_hat, self.L0_hat, self.cn2_frac_hat, result

    # ------------------------------------------------------------------
    # Stochastic Levenberg-Marquardt
    # ------------------------------------------------------------------

    def estimate_stochastic_lm(
        self,
        r0_init,
        L0_init,
        cn2_frac_init,
        n_iterations=50,
    ):
        """
        Stochastic Levenberg-Marquardt.

        Each external iteration fits only a random subset of covariance
        matrix elements.
        """

        if self.slopes_covmat is None:
            raise RuntimeError("slopes_covmat has not been estimated yet.")

        theta = self.physical_to_theta(
            r0=r0_init,
            L0=L0_init,
            cn2_frac=cn2_frac_init,
        )

        n_slopes = self.slopes_covmat.shape[0]

        ii_all, jj_all = np.triu_indices(n_slopes)
        n_terms = ii_all.size

        history = []

        for it in range(n_iterations):
            sample = self.rng.choice(
                n_terms,
                size=min(self.batch_size, n_terms),
                replace=False,
            )

            ii = ii_all[sample]
            jj = jj_all[sample]

            y = self.slopes_covmat[ii, jj]

            def residual(theta_local):
                r0, L0, cn2_frac = self.theta_to_physical(theta_local)

                C_model = self.compute_covariance_matrix(
                    r0=r0,
                    L0=L0,
                    cn2_frac=cn2_frac,
                )

                return C_model[ii, jj] - y

            result = least_squares(
                residual,
                x0=theta,
                method="lm",
                max_nfev=self.lm_inner_steps,
                x_scale="jac",
            )

            theta = result.x

            r0, L0, cn2_frac = self.theta_to_physical(theta)

            rms = np.sqrt(np.mean(result.fun**2))

            history.append(
                {
                    "iteration": it,
                    "r0": r0,
                    "L0": L0,
                    "cn2_frac": cn2_frac.copy(),
                    "rms": rms,
                    "success": result.success,
                }
            )

        self.theta = theta
        self.r0_hat, self.L0_hat, self.cn2_frac_hat = self.theta_to_physical(
            theta
        )

        return self.r0_hat, self.L0_hat, self.cn2_frac_hat, history            
    
    def reconstruct(self, combined_slopes):
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
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()