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
                 lightPaths=None,
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
        
        # Extracción de parámetros de los lightPaths
        self.n_layers = 0
        self.wfs_geometry = []
        self.src_positions = []
        
        if lightPaths is not None:
            # Assuming all light paths share the same atmosphere
            for lp in lightPaths:
                if lp.atm is not None:
                    self.n_layers = lp.atm.nLayer
                    break
            
            for lp in lightPaths:
                if lp.wfs is not None:
                    # Coordenadas de la fuente guía (elevación en arcsec, azimuth en grados)
                    self.src_positions.append({
                        'zenith': lp.src.coordinates[0],
                        'azimuth': lp.src.coordinates[1],
                        'is_lgs': lp.src.type == 'LGS',
                        'altitude': lp.src.altitude
                    })
                    # Geometría del sensor
                    self.wfs_geometry.append({
                        'nSubap': lp.wfs.nSubap,
                        'nSignal': lp.wfs.nSignal,
                        'subap_size': lp.wfs.subaperture_size,
                        'valid_subap_mask': getattr(lp.wfs, 'valid_subapertures', None)
                    })
        else:
            if not self.external_logger_flag:
                self.logger.warning("tomopLA: No lightPaths provided at initialization. Geometry and n_layers may be undefined.")
            else:
                self.logger.warning("tomopLA: No lightPaths provided at initialization. Geometry and n_layers may be undefined.")
                
        # Parámetros para la estimación estocástica Levenberg-Marquardt
        self.rng = np.random.default_rng(kwargs.get('seed', None))
        self.batch_size = kwargs.get('batch_size', 50)
        self.lm_inner_steps = kwargs.get('lm_inner_steps', 5)

        # Paso 1: Computar las coordenadas de las subaperturas en la pupila
        self.layer_altitudes = []
        if lightPaths is not None:
            for lp in lightPaths:
                if lp.atm is not None:
                    self.layer_altitudes = lp.atm.altitude
                    break
                    
            self._compute_pupil_coordinates()
            self._project_to_layers(self.layer_altitudes)
            self._prepare_slope_coordinates()

    def _compute_pupil_coordinates(self):
        """
        Calcula las coordenadas (x,y) en el plano de la pupila de todas las subaperturas
        válidas para cada WFS.
        """
        self.pupil_coords = []
        for wfs_geom in self.wfs_geometry:
            nSubap = wfs_geom['nSubap']
            d = wfs_geom['subap_size']
            valid_mask = wfs_geom['valid_subap_mask']
            
            if valid_mask is not None:
                # El grid de subaperturas está centrado en la pupila.
                # vector 1D con los centros de subapertura
                centers = np.arange(nSubap) * d - (nSubap - 1) * d / 2.0
                
                # Asumimos que valid_mask tiene forma (nSubap, nSubap) y el orden es (X, Y)
                valid_x, valid_y = np.where(valid_mask == True)
                
                x_coords = centers[valid_x]
                y_coords = centers[valid_y]
                
                self.pupil_coords.append((x_coords, y_coords))
            else:
                self.pupil_coords.append((None, None))
                
    def _project_to_layers(self, altitudes):
        """
        Proyecta las coordenadas de las subaperturas a las distintas altitudes de las capas.
        Devuelve self.projected_coords, que es una lista de tamaño n_layers, donde cada 
        elemento es una lista de las coordenadas (x,y) proyectadas de cada WFS en esa capa.
        
        Parameters
        ----------
        altitudes : list of float
            Lista con las altitudes de cada capa en metros.
        """
        self.projected_coords = []
        for h in altitudes:
            layer_coords = []
            for wfs_idx, (x_p, y_p) in enumerate(self.pupil_coords):
                if x_p is None:
                    layer_coords.append((None, None))
                    continue
                
                src = self.src_positions[wfs_idx]
                
                # Conversión a radianes
                zenith_rad = src['zenith'] / 206265.0
                azimuth_rad = np.deg2rad(src['azimuth'])
                
                # Desplazamiento del centro del haz por paralaje
                r = h * np.tan(zenith_rad)
                dx = r * np.cos(azimuth_rad)
                dy = r * np.sin(azimuth_rad)
                
                # Factor de compresión del cono para estrellas artificiales
                gamma = 1.0 - (h / src['altitude']) if src['is_lgs'] else 1.0
                
                # Proyección escalar
                x_h = x_p * gamma + dx
                y_h = y_p * gamma + dy
                
                layer_coords.append((x_h, y_h))
            self.projected_coords.append(layer_coords)

    def _prepare_slope_coordinates(self):
        """
        Organiza las coordenadas proyectadas en una estructura eficiente (X, Y, Axis) 
        para el cálculo vectorizado de la covarianza de pendientes.
        """
        self.layer_slope_coords = []
        for k in range(self.n_layers):
            X_all, Y_all, Axis_all, D_all = [], [], [], []
            for wfs_idx, coords in enumerate(self.projected_coords[k]):
                if coords[0] is None:
                    continue
                x_h, y_h = coords
                N = len(x_h)
                d = self.wfs_geometry[wfs_idx]['subap_size']
                
                # Pendientes en X (Axis = 0)
                X_all.append(x_h)
                Y_all.append(y_h)
                Axis_all.append(np.zeros(N, dtype=int))
                D_all.append(np.full(N, d))
                
                # Pendientes en Y (Axis = 1)
                X_all.append(x_h)
                Y_all.append(y_h)
                Axis_all.append(np.ones(N, dtype=int))
                D_all.append(np.full(N, d))
                
            self.layer_slope_coords.append({
                'X': np.concatenate(X_all),
                'Y': np.concatenate(Y_all),
                'Axis': np.concatenate(Axis_all),
                'D': np.concatenate(D_all)
            })

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

    def _phase_covariance(self, rho, r0, L0):
        """
        Función analítica de covarianza de fase (modelo de von Karman).
        C_phi(rho) = (L0/r0)^(5/3) * (Gamma(11/6) / (2^(5/6) * pi^(8/3))) * ...
        """
        import scipy.special as sp
        
        rho = np.asarray(rho, dtype=float)
        L0r0Ratio = (L0 / r0)**(5./3)
        
        # Constantes
        cst = (24. * sp.gamma(6./5) / 5.)**(5./6) * (sp.gamma(11./6) / ( (2.**(5./6)) * np.pi**(8./3) )) * L0r0Ratio
        out = np.ones_like(rho) * (24. * sp.gamma(6./5) / 5.)**(5./6) * (sp.gamma(11./6) * sp.gamma(5./6) / (2. * np.pi**(8./3))) * L0r0Ratio
        
        index = rho != 0
        u = 2 * np.pi * rho[index] / L0
        out[index] = cst * (u**(5./6)) * sp.kv(5./6, u)
        
        return out

    def compute_covariance_matrix(self, r0, L0, cn2_frac):
        """
        Construye la matriz de covarianza teórica C_s sumando las contribuciones
        de cada capa turbulenta usando aproximación por diferencias finitas.
        """
        
        if not hasattr(self, 'layer_slope_coords') or len(self.layer_slope_coords) == 0:
            raise RuntimeError("Geometry not initialized properly. Cannot compute analytical covariance.")
            
        N_total = len(self.layer_slope_coords[0]['X'])
        C_model = np.zeros((N_total, N_total))
        
        for k in range(self.n_layers):
            if cn2_frac[k] <= 0:
                continue
                
            coords = self.layer_slope_coords[k]
            X = coords['X']
            Y = coords['Y']
            Axis = coords['Axis']
            D = coords['D']
            
            dX = X[:, None] - X[None, :]
            dY = Y[:, None] - Y[None, :]
            d_ij = D[:, None] * D[None, :]
            
            mask_XX = (Axis[:, None] == 0) & (Axis[None, :] == 0)
            mask_YY = (Axis[:, None] == 1) & (Axis[None, :] == 1)
            mask_XY = (Axis[:, None] == 0) & (Axis[None, :] == 1)
            mask_YX = (Axis[:, None] == 1) & (Axis[None, :] == 0)
            
            d_val = D[0] # Asumimos d uniforme
            
            C_k = np.zeros((N_total, N_total))
            
            def C_phi(dx, dy):
                rho = np.sqrt(dx**2 + dy**2)
                return self._phase_covariance(rho, r0, L0)

            # Diferencias finitas para slopes en X-X
            if np.any(mask_XX):
                dx_xx, dy_xx = dX[mask_XX], dY[mask_XX]
                C_k[mask_XX] = (2 * C_phi(dx_xx, dy_xx) - C_phi(dx_xx - d_val, dy_xx) - C_phi(dx_xx + d_val, dy_xx)) / d_ij[mask_XX]
                
            # Diferencias finitas para slopes en Y-Y
            if np.any(mask_YY):
                dx_yy, dy_yy = dX[mask_YY], dY[mask_YY]
                C_k[mask_YY] = (2 * C_phi(dx_yy, dy_yy) - C_phi(dx_yy, dy_yy - d_val) - C_phi(dx_yy, dy_yy + d_val)) / d_ij[mask_YY]
                
            # Diferencias finitas para slopes cruzados X-Y y Y-X
            if np.any(mask_XY):
                dx_xy, dy_xy = dX[mask_XY], dY[mask_XY]
                C_k[mask_XY] = (C_phi(dx_xy - d_val/2, dy_xy + d_val/2) + C_phi(dx_xy + d_val/2, dy_xy - d_val/2)
                              - C_phi(dx_xy - d_val/2, dy_xy - d_val/2) - C_phi(dx_xy + d_val/2, dy_xy + d_val/2)) / d_ij[mask_XY]
                              
            if np.any(mask_YX):
                dx_yx, dy_yx = dX[mask_YX], dY[mask_YX]
                C_k[mask_YX] = (C_phi(dx_yx - d_val/2, dy_yx + d_val/2) + C_phi(dx_yx + d_val/2, dy_yx - d_val/2)
                              - C_phi(dx_yx - d_val/2, dy_yx - d_val/2) - C_phi(dx_yx + d_val/2, dy_yx + d_val/2)) / d_ij[mask_YX]

            C_model += cn2_frac[k] * C_k
            
        return C_model
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

        N_total = len(self.layer_slope_coords[0]['X'])
        n_slopes = min(self.slopes_covmat.shape[0], N_total)
        ii, jj = np.triu_indices(n_slopes)

        y = self.slopes_covmat[:n_slopes, :n_slopes][ii, jj]

        def residual(theta):
            r0, L0, cn2_frac = self.theta_to_physical(theta)

            C_model = self.compute_covariance_matrix(
                r0=r0,
                L0=L0,
                cn2_frac=cn2_frac,
            )

            return C_model[:n_slopes, :n_slopes][ii, jj] - y

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

        N_total = len(self.layer_slope_coords[0]['X'])
        n_slopes = min(self.slopes_covmat.shape[0], N_total)

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

            y = self.slopes_covmat[:n_slopes, :n_slopes][ii, jj]

            def residual(theta_local):
                r0, L0, cn2_frac = self.theta_to_physical(theta_local)

                C_model = self.compute_covariance_matrix(
                    r0=r0,
                    L0=L0,
                    cn2_frac=cn2_frac,
                )

                return C_model[:n_slopes, :n_slopes][ii, jj] - y

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
        if self.slopes_covmat is not None:
            self.estimate_full_lm(0.08, 20, [0.8, 0.15, 0.05, 0.03, 0.02])
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