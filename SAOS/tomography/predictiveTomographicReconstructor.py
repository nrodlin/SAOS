import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .tomoDataClasses import AtmosphereProfile, TomographyConfig
from .covarianceBuilder import CovarianceBuilder


class PredictiveTomographicReconstructor:
    """
    High-level interface for L&A / pL&A tomographic reconstruction.

    It builds:
        R_tomo = Czs Css^{-1}

    where:
        Css = cov(measured_wfs, measured_wfs)
        Czs = cov(target_wfs, measured_wfs)
    """

    def __init__(self, config: TomographyConfig, device: str | None = None):
        self.config = config
        self.device = device
        self.builder = CovarianceBuilder(config, device=device)

    def build_tomographic_reconstructor(
        self,
        atmosphere: AtmosphereProfile,
        predictive: bool = True,
        delay: float | None = None,
        regularization: float | None = None,
    ) -> np.ndarray:
        """
        Build the tomographic reconstructor.

        Parameters
        ----------
        atmosphere:
            Atmospheric profile.
        predictive:
            If True, use Czs_pred. Otherwise use static Czs.
        delay:
            Prediction delay in seconds. If None, use config.delay.
        regularization:
            Diagonal regularization added to Css. If None, use config value.

        Returns
        -------
        R_tomo:
            Matrix mapping measured slopes to target slopes.
        """

        if delay is None:
            delay = self.config.delay

        if regularization is None:
            regularization = self.config.regularization

        Css = self.builder.build_css(atmosphere)

        if predictive:
            Czs = self.builder.build_czs_predictive(
                atmosphere=atmosphere,
                delay=delay,
            )
        else:
            Czs = self.builder.build_czs(atmosphere)

        Css_reg = self._regularize_covariance(Css, regularization)

        if self.device is not None:
            import torch
            # Use PyTorch Cholesky and Solve on device
            cholesky_factor = torch.linalg.cholesky(Css_reg)
            R_tomo_T = torch.cholesky_solve(Czs.T, cholesky_factor, upper=False)
            R_tomo = R_tomo_T.T
        else:
            cholesky_factor = cho_factor(
                Css_reg,
                lower=True,
                check_finite=False,
            )

            # Solve R_tomo = Czs @ inv(Css_reg) without explicitly inverting Css.
            R_tomo = cho_solve(
                cholesky_factor,
                Czs.T,
                check_finite=False,
            ).T

        return R_tomo

    def _regularize_covariance(
        self,
        covariance: np.ndarray,
        regularization: float,
    ) -> np.ndarray:
        """
        Add diagonal regularization to a covariance matrix.
        """

        if regularization < 0:
            raise ValueError("regularization must be non-negative.")

        if self.device is not None:
            import torch
            # Regularize in-place to avoid extra tensor allocation
            covariance.diagonal().add_(regularization)
            return covariance
        else:
            covariance_reg = covariance.copy()
            diagonal_indices = np.diag_indices_from(covariance_reg)
            covariance_reg[diagonal_indices] += regularization
            return covariance_reg
    
    @staticmethod
    def build_dm_projector_from_interaction_matrix(
        Dz: np.ndarray,
        alpha: float = 1e-6,
    ) -> np.ndarray:
        """
        Build a regularized least-squares DM projector.

        Dz maps DM commands to target slopes:
            z = Dz @ u

        Cdm maps target slopes to DM commands:
            u = Cdm @ z
        """

        if Dz.ndim != 2:
            raise ValueError("Dz must be a 2D matrix.")

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        n_actuators = Dz.shape[1]

        lhs = Dz.T @ Dz
        lhs += alpha * np.eye(n_actuators)

        rhs = Dz.T

        return np.linalg.solve(lhs, rhs)    
    
    def build_total_reconstructor_from_interaction_matrix(
        self,
        atmosphere: AtmosphereProfile,
        Dz: np.ndarray,
        predictive: bool = True,
        delay: float | None = None,
        covariance_regularization: float | None = None,
        dm_regularization: float = 1e-6,
    ) -> np.ndarray:
        """
        Build total reconstructor from measured slopes to DM commands.

        R_total = Cdm @ R_tomo
        """

        R_tomo = self.build_tomographic_reconstructor(
            atmosphere=atmosphere,
            predictive=predictive,
            delay=delay,
            regularization=covariance_regularization,
        )

        if Dz.shape[0] != R_tomo.shape[0]:
            raise ValueError(
                "Dz row count must match the number of target slopes."
            )

        Cdm = self.build_dm_projector_from_interaction_matrix(
            Dz=Dz,
            alpha=dm_regularization,
        )

        return Cdm @ R_tomo    