import numpy as np

from .tomoDataClasses import WFSGeometry, AtmosphereProfile, TomographyConfig
from .vonKarmanStructureFunction import StructureFunctionVK

class CovarianceBuilder:
    """
    Builds analytical slope covariance matrices for L&A / pL&A tomography.

    Main matrices:
        Css      = cov(measured_wfs, measured_wfs)
        Czs      = cov(target_wfs, measured_wfs)
        Czs_pred = cov(target_wfs at t + delay, measured_wfs at t)
    """

    def __init__(self, config: TomographyConfig, device: str | None = None):
        self.config = config
        self.device = device

        self.measured_wfs = config.measured_wfs
        self.target_wfs = (
            config.measured_wfs
            if config.target_wfs is None
            else config.target_wfs
        )

        self._validate_wfs_list(self.measured_wfs, "measured_wfs")
        self._validate_wfs_list(self.target_wfs, "target_wfs")

        self._geometry_cache = {}

    def build_css(self, atmosphere: AtmosphereProfile) -> np.ndarray:
        """
        Build Css = cov(measured_wfs, measured_wfs).
        """

        return self.build_covariance(
            output_wfs=self.measured_wfs,
            input_wfs=self.measured_wfs,
            atmosphere=atmosphere,
            predictive_delay=0.0,
            shift_output=False,
        )

    def build_czs(self, atmosphere: AtmosphereProfile) -> np.ndarray:
        """
        Build Czs = cov(target_wfs, measured_wfs).
        """

        return self.build_covariance(
            output_wfs=self.target_wfs,
            input_wfs=self.measured_wfs,
            atmosphere=atmosphere,
            predictive_delay=0.0,
            shift_output=False,
        )

    def build_czs_predictive(
        self,
        atmosphere: AtmosphereProfile,
        delay: float | None = None,
    ) -> np.ndarray:
        """
        Build Czs_pred = cov(target_wfs at t + delay, measured_wfs at t).
        """

        if delay is None:
            delay = self.config.delay

        return self.build_covariance(
            output_wfs=self.target_wfs,
            input_wfs=self.measured_wfs,
            atmosphere=atmosphere,
            predictive_delay=delay,
            shift_output=True,
        )

    def build_covariance(
        self,
        output_wfs: list[WFSGeometry],
        input_wfs: list[WFSGeometry],
        atmosphere: AtmosphereProfile,
        predictive_delay: float = 0.0,
        shift_output: bool = False,
    ) -> np.ndarray:
        """
        Build a generic analytical slope covariance matrix.

        The output WFS list defines the row space.
        The input WFS list defines the column space.

        If shift_output is True, the output WFS projected coordinates are
        shifted by wind * predictive_delay at each atmospheric layer.
        """

        self._validate_atmosphere(atmosphere)

        n_output_slopes = self._get_n_slopes(output_wfs)
        n_input_slopes = self._get_n_slopes(input_wfs)

        if self.device is not None:
            import torch
            covariance = torch.zeros(
                (n_output_slopes, n_input_slopes),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            covariance = np.zeros(
                (n_output_slopes, n_input_slopes),
                dtype=float,
            )

        structure_function = StructureFunctionVK(
            r0=atmosphere.r0,
            L0=atmosphere.L0,
            device=self.device,
        )

        output_offsets = self._compute_wfs_offsets(output_wfs)
        input_offsets = self._compute_wfs_offsets(input_wfs)

        for layer_index, layer_altitude in enumerate(atmosphere.layer_altitudes):
            layer_weight = atmosphere.cn2_weights[layer_index]

            if layer_weight == 0.0:
                continue

            if shift_output:
                wind_shift = (
                    atmosphere.wind_vx[layer_index] * predictive_delay,
                    atmosphere.wind_vy[layer_index] * predictive_delay,
                )
            else:
                wind_shift = None

            for output_index, output in enumerate(output_wfs):
                output_centers = self._project_wfs_to_layer(
                    wfs=output,
                    layer_altitude=layer_altitude,
                    wind_shift=wind_shift,
                )

                output_row_x, output_row_y = output_offsets[output_index]
                n_output_subaps = output_centers.shape[0]

                for input_index, input_ in enumerate(input_wfs):
                    input_centers = self._project_wfs_to_layer(
                        wfs=input_,
                        layer_altitude=layer_altitude,
                        wind_shift=None,
                    )

                    input_col_x, input_col_y = input_offsets[input_index]
                    n_input_subaps = input_centers.shape[0]

                    subap_size = output.diameter / output.n_subaps

                    if self.device is not None:
                        output_centers_t = torch.tensor(output_centers, dtype=torch.float32, device=self.device)
                        input_centers_t = torch.tensor(input_centers, dtype=torch.float32, device=self.device)
                        weight_t = torch.tensor(layer_weight, dtype=torch.float32, device=self.device)

                        C_xx, C_yy, C_xy, C_yx = (
                            slope_covariance_blocks_from_centers_torch(
                                centers_i=output_centers_t,
                                centers_j=input_centers_t,
                                subap_size=subap_size,
                                structure_function=structure_function,
                                device=self.device,
                            )
                        )

                        covariance[
                            output_row_x:output_row_x + n_output_subaps,
                            input_col_x:input_col_x + n_input_subaps,
                        ] += weight_t * C_xx

                        covariance[
                            output_row_y:output_row_y + n_output_subaps,
                            input_col_y:input_col_y + n_input_subaps,
                        ] += weight_t * C_yy

                        covariance[
                            output_row_x:output_row_x + n_output_subaps,
                            input_col_y:input_col_y + n_input_subaps,
                        ] += weight_t * C_xy

                        covariance[
                            output_row_y:output_row_y + n_output_subaps,
                            input_col_x:input_col_x + n_input_subaps,
                        ] += weight_t * C_yx
                    else:
                        C_xx, C_yy, C_xy, C_yx = (
                            slope_covariance_blocks_from_centers(
                                centers_i=output_centers,
                                centers_j=input_centers,
                                subap_size=subap_size,
                                structure_function=structure_function,
                            )
                        )

                        covariance[
                            output_row_x:output_row_x + n_output_subaps,
                            input_col_x:input_col_x + n_input_subaps,
                        ] += layer_weight * C_xx

                        covariance[
                            output_row_y:output_row_y + n_output_subaps,
                            input_col_y:input_col_y + n_input_subaps,
                        ] += layer_weight * C_yy

                        covariance[
                            output_row_x:output_row_x + n_output_subaps,
                            input_col_y:input_col_y + n_input_subaps,
                        ] += layer_weight * C_xy

                        covariance[
                            output_row_y:output_row_y + n_output_subaps,
                            input_col_x:input_col_x + n_input_subaps,
                        ] += layer_weight * C_yx

        return covariance

    @staticmethod
    def _validate_wfs_list(wfs_list: list[WFSGeometry], name: str) -> None:
        if len(wfs_list) == 0:
            raise ValueError(f"{name} cannot be empty.")

        diameter = wfs_list[0].diameter
        n_subaps = wfs_list[0].n_subaps

        for wfs in wfs_list:
            if wfs.diameter != diameter:
                raise ValueError("All WFS must currently share the same diameter.")

            if wfs.n_subaps != n_subaps:
                raise ValueError("All WFS must currently share the same n_subaps.")

    @staticmethod
    def _validate_atmosphere(atmosphere: AtmosphereProfile) -> None:
        n_layers = len(atmosphere.layer_altitudes)

        if len(atmosphere.cn2_weights) != n_layers:
            raise ValueError("cn2_weights must have the same length as layer_altitudes.")

        if len(atmosphere.wind_vx) != n_layers:
            raise ValueError("wind_vx must have the same length as layer_altitudes.")

        if len(atmosphere.wind_vy) != n_layers:
            raise ValueError("wind_vy must have the same length as layer_altitudes.")

        if atmosphere.r0 <= 0:
            raise ValueError("r0 must be positive.")

        if atmosphere.L0 <= 0:
            raise ValueError("L0 must be positive.")
        
    def _get_base_subaperture_centers(
        self,
        wfs: WFSGeometry,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return valid subaperture centers and selected indices for a WFS.

        Returns
        -------
        centers:
            Array with shape (n_valid_subaps, 2).
        selected_indices:
            Indices of selected valid subapertures.
        """

        cache_key = (
            "base_centers",
            wfs.diameter,
            wfs.n_subaps,
            id(wfs.valid_subaps),
        )

        if cache_key in self._geometry_cache:
            return self._geometry_cache[cache_key]

        subap_size = wfs.diameter / wfs.n_subaps

        x = np.linspace(
            -wfs.diameter / 2.0 + subap_size / 2.0,
             wfs.diameter / 2.0 - subap_size / 2.0,
             wfs.n_subaps,
        )

        y = np.linspace(
            -wfs.diameter / 2.0 + subap_size / 2.0,
             wfs.diameter / 2.0 - subap_size / 2.0,
             wfs.n_subaps,
        )

        xx, yy = np.meshgrid(x, y)

        all_centers = np.column_stack([
            xx.ravel(),
            yy.ravel(),
        ])

        rho = np.sqrt(all_centers[:, 0] ** 2 + all_centers[:, 1] ** 2)
        circular_mask = rho <= wfs.diameter / 2.0

        if wfs.valid_subaps is None:
            valid_mask = circular_mask
        else:
            valid_subaps = np.asarray(wfs.valid_subaps)

            if valid_subaps.dtype == bool:
                if valid_subaps.size != all_centers.shape[0]:
                    raise ValueError(
                        f"valid_subaps for WFS {wfs.name} has wrong size."
                    )

                valid_mask = circular_mask & valid_subaps.ravel()
            else:
                valid_mask = np.zeros(all_centers.shape[0], dtype=bool)
                valid_mask[valid_subaps.astype(int)] = True
                valid_mask &= circular_mask

        selected_indices = np.flatnonzero(valid_mask)
        centers = all_centers[selected_indices]

        self._geometry_cache[cache_key] = (centers, selected_indices)

        return centers, selected_indices

    @staticmethod
    def _field_angle_to_rad(
        field_angle_arcsec: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Convert field angle from arcsec to radians.
        """

        theta_x_arcsec, theta_y_arcsec = field_angle_arcsec

        return (
            theta_x_arcsec / 206265.0,
            theta_y_arcsec / 206265.0,
        )

    def _project_wfs_to_layer(
        self,
        wfs: WFSGeometry,
        layer_altitude: float,
        wind_shift: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """
        Project WFS subaperture centers onto an atmospheric layer.

        Parameters
        ----------
        wfs:
            WFS geometry.
        layer_altitude:
            Atmospheric layer altitude in meters.
        wind_shift:
            Optional layer shift (dx, dy) in meters, used for predictive
            covariance matrices.

        Returns
        -------
        projected_centers:
            Array with shape (n_valid_subaps, 2).
        """

        base_centers, _ = self._get_base_subaperture_centers(wfs)

        theta_x, theta_y = self._field_angle_to_rad(wfs.field_angle_arcsec)

        shift_x = theta_x * layer_altitude
        shift_y = theta_y * layer_altitude

        if wind_shift is not None:
            shift_x += wind_shift[0]
            shift_y += wind_shift[1]

        projected_centers = np.empty_like(base_centers)
        projected_centers[:, 0] = base_centers[:, 0] + shift_x
        projected_centers[:, 1] = base_centers[:, 1] + shift_y

        return projected_centers

    def _get_n_valid_subaps(
        self,
        wfs: WFSGeometry,
    ) -> int:
        """
        Return the number of valid subapertures for a WFS.
        """

        centers, _ = self._get_base_subaperture_centers(wfs)

        return centers.shape[0]

    def _get_n_slopes(
        self,
        wfs_list: list[WFSGeometry],
    ) -> int:
        """
        Return the total number of slopes for a list of WFSs.
        """

        return sum(2 * self._get_n_valid_subaps(wfs) for wfs in wfs_list)        
    
    def _compute_wfs_offsets(
        self,
        wfs_list: list[WFSGeometry],
    ) -> list[tuple[int, int]]:
        """
        Compute x/y slope offsets for each WFS in a concatenated slope vector.

        Slope layout per WFS:
            [x slopes, y slopes]
        """

        offsets = []
        current_offset = 0

        for wfs in wfs_list:
            n_subaps = self._get_n_valid_subaps(wfs)

            x_offset = current_offset
            y_offset = current_offset + n_subaps

            offsets.append((x_offset, y_offset))

            current_offset += 2 * n_subaps

        return offsets   
    
    def build_css_per_layer(
        self,
        atmosphere: AtmosphereProfile,
    ) -> list[np.ndarray]:
        """
        Build Css contribution for each atmospheric layer separately.
        """

        self._validate_atmosphere(atmosphere)

        layers = []
        original_weights = atmosphere.cn2_weights.copy()

        try:
            for layer_index in range(len(original_weights)):
                atmosphere.cn2_weights[:] = 0.0
                atmosphere.cn2_weights[layer_index] = 1.0

                layer_cov = self.build_css(atmosphere)
                if hasattr(layer_cov, 'cpu'):
                    layer_cov = layer_cov.cpu()
                layers.append(layer_cov)

        finally:
            atmosphere.cn2_weights[:] = original_weights

        return layers

    def build_cdt_per_layer(
        self,
        atmosphere: AtmosphereProfile,
        delay: float | None = None,
    ) -> list[np.ndarray]:
        """
        Build delayed measured-measured covariance contribution
        for each atmospheric layer separately.

        Cdt = cov(s(t + delay), s(t))
        """

        self._validate_atmosphere(atmosphere)

        if delay is None:
            delay = self.config.delay

        layers = []
        original_weights = atmosphere.cn2_weights.copy()

        try:
            for layer_index in range(len(original_weights)):
                atmosphere.cn2_weights[:] = 0.0
                atmosphere.cn2_weights[layer_index] = 1.0

                layer_covariance = self.build_covariance(
                    output_wfs=self.measured_wfs,
                    input_wfs=self.measured_wfs,
                    atmosphere=atmosphere,
                    predictive_delay=delay,
                    shift_output=True,
                )
                
                if hasattr(layer_covariance, 'cpu'):
                    layer_covariance = layer_covariance.cpu()

                layers.append(layer_covariance)

        finally:
            atmosphere.cn2_weights[:] = original_weights

        return layers     

def slope_covariance_blocks_from_centers(
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    subap_size: float,
    structure_function: StructureFunctionVK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute slope covariance blocks between two WFS projected footprints.

    Returns
    -------
    C_xx, C_yy, C_xy, C_yx:
        Slope covariance blocks.
    """

    xi = centers_i[:, 0]
    yi = centers_i[:, 1]

    xj = centers_j[:, 0]
    yj = centers_j[:, 1]

    dx = xi[:, None] - xj[None, :]
    dy = yi[:, None] - yj[None, :]

    d = subap_size
    half_d = 0.5 * d

    D0 = structure_function(np.sqrt(dx**2 + dy**2))

    C_xx = structure_function(np.sqrt((dx + d)**2 + dy**2))
    C_xx += structure_function(np.sqrt((dx - d)**2 + dy**2))
    C_xx -= 2.0 * D0
    C_xx *= 0.5

    C_yy = structure_function(np.sqrt(dx**2 + (dy + d)**2))
    C_yy += structure_function(np.sqrt(dx**2 + (dy - d)**2))
    C_yy -= 2.0 * D0
    C_yy *= 0.5

    C_xy = -structure_function(
        np.sqrt((dx + half_d)**2 + (dy - half_d)**2)
    )
    C_xy += structure_function(
        np.sqrt((dx + half_d)**2 + (dy + half_d)**2)
    )
    C_xy += structure_function(
        np.sqrt((dx - half_d)**2 + (dy - half_d)**2)
    )
    C_xy -= structure_function(
        np.sqrt((dx - half_d)**2 + (dy + half_d)**2)
    )
    C_xy *= 0.5

    # Compute yx explicitly. Do not assume C_yx == C_xy for different WFSs.
    C_yx = -structure_function(
        np.sqrt((dx - half_d)**2 + (dy + half_d)**2)
    )
    C_yx += structure_function(
        np.sqrt((dx + half_d)**2 + (dy + half_d)**2)
    )
    C_yx += structure_function(
        np.sqrt((dx - half_d)**2 + (dy - half_d)**2)
    )
    C_yx -= structure_function(
        np.sqrt((dx + half_d)**2 + (dy - half_d)**2)
    )
    C_yx *= 0.5

    return C_xx, C_yy, C_xy, C_yx


def slope_covariance_blocks_from_centers_torch(
    centers_i,
    centers_j,
    subap_size: float,
    structure_function,
    device: str,
):
    import torch
    xi = centers_i[:, 0]
    yi = centers_i[:, 1]

    xj = centers_j[:, 0]
    yj = centers_j[:, 1]

    dx = xi[:, None] - xj[None, :]
    dy = yi[:, None] - yj[None, :]

    d = subap_size
    half_d = 0.5 * d

    d_t = torch.tensor(d, dtype=torch.float32, device=device)
    half_d_t = torch.tensor(half_d, dtype=torch.float32, device=device)

    D0 = structure_function(torch.sqrt(dx**2 + dy**2))

    C_xx = structure_function(torch.sqrt((dx + d_t)**2 + dy**2))
    C_xx += structure_function(torch.sqrt((dx - d_t)**2 + dy**2))
    C_xx -= 2.0 * D0
    C_xx *= 0.5

    C_yy = structure_function(torch.sqrt(dx**2 + (dy + d_t)**2))
    C_yy += structure_function(torch.sqrt(dx**2 + (dy - d_t)**2))
    C_yy -= 2.0 * D0
    C_yy *= 0.5

    C_xy = -structure_function(
        torch.sqrt((dx + half_d_t)**2 + (dy - half_d_t)**2)
    )
    C_xy += structure_function(
        torch.sqrt((dx + half_d_t)**2 + (dy + half_d_t)**2)
    )
    C_xy += structure_function(
        torch.sqrt((dx - half_d_t)**2 + (dy - half_d_t)**2)
    )
    C_xy -= structure_function(
        torch.sqrt((dx - half_d_t)**2 + (dy + half_d_t)**2)
    )
    C_xy *= 0.5

    C_yx = -structure_function(
        torch.sqrt((dx - half_d_t)**2 + (dy + half_d_t)**2)
    )
    C_yx += structure_function(
        torch.sqrt((dx + half_d_t)**2 + (dy + half_d_t)**2)
    )
    C_yx += structure_function(
        torch.sqrt((dx - half_d_t)**2 + (dy - half_d_t)**2)
    )
    C_yx -= structure_function(
        torch.sqrt((dx + half_d_t)**2 + (dy - half_d_t)**2)
    )
    C_yx *= 0.5

    return C_xx, C_yy, C_xy, C_yx


