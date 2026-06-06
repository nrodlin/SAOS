import numpy as np
from scipy.optimize import minimize
from tomoDataClasses import AtmosphereProfile, TomographyConfig
from covarianceBuilder import CovarianceBuilder


def _remove_wfs_spatial_mean(matrix, subaps_per_wfs, device=None):
    """
    Remove the spatial mean (tip-tilt) from each WFS block in the matrix.
    Works for both NumPy array and PyTorch tensor.
    """
    n_wfs = len(subaps_per_wfs)
    offsets = []
    current_offset = 0
    for n_sub in subaps_per_wfs:
        offsets.append(current_offset)
        current_offset += 2 * n_sub

    n_slopes = matrix.shape[0]

    if device is not None:
        import torch

        H = torch.eye(n_slopes, dtype=torch.float32, device=device)
        for w in range(n_wfs):
            n_sub = subaps_per_wfs[w]
            x_start = offsets[w]
            y_start = offsets[w] + n_sub
            x_idx = x_start + torch.arange(n_sub, device=device)
            y_idx = y_start + torch.arange(n_sub, device=device)
            H[x_idx[:, None], x_idx[None, :]] -= 1.0 / n_sub
            H[y_idx[:, None], y_idx[None, :]] -= 1.0 / n_sub
        return H @ matrix @ H.T
    else:
        H = np.eye(n_slopes, dtype=np.float32)
        for w in range(n_wfs):
            n_sub = subaps_per_wfs[w]
            x_start = offsets[w]
            y_start = offsets[w] + n_sub
            x_idx = x_start + np.arange(n_sub)
            y_idx = y_start + np.arange(n_sub)
            H[x_idx[:, None], x_idx[None, :]] -= 1.0 / n_sub
            H[y_idx[:, None], y_idx[None, :]] -= 1.0 / n_sub
        return H @ matrix @ H.T


def estimate_wind_from_cdt(
    Cdt_emp: np.ndarray,
    atmosphere: AtmosphereProfile,
    config: TomographyConfig,
    delay_seconds: float,
    wfs_indices: list[int] = [0, 1],
    device: str | None = None,
    remove_spatial_mean: bool = True,
    cn2_threshold: float = 0.005,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the wind velocity vectors (vx, vy) for active layers of the atmosphere.
    Uses delayed cross-covariance Cdt_emp, builder and Nelder-Mead optimization.
    """
    # Detect PyTorch device
    is_torch = False
    if device is not None:
        try:
            import torch
            is_torch = True
        except ImportError:
            device = None

    # Calculate global slope indices and offsets
    global_offsets = []
    current_offset = 0
    subaps_per_wfs = []
    for wfs in config.measured_wfs:
        n_subaps = wfs.valid_subaps.sum()
        subaps_per_wfs.append(n_subaps)
        global_offsets.append(current_offset)
        current_offset += 2 * n_subaps

    # Extract sub-selection WFS geometry and indices
    sub_wfs = [config.measured_wfs[i] for i in wfs_indices]
    subaps_sub = [subaps_per_wfs[i] for i in wfs_indices]

    indices_to_keep = []
    for idx in wfs_indices:
        n_sub = subaps_per_wfs[idx]
        x_start = global_offsets[idx]
        y_start = global_offsets[idx] + n_sub
        indices_to_keep.extend(range(x_start, x_start + n_sub))
        indices_to_keep.extend(range(y_start, y_start + n_sub))

    indices_to_keep = np.array(indices_to_keep)

    # Convert Cdt_emp to numpy float32 to extract slice
    if is_torch and isinstance(Cdt_emp, torch.Tensor):
        Cdt_emp_np = Cdt_emp.detach().cpu().numpy().astype(np.float32)
    else:
        Cdt_emp_np = np.asarray(Cdt_emp, dtype=np.float32)

    Cdt_emp_sub = Cdt_emp_np[indices_to_keep[:, None], indices_to_keep[None, :]]

    # Apply spatial mean removal to empirical Cdt if requested
    if remove_spatial_mean:
        Cdt_emp_sub = _remove_wfs_spatial_mean(Cdt_emp_sub, subaps_sub, device=None)

    # Calculate the Frobenius norm of empirical sub-covariance to normalize loss
    norm_emp = np.linalg.norm(Cdt_emp_sub, ord="fro")
    if norm_emp <= 0:
        raise ValueError("Empirical Cdt sub-matrix has zero norm.")

    # Filter active layers to optimize
    active_layers = [
        l for l in range(len(atmosphere.layer_altitudes))
        if atmosphere.cn2_weights[l] > cn2_threshold
    ]

    if len(active_layers) == 0:
        raise ValueError(f"No active atmospheric layers found with cn2_weights > {cn2_threshold}.")

    # Initialize velocities to search
    v_init = []
    for l in active_layers:
        v_init.append(atmosphere.wind_vx[l])
        v_init.append(atmosphere.wind_vy[l])
    v_init = np.array(v_init, dtype=float)

    # Setup the sub-selection covariance builder
    sub_config = TomographyConfig(
        measured_wfs=sub_wfs,
        target_wfs=None,
        dms=[],
        delay=delay_seconds,
    )
    sub_builder = CovarianceBuilder(sub_config, device=device)

    # Objective function for scipy minimize
    def objective_function(v_active):
        # Construct current wind velocities
        vx = np.copy(atmosphere.wind_vx)
        vy = np.copy(atmosphere.wind_vy)
        for i, l in enumerate(active_layers):
            vx[l] = v_active[2 * i]
            vy[l] = v_active[2 * i + 1]

        # Define temporary atmosphere profile
        atm_temp = AtmosphereProfile(
            layer_altitudes=atmosphere.layer_altitudes,
            cn2_weights=atmosphere.cn2_weights,
            r0=atmosphere.r0,
            L0=atmosphere.L0,
            wind_vx=vx,
            wind_vy=vy,
        )

        # Build analytical delayed covariance matrix for sub-selection
        Cdt_model_sub = sub_builder.build_covariance(
            output_wfs=sub_wfs,
            input_wfs=sub_wfs,
            atmosphere=atm_temp,
            predictive_delay=delay_seconds,
            shift_output=True,
        )

        if is_torch:
            # Apply tip-tilt projection on GPU if requested
            if remove_spatial_mean:
                Cdt_model_sub = _remove_wfs_spatial_mean(Cdt_model_sub, subaps_sub, device=device)
            Cdt_model_sub_np = Cdt_model_sub.detach().cpu().numpy().astype(np.float32)
        else:
            if remove_spatial_mean:
                Cdt_model_sub = _remove_wfs_spatial_mean(Cdt_model_sub, subaps_sub, device=None)
            Cdt_model_sub_np = np.asarray(Cdt_model_sub, dtype=np.float32)

        # Calculate optimal least-squares scaling factor
        scale = np.sum(Cdt_emp_sub * Cdt_model_sub_np) / np.sum(
            Cdt_model_sub_np * Cdt_model_sub_np + 1e-12
        )

        diff = Cdt_emp_sub - scale * Cdt_model_sub_np
        loss = np.linalg.norm(diff, ord="fro") / norm_emp
        return float(loss)

    # Create initial simplex to avoid extremely small default step sizes
    N = len(v_init)
    simplex = np.zeros((N + 1, N))
    simplex[0] = v_init
    for i in range(N):
        simplex[i + 1] = v_init.copy()
        simplex[i + 1, i] = 2.0  # Step size of 2.0 m/s

    # Run Powell/Nelder-Mead minimize with custom tolerances and simplex
    res = minimize(
        objective_function,
        v_init,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "maxiter": max_iter, "disp": False, "xatol": 1e-3, "fatol": 1e-5},
    )

    # Extract optimized velocities
    vx_opt = np.copy(atmosphere.wind_vx)
    vy_opt = np.copy(atmosphere.wind_vy)
    for i, l in enumerate(active_layers):
        vx_opt[l] = res.x[2 * i]
        vy_opt[l] = res.x[2 * i + 1]

    return vx_opt, vy_opt
