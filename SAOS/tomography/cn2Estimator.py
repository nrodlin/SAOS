import numpy as np
from scipy.optimize import nnls


def estimate_cn2_from_css(
    Css_emp,
    Css_layers: list,
    normalize: bool = True,
    non_negative: bool = True,
    stride: int = 20,
    wfs_geometries: list = None,
    remove_spatial_mean: bool = False,
    estimate_noise: bool = False,
    frobenius_normalization: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Estimate Cn2 layer weights from an empirical Css covariance matrix.
    Optionally removes spatial mean (tip-tilt) and estimates WFS noise.
    """

    if len(Css_layers) == 0:
        raise ValueError("Css_layers cannot be empty.")

    # Detect if we are using PyTorch tensors
    is_torch = False
    device = None
    dtype = None

    try:
        import torch
        if isinstance(Css_emp, torch.Tensor):
            is_torch = True
            device = Css_emp.device
            dtype = Css_emp.dtype
    except ImportError:
        pass

    n_slopes = Css_emp.shape[0]

    # Pre-calculate offsets and counts for each WFS if needed
    if remove_spatial_mean or estimate_noise:
        if wfs_geometries is None:
            raise ValueError("wfs_geometries must be provided to estimate noise or remove spatial mean.")
        offsets = []
        current_offset = 0
        subaps_per_wfs = []
        for wfs in wfs_geometries:
            n_subaps = wfs.valid_subaps.sum()
            subaps_per_wfs.append(n_subaps)
            offsets.append(current_offset)
            current_offset += 2 * n_subaps
        n_wfs = len(wfs_geometries)

    # Perform tip-tilt subtraction (spatial mean removal) on GPU/CPU
    if remove_spatial_mean:
        if is_torch:
            import torch
            H_gpu = torch.eye(n_slopes, dtype=torch.float32, device=device)
            for w in range(n_wfs):
                n_sub = subaps_per_wfs[w]
                x_start = offsets[w]
                y_start = offsets[w] + n_sub
                x_idx = x_start + torch.arange(n_sub, device=device)
                y_idx = y_start + torch.arange(n_sub, device=device)
                H_gpu[x_idx[:, None], x_idx[None, :]] -= 1.0 / n_sub
                H_gpu[y_idx[:, None], y_idx[None, :]] -= 1.0 / n_sub
                
            Css_emp_proj = H_gpu @ Css_emp @ H_gpu.T
            Css_layers_proj_np = []
            for C in Css_layers:
                C_proj = H_gpu @ C @ H_gpu.T
                Css_layers_proj_np.append(C_proj.detach().cpu().numpy().astype(np.float32))
                del C_proj
                torch.cuda.empty_cache()
            
            Css_emp_np = Css_emp_proj.detach().cpu().numpy().astype(np.float32)
            del H_gpu, Css_emp_proj
            torch.cuda.empty_cache()
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
                
            Css_emp_np = H @ Css_emp @ H.T
            Css_layers_proj_np = [H @ C @ H.T for C in Css_layers]
    else:
        if is_torch:
            Css_emp_np = Css_emp.detach().cpu().numpy().astype(np.float32)
            Css_layers_proj_np = [layer.detach().cpu().numpy().astype(np.float32) for layer in Css_layers]
        else:
            Css_emp_np = np.asarray(Css_emp, dtype=np.float32)
            Css_layers_proj_np = [np.asarray(layer, dtype=np.float32) for layer in Css_layers]

    # Apply Frobenius normalization if requested
    if frobenius_normalization:
        Css_emp_np = Css_emp_np / np.linalg.norm(Css_emp_np, ord='fro')
        Css_layers_proj_norm = [C / np.linalg.norm(C, ord='fro') for C in Css_layers_proj_np]
    else:
        Css_layers_proj_norm = Css_layers_proj_np

    # Build atmospheric columns for design matrix A
    A_atmos = np.column_stack([
        C.reshape(-1)[::stride]
        for C in Css_layers_proj_norm
    ])

    # Build and project WFS noise columns if requested
    noise_columns = []
    if estimate_noise:
        len_sliced = (n_slopes * n_slopes + stride - 1) // stride
        if is_torch:
            import torch
            if remove_spatial_mean:
                H_gpu = torch.eye(n_slopes, dtype=torch.float32, device=device)
                for w in range(n_wfs):
                    n_sub = subaps_per_wfs[w]
                    x_start = offsets[w]
                    y_start = offsets[w] + n_sub
                    x_idx = x_start + torch.arange(n_sub, device=device)
                    y_idx = y_start + torch.arange(n_sub, device=device)
                    H_gpu[x_idx[:, None], x_idx[None, :]] -= 1.0 / n_sub
                    H_gpu[y_idx[:, None], y_idx[None, :]] -= 1.0 / n_sub

                for w in range(n_wfs):
                    n_sub = subaps_per_wfs[w]
                    x_start = offsets[w]
                    y_start = offsets[w] + n_sub
                    
                    noise_mat = torch.zeros((n_slopes, n_slopes), dtype=torch.float32, device=device)
                    noise_mat[torch.arange(x_start, x_start + n_sub, device=device), torch.arange(x_start, x_start + n_sub, device=device)] = 1.0
                    noise_mat[torch.arange(y_start, y_start + n_sub, device=device), torch.arange(y_start, y_start + n_sub, device=device)] = 1.0
                    
                    noise_mat_proj = H_gpu @ noise_mat @ H_gpu.T
                    col = noise_mat_proj.reshape(-1)[::stride].detach().cpu().numpy().astype(np.float32)
                    norm_col = np.linalg.norm(col)
                    if norm_col > 0:
                        col = col / norm_col
                    noise_columns.append(col)
                    
                    del noise_mat, noise_mat_proj
                    torch.cuda.empty_cache()
                del H_gpu
                torch.cuda.empty_cache()
            else:
                for w in range(n_wfs):
                    n_sub = subaps_per_wfs[w]
                    x_start = offsets[w]
                    y_start = offsets[w] + n_sub
                    col = np.zeros(len_sliced, dtype=np.float32)
                    diag_indices = [i * (n_slopes + 1) for i in range(x_start, x_start + n_sub)] + \
                                   [i * (n_slopes + 1) for i in range(y_start, y_start + n_sub)]
                    diag_indices = np.array(diag_indices)
                    mask = (diag_indices % stride) == 0
                    col[diag_indices[mask] // stride] = 1.0
                    norm_col = np.linalg.norm(col)
                    if norm_col > 0:
                        col = col / norm_col
                    noise_columns.append(col)
        else:
            if remove_spatial_mean:
                H = np.eye(n_slopes, dtype=np.float32)
                for w in range(n_wfs):
                    n_sub = subaps_per_wfs[w]
                    x_start = offsets[w]
                    y_start = offsets[w] + n_sub
                    x_idx = x_start + np.arange(n_sub)
                    y_idx = y_start + np.arange(n_sub)
                    H[x_idx[:, None], x_idx[None, :]] -= 1.0 / n_sub
                    H[y_idx[:, None], y_idx[None, :]] -= 1.0 / n_sub

                for w in range(n_wfs):
                    n_sub = subaps_per_wfs[w]
                    x_start = offsets[w]
                    y_start = offsets[w] + n_sub
                    noise_mat = np.zeros((n_slopes, n_slopes), dtype=np.float32)
                    noise_mat[np.arange(x_start, x_start + n_sub), np.arange(x_start, x_start + n_sub)] = 1.0
                    noise_mat[np.arange(y_start, y_start + n_sub), np.arange(y_start, y_start + n_sub)] = 1.0
                    
                    noise_mat_proj = H @ noise_mat @ H.T
                    col = noise_mat_proj.reshape(-1)[::stride]
                    norm_col = np.linalg.norm(col)
                    if norm_col > 0:
                        col = col / norm_col
                    noise_columns.append(col)
            else:
                for w in range(n_wfs):
                    n_sub = subaps_per_wfs[w]
                    x_start = offsets[w]
                    y_start = offsets[w] + n_sub
                    col = np.zeros(len_sliced, dtype=np.float32)
                    diag_indices = [i * (n_slopes + 1) for i in range(x_start, x_start + n_sub)] + \
                                   [i * (n_slopes + 1) for i in range(y_start, y_start + n_sub)]
                    diag_indices = np.array(diag_indices)
                    mask = (diag_indices % stride) == 0
                    col[diag_indices[mask] // stride] = 1.0
                    norm_col = np.linalg.norm(col)
                    if norm_col > 0:
                        col = col / norm_col
                    noise_columns.append(col)

    if estimate_noise:
        A = np.column_stack([A_atmos, np.column_stack(noise_columns)])
    else:
        A = A_atmos

    b = Css_emp_np.reshape(-1)[::stride]

    if non_negative:
        weights, _ = nnls(A, b)
    else:
        weights, *_ = np.linalg.lstsq(A, b, rcond=None)

    n_layers = len(Css_layers)
    atmos_weights = weights[:n_layers]

    if normalize:
        total = np.sum(atmos_weights)
        if total <= 0:
            raise ValueError("Estimated Cn2 weights have zero total power.")
        atmos_weights = atmos_weights / total

    if is_torch:
        import torch
        atmos_weights = torch.tensor(atmos_weights, dtype=dtype, device=device)

    if estimate_noise:
        sensor_noises = weights[n_layers:]
        if is_torch:
            sensor_noises = torch.tensor(sensor_noises, dtype=dtype, device=device)
        return atmos_weights, sensor_noises
    else:
        return atmos_weights