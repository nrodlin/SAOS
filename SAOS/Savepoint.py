"""
Created on May 15 2025
@author: nrodlin
"""
import numpy as np
import h5py
import os
import sys
import json
import getpass
import platform
import subprocess
from datetime import datetime, timezone

from scipy import ndimage, signal

import logging
from .LoggingHelper import get_logger

"""
Savepoint Module
=================

This module contains the `Savepoint` class, used for to save the data of adaptive optics simulations.
"""


def peak_subpixel(img, half_width=1.0, step=1.0 / 32):
    """
    Peak value of a band-limited image: maximum of its Fourier (sinc) interpolation on a grid of `step` pixels within
    +-half_width pixels of the brightest pixel. Exact for images sampled at Nyquist or better (diffraction-limited PSFs
    with lambda/D >= 2 px); for undersampled images it is an interpolation estimate.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    half_width : float
        Half size of the search region around the brightest pixel [px].
    step : float
        Sampling of the search grid [px].

    Returns
    -------
    float
        Interpolated peak value (same units as img).
    """
    img = np.asarray(img, dtype=float)
    n0, n1 = img.shape
    r0, c0 = np.unravel_index(np.argmax(img), img.shape)
    F = np.fft.fft2(img)
    off = np.arange(-half_width, half_width + step / 2, step)
    # Local inverse DFT: I(y, x) = 1/(n0 n1) sum_k,l F[k, l] exp(2 pi i (k y / n0 + l x / n1)), k, l integer frequencies
    Ey = np.exp(2j * np.pi * np.outer(r0 + off, np.fft.fftfreq(n0) * n0) / n0)
    Ex = np.exp(2j * np.pi * np.outer(np.fft.fftfreq(n1) * n1, c0 + off) / n1)
    local = np.real(Ey @ F @ Ex) / (n0 * n1)
    return float(max(local.max(), img[r0, c0]))


_MAX_ATTR_SIZE = 10000                                     # larger arrays are not stored as metadata
_MAX_ATTR_BYTES = 16384                                    # larger values go to datasets (HDF5 attributes are limited to 64 kB per object header)

_RUN_INFO = None


def _git_info(path):
    """(commit, dirty) of the git repository holding path; (None, None) outside a repository or without git."""
    try:
        commit = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                                text=True, timeout=5)
        return commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def run_info():
    """
    Provenance of the running simulation: SAOS version and git commit, Python and library versions, host, user, command
    line. Computed once per process.

    Returns
    -------
    dict
        Metadata (str, bool or None values).
    """
    global _RUN_INFO
    if _RUN_INFO is None:
        from . import __version__
        commit, dirty = _git_info(os.path.dirname(os.path.abspath(__file__)))
        info = dict(saos_version=__version__, saos_git_commit=commit, saos_git_dirty=dirty,
                    python_version=sys.version.split()[0], gil_enabled=getattr(sys, '_is_gil_enabled', lambda: True)(),
                    hostname=platform.node(), platform=platform.platform(), command_line=' '.join(sys.argv), cwd=os.getcwd())
        try:
            info['user'] = getpass.getuser()
        except Exception:
            info['user'] = None
        for name in ('numpy', 'scipy', 'h5py', 'torch'):
            module = sys.modules.get(name)
            info[f'{name}_version'] = getattr(module, '__version__', None) if module is not None else None
        torch = sys.modules.get('torch')
        try:
            info['cuda_device'] = torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else None
        except Exception:
            info['cuda_device'] = None
        _RUN_INFO = info
    return dict(_RUN_INFO)


def _attr_value(value):
    """Value storable as an HDF5 attribute (scalars, strings, small numeric arrays; dicts and mixed lists as JSON), or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)                                   # np.str_ has no HDF5 conversion
    if isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
        return value
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value)
        except ValueError:
            return json.dumps(value, default=str)
        if arr.dtype.kind in 'biuf' and arr.size <= _MAX_ATTR_SIZE:
            return arr
        if arr.dtype.kind in 'OUS' and arr.size <= _MAX_ATTR_SIZE:
            return json.dumps(np.asarray(value, dtype=object).tolist(), default=str)
        return None
    return None


def _write_attrs(group, values):
    for key, value in values.items():
        value = _attr_value(value)
        if value is None:
            continue
        size = value.nbytes if isinstance(value, np.ndarray) else len(value.encode()) if isinstance(value, str) else 0
        if size <= _MAX_ATTR_BYTES:
            group.attrs[key] = value
        elif key not in group:                              # written once; never replaces a data member of the group
            group.create_dataset(key, data=value)


def _pick(obj, names):
    return {n: getattr(obj, n) for n in names if hasattr(obj, n)}


def module_metadata(obj):
    """
    Configuration of a SAOS module (Telescope, Atmosphere, DeformableMirror, Source/ExtendedSource, WFS, ScienceCam,
    Vibration, LightPath) as a flat dict of metadata. Unknown objects give only their class.
    """
    meta = {'class': type(obj).__name__}
    tag = getattr(obj, 'tag', None)
    if tag == 'lightpath':
        meta['delay'] = getattr(obj, 'delay', None)
        dms = getattr(obj, 'dm', None) or []
        meta['n_dms'] = len(dms)
        meta['dm_altitudes'] = [getattr(d, 'altitude', None) for d in dms]
        for prefix, sub in (('src', getattr(obj, 'src', None)), ('wfs', getattr(obj, 'wfs', None)), ('sci', getattr(obj, 'sci', None)),
                            ('vibration', getattr(obj, 'vibration', None))):
            if sub is not None:
                meta.update({f'{prefix}_{k}': v for k, v in module_metadata(sub).items()})
        meta['has_atmosphere'] = getattr(obj, 'atm', None) is not None
        return meta
    if tag == 'atmosphere':
        meta.update(_pick(obj, ('r0', 'L0', 'fractionalR0', 'altitude', 'windSpeed', 'windDirection', 'nLayer', 'randomState')))
        if hasattr(obj, 'zenith'):
            meta['zenith_deg'] = float(np.rad2deg(obj.zenith))
        return meta
    if tag == 'deformableMirror':
        meta.update(_pick(obj, ('typeDM', 'nActs', 'nValidAct', 'altitude', 'pitch', 'mechCoupling', 'validActThreshpercentage',
                                'maxStrokePtV', 'dynamic_model_path')))
        if getattr(obj, 'misReg', None) is not None:
            meta.update({f'misReg_{k}': v for k, v in _pick(obj.misReg, ('shiftX', 'shiftY', 'rotation', 'radialScaling')).items()})
        return meta
    if tag in ('source', 'sun'):
        meta.update(_pick(obj, ('tag', 'optBand', 'wavelength', 'bandwidth', 'coordinates', 'magnitude', 'fov', 'nSubDirs')))
        return meta
    if tag == 'telescope':
        meta.update(_pick(obj, ('D', 'resolution', 'pixelSize', 'centralObstruction', 'samplingTime', 'fov')))
        meta['spiders'] = getattr(obj, 'spiders', None) or 'none'
        return meta
    if hasattr(obj, 'nSubap'):                                                          # wavefront sensors
        meta.update(_pick(obj, ('nSubap', 'nValidSubaperture', 'plate_scale', 'fieldOfView', 'lightRatio', 'use_brightest',
                                'unit_in_rad', 'guardPx', 'fft_fieldOfView_oversampling', 'validSubapThreshold', 'camera_params')))
        return meta
    if hasattr(obj, 'integrationTime'):                                                 # science cameras
        meta.update(_pick(obj, ('plate_scale', 'fieldOfView', 'nPix', 'integrationTime', 'samplingTime', 'lightRatio', 'camera_params')))
        return meta
    meta.update(_pick(obj, ('Ts',)))
    return meta


class Savepoint:
    def __init__(self, file_path=None, atm=0, atm_per_dir=0, dm=0, dm_per_dir=0, slopes=0, error= 0, wfs=0, wfs_frame=0, sci=0, sci_frame=0, 
                 only_metrics=False, only_long_exp=False, metadata=None, logger=None):
        """
        Initialize the Savepoint object for saving simulation data.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file where data will be saved. By default, creates a folder in the home directory.
        atm : int
            Flag to save atmosphere phase, per layer. If 0, the atmosphere phase will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        atm_per_dir : int
            Flag to save the atmosphere phase projection per direction. If 0, it will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        dm : int
            Flag to save deformable mirror phase. If 0, the DM phase will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        dm_per_dir : int
            Flag to save the DM phase projection per direction. If 0, it will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        slopes : int
            Flag to save slopes data. If 0, the slopes data will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        error : int
            Flag to save the wavefront error buffer. If 0, the error data will not be saved.
            Larger than 0, it will be saved with the specified decimation factor.
        wfs : int
            Flag to save wavefront sensor phase. If 0, the WFS data will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        wfs_frame : int
            Flag to save WFS frame. If 0, the WFS frame will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        sci : int
            Flag to save science phase. If 0, the science phase will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        sci_frame : int
            Flag to save science frame. If 0, the science frame will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        only_metrics : bool
            If True, only the metrics of the buffers are saved. If false, the data
            is included as well.
        only_long_exp : bool
            If True, only the long exposure frame is saved. By default, False.
        metadata : dict, optional
            User metadata of the simulation (case, draw, controller settings...), stored in /metadata/user (as attributes; values
            above 16 kB as datasets). More can be
            added with add_metadata().
        logger : logging.Logger, optional
            External logger to use. If None, initializes internal logging.

        Notes
        -----
        Every file has a group /metadata, written at the first save:
          - its attributes: SAOS version and git commit (saos_git_dirty = uncommitted changes), Python, library versions,
            GIL, host, user, command line, creation time (UTC) and the Savepoint settings; last_iteration and
            last_saved_utc are updated at every save (an interrupted run can be detected);
          - /metadata/user: the user metadata;
          - /metadata/telescope, /metadata/atmosphere and /metadata/dm_<i>: configuration of the modules of the first
            light path that has them (written whatever is saved);
        and every LightPath_<k>, Atmosphere_<i> and DeformableMirror_<i> group has the configuration of its module as
        attributes (source, WFS, science camera, delay...). Values are scalars, small arrays or JSON strings.
        """
        self.logger = logger if logger is not None else get_logger()
        
        self.file_path = file_path
        self.atm = atm
        self.atm_per_dir = atm_per_dir
        self.dm = dm
        self.dm_per_dir = dm_per_dir
        self.slopes = slopes
        self.error = error
        self.wfs = wfs
        self.wfs_frame = wfs_frame
        self.sci = sci
        self.only_long_exp = only_long_exp
        self.sci_frame = sci_frame
        self.only_metrics = only_metrics
        self.metadata = dict(metadata) if metadata else {}
        self._metadata_pending = bool(self.metadata)

        # Create the file path if not provided
        if not self.file_path:
            simulations_dir = os.path.expanduser('~/simulations/results')
            os.makedirs(simulations_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.file_path = os.path.join(simulations_dir, f'saos_savepoint_{timestamp}.h5')
        
        self.logger.info(f'Savepoint::__init__ - Saving path (not yet created): {self.file_path}')


    def add_metadata(self, **metadata):
        """
        Add user metadata (stored in /metadata/user at the next save; a key given twice keeps the last value).

        Parameters
        ----------
        **metadata
            Name and value of every entry: scalars, strings, small arrays, lists or dicts (as JSON).
        """
        self.metadata.update(metadata)
        self._metadata_pending = True

    def _settings(self):
        names = ('atm', 'atm_per_dir', 'dm', 'dm_per_dir', 'slopes', 'error', 'wfs', 'wfs_frame', 'sci', 'sci_frame',
                 'only_metrics', 'only_long_exp')
        return json.dumps({n: getattr(self, n) for n in names}, default=str)

    def _write_metadata(self, f, data_object, iteration):
        """Run, module and user metadata in /metadata (see __init__)."""
        meta = f.require_group('metadata')
        if 'saos_version' not in meta.attrs:
            _write_attrs(meta, run_info())
            meta.attrs['created_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            meta.attrs['savepoint_settings'] = self._settings()
        meta.attrs['last_iteration'] = int(iteration)
        meta.attrs['last_saved_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        if self._metadata_pending:
            _write_attrs(meta.require_group('user'), self.metadata)
            self._metadata_pending = False
        for obj in data_object:
            if getattr(obj, 'tag', None) != 'lightpath':
                continue
            if 'telescope' not in meta and getattr(obj, 'tel', None) is not None:
                _write_attrs(meta.create_group('telescope'), module_metadata(obj.tel))
            if 'atmosphere' not in meta and getattr(obj, 'atm', None) is not None:
                _write_attrs(meta.create_group('atmosphere'), module_metadata(obj.atm))
            for i, dm in enumerate(getattr(obj, 'dm', None) or []):
                if f'dm_{i}' not in meta:
                    _write_attrs(meta.create_group(f'dm_{i}'), module_metadata(dm))

    def _resolve_setting(self, setting_value, lp, lp_index):
        """
        Resolves the setting for a specific lightpath index, returning either:
        - An integer (decimation factor)
        - A dict with 'short' and/or 'long' keys mapped to integers (decimation factors)
        - Or 0 (disabled)
        """
        if isinstance(setting_value, dict):
            # Check if it is a direct configuration for short/long exposure: {'short': X, 'long': Y}
            is_direct_config = any(k in setting_value for k in ('short', 'long'))
            
            # Check if the dictionary contains a key associated with the LightPath index
            has_lp_key = False
            if lp_index is not None:
                has_lp_key = (lp_index in setting_value)
                
            if has_lp_key:
                # It is a mapping from LightPath index to configuration value
                lp_val = setting_value[lp_index]
                return self._resolve_setting(lp_val, lp, lp_index)
            
            elif is_direct_config:
                # It is a direct configuration of frame types {'short': X, 'long': Y}
                resolved_dict = {}
                for k in ('short', 'long'):
                    val = setting_value.get(k, 0)
                    resolved_dict[k] = self._resolve_setting(val, lp, lp_index)
                return resolved_dict
            
            else:
                # Generic dictionary without specific keys, look for a 'default' fallback
                if 'default' in setting_value:
                    return self._resolve_setting(setting_value['default'], lp, lp_index)
                return 0
                
        elif isinstance(setting_value, (list, tuple)):
            if lp_index is not None and lp_index < len(setting_value):
                lp_val = setting_value[lp_index]
                return self._resolve_setting(lp_val, lp, lp_index)
            return 0
        else:
            # Scalar (int, float, bool)
            if setting_value is None:
                return 0
            return int(setting_value)

    def initialize_hdf5_file(self, f, group_name, data_group, iteration):
        """
        Initialize the HDF5 file with the structure for saving light path data.

        Parameters
        ----------
        f : h5py.File
            Opened HDF5 file object.
        group_name : int
            Name of the group that will be initialized in the HDF5 file.            
        data_group : Object storing the data to be saved
            Data object containing simulation results.
        iteration : int
            Current iteration number.
        """
        group = f.create_group(group_name)
        _write_attrs(group, module_metadata(data_group))

        lp_index = None
        if group_name.startswith('LightPath_'):
            try:
                lp_index = int(group_name.split('_')[-1])
            except ValueError:
                pass

        if self.atm and group_name.find('Atmosphere')>=0:
            for i in range(data_group.nLayer):
                layer_name = 'layer_' + str(i+1)
                atm_layer_grp = group.create_group(layer_name)
                self.custom_create_dataset('atmosphere_layered', atm_layer_grp, iteration, getattr(data_group, layer_name).screen.scrn, mask=None)
        
        if self.dm and group_name.find('DeformableMirror')>=0:
            self.custom_create_dataset('dm_layer', group, iteration, data_group.dm_layer.cmd_1D, mask=data_group.validAct[None, ...])

        # Resolve setting values for this light path
        atm_per_dir_setting = self._resolve_setting(self.atm_per_dir, data_group, lp_index)
        dm_per_dir_setting = self._resolve_setting(self.dm_per_dir, data_group, lp_index)
        slopes_setting = self._resolve_setting(self.slopes, data_group, lp_index)
        error_setting = self._resolve_setting(self.error, data_group, lp_index)
        wfs_setting = self._resolve_setting(self.wfs, data_group, lp_index)
        sci_setting = self._resolve_setting(self.sci, data_group, lp_index)
        wfs_frame_setting = self._resolve_setting(self.wfs_frame, data_group, lp_index)
        sci_frame_setting = self._resolve_setting(self.sci_frame, data_group, lp_index)

        if isinstance(sci_frame_setting, dict):
            short_rate = sci_frame_setting.get('short', 0)
            long_rate = sci_frame_setting.get('long', 0)
        else:
            short_rate = sci_frame_setting
            long_rate = sci_frame_setting
                
        if atm_per_dir_setting and group_name.find('LightPath')>=0:
            # Pupil mask to compute statistics
            mask = data_group.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            atm_opd_grp = group.create_group('atm_opd')
            self.custom_create_dataset('opd', atm_opd_grp, iteration, data_group.atmosphere_opd, mask)
            atm_phase_grp = group.create_group('atm_phase')
            self.custom_create_dataset('phase', atm_phase_grp, iteration, data_group.atmosphere_phase, mask)
        
        if dm_per_dir_setting and group_name.find('LightPath')>=0:
            for i in range(len(data_group.dm)):
                # Pupil mask to compute statistics --> we force the data to be 3D to stadarize night and solar cases
                mask = data_group.tel.pupil[None, ...]
                # Create group and add the datasets with the statistics
                dm_opd_grp = group.create_group('dm_opd_' + str(i))
                self.custom_create_dataset('opd', dm_opd_grp, iteration, data_group.dm_opd[i], mask)
                dm_phase_grp = group.create_group('dm_phase_' + str(i))
                self.custom_create_dataset('phase', dm_phase_grp, iteration, data_group.dm_phase[i], mask)

        if slopes_setting and group_name.find('LightPath')>=0 and (data_group.wfs is not None):
            # Create group and add the datasets with the statistics ( mask is not needed)
            slopes1d_grp = group.create_group('slopes_1D')
            self.custom_create_dataset('slopes_1D', slopes1d_grp, iteration, data_group.slopes_1D, mask=None)
            slopes2d_grp = group.create_group('slopes_2D')
            self.custom_create_dataset('slopes_2D', slopes2d_grp, iteration, data_group.slopes_2D, mask=None)

        if error_setting and group_name.find('LightPath')>=0 and (data_group.wfs is not None):
            # Create group and add the datasets with the statistics ( mask is not needed)
            error_grp = group.create_group('error')
            self.custom_create_dataset('slopes_1D', error_grp, iteration, data_group.get_wavefront_error(), mask=None)            

        if wfs_setting and group_name.find('LightPath')>=0 and (data_group.wfs is not None):
            # Pupil mask to compute statistics
            mask = data_group.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            wfs_opd_grp = group.create_group('wfs_opd')
            self.custom_create_dataset('opd', wfs_opd_grp, iteration, data_group.wfs_opd, mask)
            wfs_phase_grp = group.create_group('wfs_phase')
            self.custom_create_dataset('phase', wfs_phase_grp, iteration, data_group.wfs_phase, mask)

        if sci_setting and group_name.find('LightPath')>=0 and (data_group.sci is not None):
            # Pupil mask to compute statistics
            mask = data_group.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            sci_opd_grp = group.create_group('sci_opd')
            self.custom_create_dataset('opd', sci_opd_grp, iteration, data_group.sci_opd, mask)
            sci_phase_grp = group.create_group('sci_phase')
            self.custom_create_dataset('phase', sci_phase_grp, iteration, data_group.sci_phase, mask)

        if wfs_frame_setting and group_name.find('LightPath')>=0 and (data_group.wfs is not None):
            # Create group and add the datasets with the statistics ( mask is not needed)
            wfs_frame_grp = group.create_group('wfs_frame')
            self.custom_create_dataset('wfs_frame', wfs_frame_grp, iteration, data_group.wfs_frame, mask=None)

        if (short_rate or long_rate) and group_name.find('LightPath')>=0 and (data_group.sci is not None):
            # Create group and add the datasets with the statistics ( mask is not needed)
            if short_rate and data_group.sci_frame is not None:
                sci_frame_grp = group.create_group('sci_frame_shortExp')
                self.custom_create_dataset('sci_frame', sci_frame_grp, iteration, data_group.sci_frame, None, data_group)
            if long_rate and data_group.long_exposure_frame is not None:
                sci_frame_grp = group.create_group('sci_frame_longExp')
                self.custom_create_dataset('sci_frame', sci_frame_grp, iteration, data_group.long_exposure_frame, None, data_group)

    def _get_lp_index_from_group(self, group):
        """
        Extract the LightPath, Atmosphere, or DM index from the HDF5 group name.
        """
        parts = group.name.split('/')
        for part in parts:
            if part.startswith('LightPath_'):
                try:
                    return int(part.split('_')[-1])
                except ValueError:
                    pass
            elif part.startswith('Atmosphere_'):
                try:
                    return int(part.split('_')[-1])
                except ValueError:
                    pass
            elif part.startswith('DeformableMirror_'):
                try:
                    return int(part.split('_')[-1])
                except ValueError:
                    pass
        return None

    def custom_create_dataset(self, data_type, group, iteration, data, mask, lp=None):
        """
        Create a dataset in the HDF5 file.

        Parameters
        ----------
        data_type : str
            Type of data ('phase', 'opd', 'slopes_1D', 'slopes_2D', 'sci_frame').
        group : h5py.Group
            The group in which to create the dataset.
        iteration : int
            Current iteration number.
        data : numpy.ndarray
            Data to be saved in the dataset.
        mask : numpy.ndarray
            Mask to apply to the data.
        lp : LightPath
            LightPath object containing simulation modules, required to get the ideal PSF
            and compute the Strehl ratio.
        """
        if np.ndim(data) == 2:
            data = data[None, ...]

        lp_index = self._get_lp_index_from_group(group)
        only_metrics_resolved = bool(self._resolve_setting(self.only_metrics, lp, lp_index))

        if not only_metrics_resolved:
            group.create_dataset('data', data=data[None, ...], maxshape=(None,) + data.shape, chunks=True)        
        group.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)

        stats = self.compute_stats(data, mask, data_type, lp)
        
        for stat_name, value in stats.items():
            if value is not None:
                group.create_dataset(stat_name, data=value, maxshape=(None,) + value.shape[1:], chunks=True)

    def compute_stats(self, data, mask, data_type, lp=None):
        """
        Compute statistics for the given data.

        Parameters
        ----------
        data : numpy.ndarray
            Data to compute statistics on.
        mask : numpy.ndarray
            Mask to apply to the data.
        data_type : str
            Type of data ('phase', 'opd', 'slopes_1D', 'slopes_2D', 'sci_frame').
        lp : LightPath
            LightPath object containing simulation modules, required to get the ideal PSF
            and compute the Strehl ratio.

        Notes
        -----
        Strehl ratio of point-source frames ('strehl'): measured SR, as an instrument would estimate it,
        SR = (peak / energy in the field) / (ideal peak / ideal energy in the field). The normalisation uses the energy
        recorded inside the camera field (the observational convention), so with small fields the aberrated halo that
        leaves the field raises the SR slightly. The peaks are the maxima of the band-limited (Fourier) interpolation of
        the frames (see peak_subpixel), so the SR does not depend on where the PSF peak falls with respect to the pixel
        grid; this is exact for frames sampled at Nyquist or better.

        Returns
        -------
        dict
            Dictionary containing the computed statistics.
        """

        dict_stats = {
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'rms': None,
            'contrast': None,
            'MFGS': None,
            'strehl': None
        }

        if data_type == 'phase' or data_type == 'opd':
            dict_stats['min'] = np.array([np.min(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['max'] = np.array([np.max(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['mean'] = np.array([np.mean(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['std'] = np.array([np.std(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['rms'] = np.array([np.sqrt(np.mean(data.reshape(data.shape[0], -1)[:, mask.ravel()]**2, axis=1))])
        elif data_type == 'atmosphere_layered':
            dict_stats['min'] = np.array([np.min(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['max'] = np.array([np.max(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['mean'] = np.array([np.mean(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['std'] = np.array([np.std(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['rms'] = np.array([np.sqrt(np.mean(data.reshape(data.shape[0], -1)**2, axis=1))])
        elif data_type == 'dm_layer':
            dict_stats['min'] = np.array([np.min(data[mask.ravel()])])
            dict_stats['max'] = np.array([np.max(data[mask.ravel()])])
            dict_stats['mean'] = np.array([np.mean(data[mask.ravel()])])
            dict_stats['std'] = np.array([np.std(data[mask.ravel()])])
            dict_stats['rms'] = np.array([np.sqrt(np.mean(data[mask.ravel()]**2))])
        elif data_type == 'slopes_1D':
            dict_stats['min'] = np.array([[np.min(data[0:data.shape[0]//2]), np.min(data[data.shape[0]//2:])]])
            dict_stats['max'] = np.array([[np.max(data[0:data.shape[0]//2]), np.max(data[data.shape[0]//2:])]])
            dict_stats['mean'] = np.array([[np.mean(data[0:data.shape[0]//2]), np.mean(data[data.shape[0]//2:])]])
            dict_stats['std'] = np.array([[np.std(data[0:data.shape[0]//2]), np.std(data[data.shape[0]//2:])]])
            dict_stats['rms'] = np.array([[np.sqrt(np.mean(data[0:data.shape[0]//2]**2)), np.sqrt(np.mean(data[data.shape[0]//2:]**2))]])

        elif data_type == 'slopes_2D':
            # data is (1, 2 n, n): the x block is rows [0, n) and the y block rows [n, 2 n) of the 2D map
            n_rows = data.shape[1]//2
            dict_stats['min'] = np.array([[np.min(data[0, :n_rows, :]), np.min(data[0, n_rows:, :])]])
            dict_stats['max'] = np.array([[np.max(data[0, :n_rows, :]), np.max(data[0, n_rows:, :])]])
        
        elif data_type == 'sci_frame' and (lp is not None):
            if lp.src.tag == 'sun':
                mean_intensity = np.mean(data)
                std_intensity = np.std(data)
                if mean_intensity != 0:
                    dict_stats['contrast'] = np.array([std_intensity / mean_intensity])
                else:
                    dict_stats['contrast'] = np.array([0.0])
                
                # Normalize by the mean intensity so that MFGS is independent of the frame units (C would dominate otherwise)
                squeezed_data = np.squeeze(data).astype(np.float64)
                if mean_intensity != 0:
                    squeezed_data = squeezed_data / mean_intensity
                # Get median filtered image
                # (reflected borders: zero padding biased the border medians low and created spurious gradients)
                med = ndimage.median_filter(squeezed_data, size=3, mode='reflect')
                # Create Gradient filters
                kx = np.array(([-3, 0, 3], [-10, 0, 10], [-3, 0, 3]))
                ky = np.rot90(kx)
                # compute gradient of the median filtered image
                kx_med = signal.convolve2d(med, kx, mode='same', boundary='symm')
                ky_med = signal.convolve2d(med, ky, mode='same', boundary='symm')
                # Compute gradient of the original image
                kx_img = signal.convolve2d(squeezed_data, kx, mode='same', boundary='symm')
                ky_img = signal.convolve2d(squeezed_data, ky, mode='same', boundary='symm')
                # compute gradient magnitudes pixel-by-pixel
                g_img = np.hypot(kx_img, ky_img)
                g_med = np.hypot(kx_med, ky_med)
                # compute similarity map
                C = 1e-10
                gs_map = (2.0 * g_img * g_med + C) / (g_img**2 + g_med**2 + C)
                # compute average MFGS
                dict_stats['MFGS'] = np.array([np.mean(gs_map)])                                                 
            else:
                 ideal_psf = lp.sci.fake_src_dict[str(int(lp.src.wavelength*1e9))]
                 
                 frame = np.squeeze(np.asarray(data, dtype=float))
                 if np.sum(frame) == 0:
                     i_peak_norm = 0
                 else:
                     i_peak_norm = peak_subpixel(frame) / np.sum(frame)
                 ideal = np.squeeze(np.asarray(ideal_psf, dtype=float))
                 i_peak_ideal_norm = peak_subpixel(ideal) / np.sum(ideal)

                 dict_stats['strehl'] = np.array([i_peak_norm / i_peak_ideal_norm ])

        return dict_stats
    
    def append_to_dataset(self, data_type, group, iteration, data, data_object=None):
        """
        Append data to an existing dataset in the HDF5 file.

        Parameters
        ----------
        data_type : str
            Type of data ('phase', 'opd', 'slopes_1D', 'slopes_2D', 'error', 'sci_frame').
        group : h5py.Group
            The group in which to append the dataset.
        iteration : int
            Current iteration number.
        data : numpy.ndarray
            Data to be appended to the dataset.
        data_object : Object
            Object containing simulation modules, required to get the ideal PSF
            and compute the Strehl ratio, or to get the valid actuators.
        """
        if np.ndim(data) == 2:
            data = data[None, ...]

        # Pupil mask to compute statistics --> we force the data to be 3D to stadarize night and solar cases
        if data_type == 'atmosphere_layered':
            mask = None
        elif data_type == 'dm_layer':
            mask = data_object.validAct[None, ...]
        else:
            mask = data_object.tel.pupil[None, ...]
        
        lp_index = self._get_lp_index_from_group(group)
        only_metrics_resolved = bool(self._resolve_setting(self.only_metrics, data_object, lp_index))

        new_data = self.compute_stats(data, mask, data_type, lp=data_object)
        # Add the keys for the data and the iteration, apart from the statistics
        if not only_metrics_resolved:
            new_data['data'] = data
        new_data['iteration'] = np.array([iteration])

        for stat_name, value in new_data.items():
            if value is not None:
                dset = group[stat_name]
                current_size = dset.shape[0]
                dset.resize((current_size + 1,) + dset.shape[1:])
                dset[current_size] = value
    
    def save(self, data_object, iteration):
        """
        Save the light paths data to an HDF5 file.

        Parameters
        ----------
        data_object : list
            List of objects containing simulation to be saved.
        iteration : int
            Current iteration number, shall start from 0.
        """

        if not isinstance(data_object, list):
            raise TypeError("data_object must be a list of LightPath objects.")
        
        tag = data_object[0].tag

        with h5py.File(self.file_path, 'a') as f:
            self._write_metadata(f, data_object, iteration)
            # First, check atmosphere and DM flags
            if tag == 'atmosphere' and self.atm:
                for i in range(len(data_object)):
                    atm_setting = self._resolve_setting(self.atm, data_object[i], i)
                    if atm_setting:
                        group_name = 'Atmosphere_' + str(i)
                        if group_name not in f:
                            self.initialize_hdf5_file(f, group_name, data_object[i], iteration)
                        else:
                            group = f[group_name]
                            for j in range(data_object[i].nLayer):
                                layer_name = 'layer_' + str(j+1)
                                grp = group[layer_name]
                                self.append_to_dataset('atmosphere_layered', grp, iteration, getattr(data_object[i], layer_name).screen.scrn, None)                        
            if tag == 'deformableMirror' and self.dm:
                for i in range(len(data_object)):
                    dm_setting = self._resolve_setting(self.dm, data_object[i], i)
                    if dm_setting:
                        group_name = 'DeformableMirror_' + str(i)
                        if group_name not in f:
                            self.initialize_hdf5_file(f, group_name, data_object[i], iteration)
                        else:
                            grp = f[group_name]
                            self.append_to_dataset('dm_layer', grp, iteration, data_object[i].dm_layer.cmd_1D, data_object[i])                        

            if tag == 'lightpath':
                light_path = data_object
                    
                for index, lp in enumerate(light_path):
                    group_name = f'LightPath_{index}'
                    if group_name not in f:
                        self.initialize_hdf5_file(f, group_name, lp, iteration)
                    else:
                        group = f[f'LightPath_{index}']

                        # Resolve settings for this specific light path
                        atm_per_dir_setting = self._resolve_setting(self.atm_per_dir, lp, index)
                        dm_per_dir_setting = self._resolve_setting(self.dm_per_dir, lp, index)
                        slopes_setting = self._resolve_setting(self.slopes, lp, index)
                        error_setting = self._resolve_setting(self.error, lp, index)
                        wfs_setting = self._resolve_setting(self.wfs, lp, index)
                        sci_setting = self._resolve_setting(self.sci, lp, index)
                        wfs_frame_setting = self._resolve_setting(self.wfs_frame, lp, index)
                        sci_frame_setting = self._resolve_setting(self.sci_frame, lp, index)

                        if isinstance(sci_frame_setting, dict):
                            short_rate = sci_frame_setting.get('short', 0)
                            long_rate = sci_frame_setting.get('long', 0)
                        else:
                            short_rate = sci_frame_setting
                            long_rate = sci_frame_setting

                        # Atmosphere OPD
                        if atm_per_dir_setting and ((iteration+1)%atm_per_dir_setting) == 0:
                            grp = group['atm_opd']
                            self.append_to_dataset('opd', grp, iteration, lp.atmosphere_opd, lp)
                            grp = group['atm_phase']
                            self.append_to_dataset('phase', grp, iteration, lp.atmosphere_phase, lp)

                        # DM
                        if dm_per_dir_setting and ((iteration+1)%dm_per_dir_setting) == 0:
                            for i in range(len(lp.dm)):
                                grp = group['dm_opd_' + str(i)]
                                self.append_to_dataset('opd', grp, iteration, lp.dm_opd[i], lp)
                                grp = group['dm_phase_' + str(i)]
                                self.append_to_dataset('phase', grp, iteration, lp.dm_phase[i], lp)

                        # Slopes
                        if slopes_setting and ((iteration+1)%slopes_setting) == 0 and (lp.wfs is not None):
                            grp = group['slopes_1D']
                            self.append_to_dataset('slopes_1D', grp, iteration, lp.slopes_1D, lp)
                            grp = group['slopes_2D']
                            self.append_to_dataset('slopes_2D', grp, iteration, lp.slopes_2D, lp)

                        # Error
                        if error_setting and ((iteration+1)%error_setting) == 0 and (lp.wfs is not None):
                            grp = group['error']
                            self.append_to_dataset('slopes_1D', grp, iteration, lp.get_wavefront_error(), lp)

                        # WFS
                        if wfs_setting and ((iteration+1)%wfs_setting) == 0 and (lp.wfs is not None):
                            grp = group['wfs_opd']
                            self.append_to_dataset('opd', grp, iteration, lp.wfs_opd, lp)
                            grp = group['wfs_phase']
                            self.append_to_dataset('phase', grp, iteration, lp.wfs_phase, lp)

                        # Science
                        if sci_setting and ((iteration+1)%sci_setting) == 0 and (lp.sci is not None):
                            grp = group['sci_opd']
                            self.append_to_dataset('opd', grp, iteration, lp.sci_opd, lp)
                            grp = group['sci_phase']
                            self.append_to_dataset('phase', grp, iteration, lp.sci_phase, lp)

                        # WFS Frame
                        if wfs_frame_setting and ((iteration+1)%wfs_frame_setting) == 0 and (lp.wfs is not None):
                            grp = group['wfs_frame']
                            self.append_to_dataset('wfs_frame', grp, iteration, lp.wfs_frame, lp)

                        # Science Frame
                        if (short_rate or long_rate) and (lp.sci is not None):
                            if short_rate and ((iteration+1)%short_rate) == 0 and lp.sci_frame is not None:
                                if 'sci_frame_shortExp' in group.keys():
                                    grp = group['sci_frame_shortExp']
                                    self.append_to_dataset('sci_frame', grp, iteration, lp.sci_frame, lp)
                                else:
                                    sci_frame_grp = group.create_group('sci_frame_shortExp')
                                    self.custom_create_dataset('sci_frame', sci_frame_grp, iteration, lp.sci_frame, None, lp)
                            if long_rate and ((iteration+1)%long_rate) == 0 and lp.long_exposure_frame is not None:
                                if 'sci_frame_longExp' in group.keys():
                                    grp = group['sci_frame_longExp']
                                    self.append_to_dataset('sci_frame', grp, iteration, lp.long_exposure_frame, lp)
                                else:
                                    sci_frame_grp = group.create_group('sci_frame_longExp')
                                    self.custom_create_dataset('sci_frame', sci_frame_grp, iteration, lp.long_exposure_frame, None, lp)                                

    
