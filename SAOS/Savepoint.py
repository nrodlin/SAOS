"""
Created on May 15 2025
@author: nrodlin
"""
import numpy as np
import h5py
import os
from datetime import datetime

import logging
import logging.handlers
from queue import Queue

"""
Savepoint Module
=================

This module contains the `Savepoint` class, used for to save the data of adaptive optics simulations.
"""

class Savepoint:
    def __init__(self, file_path=None, atm=0, dm=0, slopes=0, wfs=0, wfs_frame=0, sci=0, sci_frame=0, logger=None):
        """
        Initialize the Savepoint object for saving simulation data.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file where data will be saved. By default, creates a folder in the home directory.
        atm : int
            Flag to save atmosphere phase. If 0, the atmosphere phase will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        dm : int
            Flag to save deformable mirror phase. If 0, the DM phase will not be saved. 
            Larger than 0, it will be saved with the specified decimation factor.
        slopes : int
            Flag to save slopes data. If 0, the slopes data will not be saved. 
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
        logger : logging.Logger, optional
            External logger to use. If None, initializes internal logging.        
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger
        
        self.file_path = file_path
        self.atm = atm
        self.dm = dm
        self.slopes = slopes
        self.wfs = wfs
        self.wfs_frame = wfs_frame
        self.sci = sci
        self.sci_frame = sci_frame

        # Create the file path if not provided
        if not self.file_path:
            simulations_dir = os.path.expanduser('~/simulations')
            os.makedirs(simulations_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.file_path = os.path.join(simulations_dir, 'results', f'saos_savepoint_{timestamp}.h5')
        
        self.logger.info(f'Savepoint::__init__ - File saved in {self.file_path}')


    def initialize_hdf5_file(self, f, lp, index, iteration):
        """
        Initialize the HDF5 file with the structure for saving light path data.

        Parameters
        ----------
        f : h5py.File
            Opened HDF5 file object.
        lp : LightPath
            LightPath object containing simulation results.
        index : int
            Index of the light path in the list of light paths.
        iteration : int
            Current iteration number.
        """
        group = f.create_group(f'LightPath_{index}')

        if self.atm:
            # Pupil mask to compute statistics
            mask = lp.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            atm_opd_grp = group.create_group('atm_opd')
            self.custom_create_dataset('opd', atm_opd_grp, iteration, lp.atmosphere_opd, mask)
            atm_phase_grp = group.create_group('atm_phase')
            self.custom_create_dataset('phase', atm_phase_grp, iteration, lp.atmosphere_opd, mask)
        
        if self.dm:
            for i in range(len(lp.dm)):
                # Pupil mask to compute statistics --> we force the data to be 3D to stadarize night and solar cases
                mask = lp.tel.pupil[None, ...]
                # Create group and add the datasets with the statistics
                dm_opd_grp = group.create_group('dm_opd_' + str(i))
                self.custom_create_dataset('opd', dm_opd_grp, iteration, lp.dm_opd[i], mask)
                dm_phase_grp = group.create_group('dm_phase_' + str(i))
                self.custom_create_dataset('phase', dm_phase_grp, iteration, lp.dm_phase[i], mask)

        if self.slopes:
            # Create group and add the datasets with the statistics ( mask is not needed)
            slopes1d_grp = group.create_group('slopes_1D')
            self.custom_create_dataset('slopes_1D', slopes1d_grp, iteration, lp.slopes_1D, mask=None)
            slopes2d_grp = group.create_group('slopes_2D')
            self.custom_create_dataset('slopes_2D', slopes2d_grp, iteration, lp.slopes_2D, mask=None)

        if self.wfs:
            # Pupil mask to compute statistics
            mask = lp.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            wfs_opd_grp = group.create_group('wfs_opd')
            self.custom_create_dataset('opd', wfs_opd_grp, iteration, lp.wfs_opd, mask)
            wfs_phase_grp = group.create_group('wfs_phase')
            self.custom_create_dataset('phase', wfs_phase_grp, iteration, lp.wfs_opd, mask)

        if self.sci:
            # Pupil mask to compute statistics
            mask = lp.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            sci_opd_grp = group.create_group('sci_opd')
            self.custom_create_dataset('opd', sci_opd_grp, iteration, lp.sci_opd, mask)
            sci_phase_grp = group.create_group('sci_phase')
            self.custom_create_dataset('phase', sci_phase_grp, iteration, lp.sci_opd, mask)

        if self.wfs_frame:
            # Create group and add the datasets with the statistics ( mask is not needed)
            wfs_frame_grp = group.create_group('wfs_frame')
            self.custom_create_dataset('wfs_frame', wfs_frame_grp, iteration, lp.wfs_frame, mask=None)

        if self.sci_frame:
            # Create group and add the datasets with the statistics ( mask is not needed)
            sci_frame_grp = group.create_group('sci_frame')
            self.custom_create_dataset('sci_frame', sci_frame_grp, iteration, lp.sci_frame, None, lp)

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
            'strehl': None,
        }

        if data_type == 'phase' or data_type == 'opd':
            dict_stats['min'] = np.array([np.min(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['max'] = np.array([np.max(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['mean'] = np.array([np.mean(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['std'] = np.array([np.std(data.reshape(data.shape[0], -1)[:, mask.ravel()], axis=1)])
            dict_stats['rms'] = np.array([np.sqrt(np.mean(data.reshape(data.shape[0], -1)[:, mask.ravel()]**2, axis=1))])
        elif data_type == 'slopes_1D':
            dict_stats['min'] = np.array([[np.min(data[0:data.shape[0]//2]), np.min(data[data.shape[0]//2:])]])
            dict_stats['max'] = np.array([[np.max(data[0:data.shape[0]//2]), np.max(data[data.shape[0]//2:])]])
            dict_stats['mean'] = np.array([[np.mean(data[0:data.shape[0]//2]), np.mean(data[data.shape[0]//2:])]])
            dict_stats['std'] = np.array([[np.std(data[0:data.shape[0]//2]), np.std(data[data.shape[0]//2:])]])
            dict_stats['rms'] = np.array([[np.sqrt(np.mean(data[0:data.shape[0]//2]**2)), np.sqrt(np.mean(data[data.shape[0]//2:]**2))]])

        elif data_type == 'slopes_2D':
            dict_stats['min'] = np.array([[np.min(data[0:data.shape[1]//2,:]), np.min(data[0,data.shape[1]//2:,:])]])
            dict_stats['max'] = np.array([[np.max(data[0:data.shape[1]//2,:]), np.max(data[0,data.shape[1]//2:,:])]])
        
        elif data_type == 'sci_frame' and (lp is not None):
            if lp.src.tag == 'sun':
                mean_intensity = np.mean(data)
                std_intensity = np.std(data)
                if mean_intensity != 0:
                    dict_stats['contrast'] = np.array([std_intensity / mean_intensity])
                else:
                    dict_stats['contrast'] = np.array([0.0])
            else:
                 i_peak_norm = np.max(data) / np.sum(data)
                 i_peak_ideal_norm = np.max(lp.sci.ideal_psf) / np.sum(lp.sci.ideal_psf)

                 dict_stats['strehl'] = np.array([i_peak_norm / i_peak_ideal_norm ])

        return dict_stats
    
    def append_to_dataset(self, data_type, group, iteration, data, lp):
        """
        Append data to an existing dataset in the HDF5 file.

        Parameters
        ----------
        data_type : str
            Type of data ('phase', 'opd', 'slopes_1D', 'slopes_2D', 'sci_frame').
        group : h5py.Group
            The group in which to append the dataset.
        iteration : int
            Current iteration number.
        data : numpy.ndarray
            Data to be appended to the dataset.
        lp : LightPath
            LightPath object containing simulation modules, required to get the ideal PSF
            and compute the Strehl ratio.
        """
        if np.ndim(data) == 2:
            data = data[None, ...]

        # Pupil mask to compute statistics --> we force the data to be 3D to stadarize night and solar cases
        mask = lp.tel.pupil[None, ...]
        
        new_data = self.compute_stats(data, mask, data_type, lp=lp)
        # Add the keys for the data and the iteration, aprt from the stadistics
        new_data['data'] = data
        new_data['iteration'] = np.array([iteration])

        for stat_name, value in new_data.items():
            if value is not None:
                dset = group[stat_name]
                current_size = dset.shape[0]
                dset.resize((current_size + 1,) + dset.shape[1:])
                dset[current_size] = value
    
    def save(self, light_path, iteration):
        """
        Save the light paths data to an HDF5 file.

        Parameters
        ----------
        light_path : list
            List of LightPath objects containing simulation results.
        iteration : int
            Current iteration number, shall start from 0.
        """
        with h5py.File(self.file_path, 'a') as f:
            for index, lp in enumerate(light_path):
                group_name = f'LightPath_{index}'
                if group_name not in f:
                    self.initialize_hdf5_file(f, lp, index, iteration)
                else:
                    group = f[f'LightPath_{index}']

                    # Atmosphere OPD
                    if self.atm and ((iteration+1)%self.atm) == 0:
                        grp = group['atm_opd']
                        self.append_to_dataset('opd', grp, iteration, lp.atmosphere_opd, lp)
                        grp = group['atm_phase']
                        self.append_to_dataset('phase', grp, iteration, lp.atmosphere_phase, lp)

                    # DM
                    if self.dm and ((iteration+1)%self.dm) == 0:
                        for i in range(len(lp.dm)):
                            grp = group['dm_opd_' + str(i)]
                            self.append_to_dataset('opd', grp, iteration, lp.dm_opd[i], lp)
                            grp = group['dm_phase_' + str(i)]
                            self.append_to_dataset('phase', grp, iteration, lp.dm_phase[i], lp)

                    # Slopes
                    if self.slopes and ((iteration+1)%self.slopes) == 0:
                        grp = group['slopes_1D']
                        self.append_to_dataset('slopes_1D', grp, iteration, lp.slopes_1D, lp)
                        grp = group['slopes_2D']
                        self.append_to_dataset('slopes_2D', grp, iteration, lp.slopes_2D, lp)

                    # WFS
                    if self.wfs and ((iteration+1)%self.wfs) == 0:
                        grp = group['wfs_opd']
                        self.append_to_dataset('opd', grp, iteration, lp.wfs_opd, lp)
                        grp = group['wfs_phase']
                        self.append_to_dataset('phase', grp, iteration, lp.wfs_phase, lp)

                    # Science
                    if self.sci and ((iteration+1)%self.sci) == 0:
                        grp = group['sci_opd']
                        self.append_to_dataset('opd', grp, iteration, lp.sci_opd, lp)
                        grp = group['sci_phase']
                        self.append_to_dataset('phase', grp, iteration, lp.sci_phase, lp)

                    # WFS Frame
                    if self.wfs_frame and ((iteration+1)%self.wfs_frame) == 0:
                        grp = group['wfs_frame']
                        self.append_to_dataset('wfs_frame', grp, iteration, lp.wfs_frame, lp)

                    # Science Frame
                    if self.sci_frame and ((iteration+1)%self.sci_frame) == 0:
                        grp = group['sci_frame']
                        self.append_to_dataset('sci_frame', grp, iteration, lp.sci_frame, lp)

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
