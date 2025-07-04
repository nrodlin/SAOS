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
    def __init__(self, file_path=None, atm=0, atm_per_dir=0, dm=0, dm_per_dir=0, slopes=0, wfs=0, wfs_frame=0, sci=0, sci_frame=0, logger=None):
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
        self.atm_per_dir = atm_per_dir
        self.dm = dm
        self.dm_per_dir = dm_per_dir
        self.slopes = slopes
        self.wfs = wfs
        self.wfs_frame = wfs_frame
        self.sci = sci
        self.sci_frame = sci_frame

        # Create the file path if not provided
        if not self.file_path:
            simulations_dir = os.path.expanduser('~/simulations/results')
            os.makedirs(simulations_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.file_path = os.path.join(simulations_dir, f'saos_savepoint_{timestamp}.h5')
        
        self.logger.info(f'Savepoint::__init__ - Saving path (not yet created): {self.file_path}')


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

        if self.atm and group_name.find('Atmosphere')>=0:
            for i in range(data_group.nLayer):
                layer_name = 'layer_' + str(i+1)
                atm_layer_grp = group.create_group(layer_name)
                self.custom_create_dataset('atmosphere_layered', atm_layer_grp, iteration, getattr(data_group, layer_name).screen.scrn, mask=None)
        
        if self.dm and group_name.find('DeformableMirror')>=0:
            self.custom_create_dataset('dm_layer', group, iteration, data_group.dm_layer.cmd_2D, mask=data_group.validAct_2D[None, ...])
                
        if self.atm_per_dir and group_name.find('LightPath')>=0:
            # Pupil mask to compute statistics
            mask = data_group.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            atm_opd_grp = group.create_group('atm_opd')
            self.custom_create_dataset('opd', atm_opd_grp, iteration, data_group.atmosphere_opd, mask)
            atm_phase_grp = group.create_group('atm_phase')
            self.custom_create_dataset('phase', atm_phase_grp, iteration, data_group.atmosphere_phase, mask)
        
        if self.dm_per_dir and group_name.find('LightPath')>=0:
            for i in range(len(data_group.dm)):
                # Pupil mask to compute statistics --> we force the data to be 3D to stadarize night and solar cases
                mask = data_group.tel.pupil[None, ...]
                # Create group and add the datasets with the statistics
                dm_opd_grp = group.create_group('dm_opd_' + str(i))
                self.custom_create_dataset('opd', dm_opd_grp, iteration, data_group.dm_opd[i], mask)
                dm_phase_grp = group.create_group('dm_phase_' + str(i))
                self.custom_create_dataset('phase', dm_phase_grp, iteration, data_group.dm_phase[i], mask)

        if self.slopes and group_name.find('LightPath')>=0:
            # Create group and add the datasets with the statistics ( mask is not needed)
            slopes1d_grp = group.create_group('slopes_1D')
            self.custom_create_dataset('slopes_1D', slopes1d_grp, iteration, data_group.slopes_1D, mask=None)
            slopes2d_grp = group.create_group('slopes_2D')
            self.custom_create_dataset('slopes_2D', slopes2d_grp, iteration, data_group.slopes_2D, mask=None)

        if self.wfs and group_name.find('LightPath')>=0:
            # Pupil mask to compute statistics
            mask = data_group.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            wfs_opd_grp = group.create_group('wfs_opd')
            self.custom_create_dataset('opd', wfs_opd_grp, iteration, data_group.wfs_opd, mask)
            wfs_phase_grp = group.create_group('wfs_phase')
            self.custom_create_dataset('phase', wfs_phase_grp, iteration, data_group.wfs_phase, mask)

        if self.sci and group_name.find('LightPath')>=0:
            # Pupil mask to compute statistics
            mask = data_group.tel.pupil[None, ...]
            # Create group and add the datasets with the statistics
            sci_opd_grp = group.create_group('sci_opd')
            self.custom_create_dataset('opd', sci_opd_grp, iteration, data_group.sci_opd, mask)
            sci_phase_grp = group.create_group('sci_phase')
            self.custom_create_dataset('phase', sci_phase_grp, iteration, data_group.sci_phase, mask)

        if self.wfs_frame and group_name.find('LightPath')>=0:
            # Create group and add the datasets with the statistics ( mask is not needed)
            wfs_frame_grp = group.create_group('wfs_frame')
            self.custom_create_dataset('wfs_frame', wfs_frame_grp, iteration, data_group.wfs_frame, mask=None)

        if self.sci_frame and group_name.find('LightPath')>=0:
            # Create group and add the datasets with the statistics ( mask is not needed)
            sci_frame_grp = group.create_group('sci_frame')
            self.custom_create_dataset('sci_frame', sci_frame_grp, iteration, data_group.sci_frame, None, data_group)

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
        elif data_type == 'atmosphere_layered':
            dict_stats['min'] = np.array([np.min(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['max'] = np.array([np.max(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['mean'] = np.array([np.mean(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['std'] = np.array([np.std(data.reshape(data.shape[0], -1), axis=1)])
            dict_stats['rms'] = np.array([np.sqrt(np.mean(data.reshape(data.shape[0], -1)**2, axis=1))])
        elif data_type == 'dm_layer':
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
    
    def append_to_dataset(self, data_type, group, iteration, data, data_object=None):
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
            mask = data_object.validAct_2D[None, ...]
        else:
            mask = data_object.tel.pupil[None, ...]
        
        new_data = self.compute_stats(data, mask, data_type, lp=data_object)
        # Add the keys for the data and the iteration, aprt from the stadistics
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
            # First, check atmosphere and DM flags
            if tag == 'atmosphere' and self.atm:
                for i in range(len(data_object)):
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
                    group_name = 'DeformableMirror_' + str(i)
                    if group_name not in f:
                        self.initialize_hdf5_file(f, group_name, data_object[i], iteration)
                    else:
                        grp = f[group_name]
                        self.append_to_dataset('dm_layer', grp, iteration, data_object[i].dm_layer.cmd_2D, data_object[i])                        

            if tag == 'lightpath':
                light_path = data_object
                    
                for index, lp in enumerate(light_path):
                    group_name = f'LightPath_{index}'
                    if group_name not in f:
                        self.initialize_hdf5_file(f, group_name, lp, iteration)
                    else:
                        group = f[f'LightPath_{index}']

                        # Atmosphere OPD
                        if self.atm_per_dir and ((iteration+1)%self.atm_per_dir) == 0:
                            grp = group['atm_opd']
                            self.append_to_dataset('opd', grp, iteration, lp.atmosphere_opd, lp)
                            grp = group['atm_phase']
                            self.append_to_dataset('phase', grp, iteration, lp.atmosphere_phase, lp)

                        # DM
                        if self.dm_per_dir and ((iteration+1)%self.dm_per_dir) == 0:
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
