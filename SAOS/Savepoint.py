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
        """
        group = f.create_group(f'LightPath_{index}')

        if self.atm:
            atm_opd_grp = group.create_group('atmosphere_opd')
            atm_opd_grp.create_dataset('data', data=lp.atmosphere_opd[None, ...], maxshape=(None,) + lp.atmosphere_opd.shape, chunks=True)
            atm_opd_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            atm_opd_grp.create_dataset('min', data=np.array([np.min(lp.atmosphere_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_opd_grp.create_dataset('max', data=np.array([np.max(lp.atmosphere_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_opd_grp.create_dataset('mean', data=np.array([np.mean(lp.atmosphere_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_opd_grp.create_dataset('std', data=np.array([np.std(lp.atmosphere_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_opd_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.atmosphere_opd**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

            atm_phase_grp = group.create_group('atmosphere_phase')
            atm_phase_grp.create_dataset('data', data=lp.atmosphere_phase[None, ...], maxshape=(None,) + lp.atmosphere_phase.shape, chunks=True)
            atm_phase_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            atm_phase_grp.create_dataset('min', data=np.array([np.min(lp.atmosphere_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_phase_grp.create_dataset('max', data=np.array([np.max(lp.atmosphere_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_phase_grp.create_dataset('mean', data=np.array([np.mean(lp.atmosphere_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_phase_grp.create_dataset('std', data=np.array([np.std(lp.atmosphere_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            atm_phase_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.atmosphere_phase**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

        if self.dm:
            for i in range(len(lp.dm)):
                dm_opd_grp = group.create_group('dm_opd_' + str(i))
                dm_opd_grp.create_dataset('data', data=lp.dm_opd[i][None, ...], maxshape=(None,) + lp.dm_opd[i].shape, chunks=True)
                dm_opd_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
                dm_opd_grp.create_dataset('min', data=np.array([np.min(lp.dm_opd[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_opd_grp.create_dataset('max', data=np.array([np.max(lp.dm_opd[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_opd_grp.create_dataset('mean', data=np.array([np.mean(lp.dm_opd[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_opd_grp.create_dataset('std', data=np.array([np.std(lp.dm_opd[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_opd_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.dm_opd[i]**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

                dm_phase_grp = group.create_group('dm_phase_' + str(i))
                dm_phase_grp.create_dataset('data', data=lp.dm_phase[i][None, ...], maxshape=(None,) + lp.dm_phase[i].shape, chunks=True)
                dm_phase_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
                dm_phase_grp.create_dataset('min', data=np.array([np.min(lp.dm_phase[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_phase_grp.create_dataset('max', data=np.array([np.max(lp.dm_phase[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_phase_grp.create_dataset('mean', data=np.array([np.mean(lp.dm_phase[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_phase_grp.create_dataset('std', data=np.array([np.std(lp.dm_phase[i], axis=(-2,-1))]), maxshape=(None,), chunks=True)
                dm_phase_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.dm_phase[i]**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

        if self.slopes:
            slopes1d_grp = group.create_group('slopes_1D')
            slopes1d_grp.create_dataset('data', data=lp.slopes_1D[None, ...], maxshape=(None,) + lp.slopes_1D.shape, chunks=True)
            slopes1d_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            slopes1d_grp.create_dataset('min', data=np.array([[np.min(lp.slopes_1D[0:lp.slopes_1D.shape[0]//2]), np.min(lp.slopes_1D[lp.slopes_1D.shape[0]//2:])]]), maxshape=(None, 2), chunks=True)
            slopes1d_grp.create_dataset('max', data=np.array([[np.max(lp.slopes_1D[0:lp.slopes_1D.shape[0]//2]), np.max(lp.slopes_1D[lp.slopes_1D.shape[0]//2:])]]), maxshape=(None, 2), chunks=True)
            slopes1d_grp.create_dataset('mean', data=np.array([[np.mean(lp.slopes_1D[0:lp.slopes_1D.shape[0]//2]), np.mean(lp.slopes_1D[lp.slopes_1D.shape[0]//2:])]]), maxshape=(None, 2), chunks=True)
            slopes1d_grp.create_dataset('std', data=np.array([[np.std(lp.slopes_1D[0:lp.slopes_1D.shape[0]//2]), np.std(lp.slopes_1D[lp.slopes_1D.shape[0]//2:])]]), maxshape=(None, 2), chunks=True)
            slopes1d_grp.create_dataset('rms', data=np.array([[np.sqrt(np.mean(lp.slopes_1D[0:lp.slopes_1D.shape[0]//2]**2)), np.sqrt(np.mean(lp.slopes_1D[lp.slopes_1D.shape[0]//2:]**2))]]), maxshape=(None, 2), chunks=True)

            slopes2d_grp = group.create_group('slopes_2D')
            slopes2d_grp.create_dataset('data', data=lp.slopes_2D[None, ...], maxshape=(None,) + lp.slopes_2D.shape, chunks=True)
            slopes2d_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            slopes2d_grp.create_dataset('min', data=np.array([[np.min(lp.slopes_2D[0:lp.slopes_2D.shape[0]//2,:]), np.min(lp.slopes_2D[lp.slopes_2D.shape[0]//2:,:])]]), maxshape=(None, 2), chunks=True)
            slopes2d_grp.create_dataset('max', data=np.array([[np.max(lp.slopes_2D[0:lp.slopes_2D.shape[0]//2,:]), np.max(lp.slopes_2D[lp.slopes_2D.shape[0]//2:,:])]]), maxshape=(None, 2), chunks=True)

        if self.wfs:
            wfs_opd_grp = group.create_group('wfs_opd')
            wfs_opd_grp.create_dataset('data', data=lp.wfs_opd[None, ...], maxshape=(None,) + lp.wfs_opd.shape, chunks=True)
            wfs_opd_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            wfs_opd_grp.create_dataset('min', data=np.array([np.min(lp.wfs_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_opd_grp.create_dataset('max', data=np.array([np.max(lp.wfs_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_opd_grp.create_dataset('mean', data=np.array([np.mean(lp.wfs_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_opd_grp.create_dataset('std', data=np.array([np.std(lp.wfs_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_opd_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.wfs_opd**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

            wfs_phase_grp = group.create_group('wfs_phase')
            wfs_phase_grp.create_dataset('data', data=lp.wfs_phase[None, ...], maxshape=(None,) + lp.wfs_phase.shape, chunks=True)
            wfs_phase_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            wfs_phase_grp.create_dataset('min', data=np.array([np.min(lp.wfs_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_phase_grp.create_dataset('max', data=np.array([np.max(lp.wfs_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_phase_grp.create_dataset('mean', data=np.array([np.mean(lp.wfs_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_phase_grp.create_dataset('std', data=np.array([np.std(lp.wfs_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            wfs_phase_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.wfs_phase**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

        if self.sci:
            sci_opd_grp = group.create_group('sci_opd')
            sci_opd_grp.create_dataset('data', data=lp.sci_opd[None, ...], maxshape=(None,) + lp.sci_opd.shape, chunks=True)
            sci_opd_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            sci_opd_grp.create_dataset('min', data=np.array([np.min(lp.sci_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_opd_grp.create_dataset('max', data=np.array([np.max(lp.sci_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_opd_grp.create_dataset('mean', data=np.array([np.mean(lp.sci_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_opd_grp.create_dataset('std', data=np.array([np.std(lp.sci_opd, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_opd_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.sci_opd**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

            sci_phase_grp = group.create_group('sci_phase')
            sci_phase_grp.create_dataset('data', data=lp.sci_phase[None, ...], maxshape=(None,) + lp.sci_phase.shape, chunks=True)
            sci_phase_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)
            sci_phase_grp.create_dataset('min', data=np.array([np.min(lp.sci_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_phase_grp.create_dataset('max', data=np.array([np.max(lp.sci_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_phase_grp.create_dataset('mean', data=np.array([np.mean(lp.sci_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_phase_grp.create_dataset('std', data=np.array([np.std(lp.sci_phase, axis=(-2,-1))]), maxshape=(None,), chunks=True)
            sci_phase_grp.create_dataset('rms', data=np.array([np.sqrt(np.mean(lp.sci_phase**2, axis=(-2,-1)))]), maxshape=(None,), chunks=True)

        if self.wfs_frame:
            wfs_frame_grp = group.create_group('wfs_frame')
            wfs_frame_grp.create_dataset('data', data=lp.wfs_frame[None, ...], maxshape=(None,) + lp.wfs_frame.shape, chunks=True)
            wfs_frame_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)

        if self.sci_frame:
            sci_frame_grp = group.create_group('sci_frame')
            sci_frame_grp.create_dataset('data', data=lp.sci_frame[None, ...], maxshape=(None,) + lp.sci_frame.shape, chunks=True)
            sci_frame_grp.create_dataset('iteration', data=np.array([iteration]), maxshape=(None,), chunks=True)

    
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
                        grp = group['atmosphere_opd']
                        self.append_to_dataset(grp, 'data', lp.atmosphere_opd)
                        self.append_stat(grp, 'iteration', iteration)
                        stats = self.compute_stats(lp.atmosphere_opd)
                        for stat_name, value in stats.items():
                            self.append_stat(grp, stat_name, value)

                        # Atmosphere Phase
                        grp = group['atmosphere_phase']
                        self.append_to_dataset(grp, 'data', lp.atmosphere_phase)
                        self.append_stat(grp, 'iteration', iteration)
                        stats = self.compute_stats(lp.atmosphere_phase)
                        for stat_name, value in stats.items():
                            self.append_stat(grp, stat_name, value)

                    # DM
                    if self.dm and ((iteration+1)%self.dm) == 0:
                        for i in range(len(lp.dm)):
                            for suffix, array in [('dm_opd_' + str(i), lp.dm_opd), ('dm_phase_' + str(i), lp.dm_phase)]:
                                grp = group[suffix]
                                self.append_to_dataset(grp, 'data', array[i])
                                self.append_stat(grp, 'iteration', iteration)
                                stats = self.compute_stats(array[i])
                                for stat_name, value in stats.items():
                                    self.append_stat(grp, stat_name, value)

                    # Slopes
                    if self.slopes and ((iteration+1)%self.slopes) == 0:
                        for slopes_type, array in [('slopes_1D', lp.slopes_1D), ('slopes_2D', lp.slopes_2D)]:
                            grp = group[slopes_type]
                            self.append_to_dataset(grp, 'data', array)
                            self.append_stat(grp, 'iteration', iteration)
                            if slopes_type == 'slopes_1D':
                                min_val = np.array([np.min(array[:array.shape[0]//2]), np.min(array[array.shape[0]//2:])])
                                max_val = np.array([np.max(array[:array.shape[0]//2]), np.max(array[array.shape[0]//2:])])
                                mean_val = np.array([np.mean(array[:array.shape[0]//2]), np.mean(array[array.shape[0]//2:])])
                                std_val = np.array([np.std(array[:array.shape[0]//2]), np.std(array[array.shape[0]//2:])])
                                rms_val = np.array([np.sqrt(np.mean(array[:array.shape[0]//2]**2)),
                                                    np.sqrt(np.mean(array[array.shape[0]//2:]**2))])

                                self.append_stat(grp, 'min', min_val)
                                self.append_stat(grp, 'max', max_val)
                                self.append_stat(grp, 'mean', mean_val)
                                self.append_stat(grp, 'std', std_val)
                                self.append_stat(grp, 'rms', rms_val)
                            else:  # slopes_2D
                                min_val = np.array([np.min(array[:array.shape[0]//2, :]), np.min(array[array.shape[0]//2:, :])])
                                max_val = np.array([np.max(array[:array.shape[0]//2, :]), np.max(array[array.shape[0]//2:, :])])

                                self.append_stat(grp, 'min', min_val)
                                self.append_stat(grp, 'max', max_val)

                    # WFS
                    if self.wfs and ((iteration+1)%self.wfs) == 0:
                        for suffix, array in [('wfs_opd', lp.wfs_opd), ('wfs_phase', lp.wfs_phase)]:
                            grp = group[suffix]
                            self.append_to_dataset(grp, 'data', array)
                            self.append_stat(grp, 'iteration', iteration)
                            stats = self.compute_stats(array)
                            for stat_name, value in stats.items():
                                self.append_stat(grp, stat_name, value)

                    # Science
                    if self.sci and ((iteration+1)%self.sci) == 0:
                        for suffix, array in [('sci_opd', lp.sci_opd), ('atmosphere_phase', lp.sci_phase)]:
                            grp = group[suffix]
                            self.append_to_dataset(grp, 'data', array)
                            self.append_stat(grp, 'iteration', iteration)
                            stats = self.compute_stats(array)
                            for stat_name, value in stats.items():
                                self.append_stat(grp, stat_name, value)

                    # WFS Frame
                    if self.wfs_frame and ((iteration+1)%self.wfs_frame) == 0:
                        grp = group['wfs_frame']
                        self.append_to_dataset(grp, 'data', lp.wfs_frame)
                        self.append_stat(grp, 'iteration', iteration)

                    # Science Frame
                    if self.sci_frame and ((iteration+1)%self.sci_frame) == 0:
                        grp = group['sci_frame']
                        self.append_to_dataset(grp, 'data', lp.sci_frame)
                        self.append_stat(grp, 'iteration', iteration)
    
    def append_to_dataset(self, group, name, data):
        dset = group['data']
        current_size = dset.shape[0]
        dset.resize((current_size + 1,) + dset.shape[1:])
        dset[current_size] = data

    def append_stat(self, group, stat_name, value):
        dset = group[stat_name]
        current_size = dset.shape[0]
        dset.resize((current_size + 1,) + dset.shape[1:])
        dset[current_size] = value

    def compute_stats(self, data):
        return {
            'min': np.min(data, axis=(-2, -1)),
            'max': np.max(data, axis=(-2, -1)),
            'mean': np.mean(data, axis=(-2, -1)),
            'std': np.std(data, axis=(-2, -1)),
            'rms': np.sqrt(np.mean(data**2, axis=(-2, -1)))
        }

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
