"""
Created on March 24 2025
@author: nrodlin
"""
import time
import zmq
import pickle

import logging
import logging.handlers
from queue import Queue

"""
Sharepoint Module
=================

This module contains the `Sharepoint` class, used for to share the data of adaptive optics simulations.
"""

class Sharepoint:
    def __init__(self, logger=None, 
                 port=5555, 
                 ip="localhost", 
                 protocol="tcp", 
                 atm=0, atm_per_dir=0, 
                 dm=0, dm_per_dir=0, 
                 slopes=0, wfs=0, wfs_frame=0, 
                 sci=0, sci_frame=0):
        """
        Initialize the Sharepoint publisher for sharing light path data.

        Parameters
        ----------
        logger : logging.Logger, optional
            External logger to use. If None, initializes internal logging.
        port : int, optional
            Port number for ZeroMQ publisher. Default is 5555.
        ip : str, optional
            IP address to bind the publisher. Default is localhost.
        protocol : str, optional
            Communication protocol (e.g., 'tcp').
        atm : int
            Flag to share atmosphere phase, per layer. If 0, the atmosphere phase will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        atm_per_dir : int
            Flag to share the atmosphere phase projection per direction. If 0, it will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        dm : int
            Flag to share deformable mirror phase. If 0, the DM phase will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        dm_per_dir : int
            Flag to share the DM phase projection per direction. If 0, it will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        slopes : int
            Flag to share slopes data. If 0, the slopes data will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        wfs : int
            Flag to share wavefront sensor phase. If 0, the WFS data will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        wfs_frame : int
            Flag to share WFS frame. If 0, the WFS frame will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        sci : int
            Flag to share science phase. If 0, the science phase will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.
        sci_frame : int
            Flag to share science frame. If 0, the science frame will not be shared. 
            Larger than 0, it will be shared with the specified decimation factor.            
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger
        
        # Decimation for the data buffers
        self.atm = atm
        self.atm_per_dir = atm_per_dir
        self.dm = dm
        self.dm_per_dir = dm_per_dir
        self.slopes = slopes
        self.wfs = wfs
        self.wfs_frame = wfs_frame
        self.sci = sci
        self.sci_frame = sci_frame

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(protocol + '://' + ip + ':' + str(port))
    
    def shareData(self, light_path, iteration, atm_list=[], dm_list=[]):
        """
        Publish all configured attributes of the light path over ZeroMQ.

        Parameters
        ----------
        light_path : list
            List of LightPath objects containing simulation results.
        dm_list : list, optional
            List of Deformable Mirror objects to share data from. Default is an empty list.
        atm_list : list, optional
            List of Atmosphere objects to share data from. Default is an empty list.
        Returns
        -------
        bool
            True if data was sent successfully.
        """
        self.logger.debug('Sharepoint::shareData')
        # Prepare the list of topics 
        topics = []

        # First, share the global objects: atmosphere layers and deformable mirrors, if requested
        if self.atm > 0 and len(atm_list) > 0:
            if ((iteration+1)%self.atm) == 0:
                for i in range(len(atm_list)):
                    atm_name = 'atmosphere_' + str(i+1)
                    for j in range(atm_list[i].nLayer):
                        layer_name = 'layer_' + str(j+1)
                        topics.append(atm_name + '/' + layer_name)
                        self.socket.send_multipart([topics[-1].encode(), pickle.dumps(getattr(atm_list[i], layer_name).phase)])
        if self.dm > 0 and len(dm_list) > 0:
            if ((iteration+1)%self.dm) == 0:
                for i in range(len(dm_list)):
                    dm_name = 'dm_' + str(i+1)
                    topics.append(dm_name + '/2D_command')
                    self.socket.send_multipart([topics[-1].encode(), pickle.dumps(dm_list[i].dm_layer.cmd_2D)])
                    topics.append(dm_name + '/1D_command')
                    self.socket.send_multipart([topics[-1].encode(), pickle.dumps(dm_list[i].dm_layer.cmd_2D[dm_list[i].validAct_2D])])

        # Check Light Path dimensions:
        nNameSpaces = len(light_path)

        t0 = time.time()
        
        for i in range(nNameSpaces):
            topic_name = 'lightPath' + str(i) + '/'

            if self.atm_per_dir > 0 and ((iteration+1) % self.atm_per_dir) == 0:
                topics.append(topic_name + 'atmosphere_opd')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].atmosphere_opd)])
                topics.append(topic_name + 'atmosphere_phase')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].atmosphere_phase)])

            if self.dm_per_dir > 0 and ((iteration+1) % self.dm_per_dir) == 0:
                for j in range(len(light_path[i].dm_opd)):
                    topics.append(topic_name + 'dm_opd_' + str(j+1))
                    self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].dm_opd[j])])
                    topics.append(topic_name + 'dm_phase_' + str(j+1))
                    self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].dm_opd[j])])
            
            if self.slopes > 0 and ((iteration+1) % self.slopes) == 0:
                topics.append(topic_name + 'slopes_1D')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].slopes_1D)])
                topics.append(topic_name + 'slopes_2D')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].slopes_2D)])
            
            if self.wfs > 0 and ((iteration+1) % self.wfs) == 0:
                topics.append(topic_name + 'wfs_opd')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].wfs_opd)])
                topics.append(topic_name + 'wfs_phase')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].wfs_phase)])

            if self.sci > 0 and ((iteration+1) % self.sci) == 0:
                topics.append(topic_name + 'sci_opd')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].sci_opd)])
                topics.append(topic_name + 'sci_phase')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].sci_phase)])
            
            if self.sci_frame > 0 and ((iteration+1) % self.sci_frame) == 0:
                topics.append(topic_name + 'sci_frame')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].sci_frame)])

            if self.wfs_frame > 0 and ((iteration+1) % self.wfs_frame) == 0:    
                topics.append(topic_name + 'wfs_frame')
                self.socket.send_multipart([topics[-1].encode(), pickle.dumps(light_path[i].wfs_frame)])
        
        self.socket.send_multipart([b"topics", ",".join(topics).encode()])
        
        self.logger.debug(f'Sharepoint::shareData - Sending took {time.time()-t0} [s]')                

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
    
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()
        if self.socket is not None and self.context is not None:
            self.socket.close()
            self.context.term()