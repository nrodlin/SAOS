import numpy as np
import time
import zmq

from joblib import Parallel, delayed

from OOPAO.LightPath import LightPath

import logging
import logging.handlers
from queue import Queue

class Sharepoint:
    def __init__(self, logger=None, port=5555, ip="localhost", protocol="tcp"):
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger
        # List of attributes to be published
        self.lp_attributes = ['atmosphere_opd', 'atmosphere_phase', 'dms_opd', 'dms_phase',
                              'wfs_opd', 'wfs_phase', 'ncpa_opd', 'ncpa_phase', 'sci_opd', 
                              'sci_phase', 'slopes_1D', 'slopes_2D', 'wfs_frame', 'sci_frame']
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(protocol + '://' + ip + ':' + str(port))
    
    def shareData(self, light_path):
        self.logger.debug('Sharepoint::shareData')

        # Check Light Path dimensions:
        nNameSpaces = len(light_path)

        # Prepare the list of topics 
        topics = []
        t0 = time.time()
        for i in range(nNameSpaces):
            topic_name = 'lightPath' + str(i) + '/'

            for j in range(len(self.lp_attributes)):
                
                buffer = getattr(light_path[i], self.lp_attributes[j])
                
                if buffer is not None:
                    topics.append(topic_name + self.lp_attributes[j])
                    self.socket.send_multipart([topics[-1].encode(), buffer.tobytes()])
        
        self.socket.send_multipart([b"topics", ",".join(topics).encode()])
        
        self.logger.info(f'Sharepoint::shareData - Sending took {time.time()-t0} [s]')                

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