"""
Created on July 10 2025
@author: nrodlin and adrianc
"""

import zmq
import pickle

import logging
from .LoggingHelper import get_logger

"""
ReceptionPoint Module
=================

This module contains the `ReceptionPoint` class, used to ask for specific buffer to an external server
"""

class ReceptionPoint:
    def __init__(self, logger=None, 
                 port=7001, 
                 ip="localhost", 
                 protocol="tcp", 
                 timeout=5000):
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
        timeout : int
            Timeout for the response to arrive in ms.       
        """
        self.logger = logger if logger is not None else get_logger()
            
        # Map arguments to attributes
        self.ip = ip
        self.port = port
        self.protocol = protocol
        self.timeout = timeout
        # Setup connecction
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

        self.socket.connect(self.protocol + '://' + self.ip + ':' + str(self.port))
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        self.actual_loop_state = None
	
    def sendRequest(self, buffer_type=''):
        """
        Send a request to a ZMQ server for an specific answer, provided
        by the command type.
		
        Parameters
        --------------------
        buffer_type : str
            Empty string by default, shall contain a specific buffer type
        Returns
        --------------------
        Value of the response, if None then an error occurred. 
        """
		
        result = None
		
        if buffer_type == 'actuator_cmd':
            while True:
                self.socket.send(pickle.dumps(buffer_type))
                self.logger.info('ReceptionPoint::sendRequest - request sent. Waiting for answer')

                if self.poller.poll(timeout=self.timeout):
                    data = self.socket.recv()
                    response = pickle.loads(data)
                    self.logger.info('ReceptionPoint::sendRequest - answer received.')
                    result = response
                    break
                else:
                    result = None
                    self._reset_socket()
                    self.logger.error('ReceptionPoint::sendRequest - Timeout.')
                    break
        
        elif buffer_type == "loop_cmd":
            while True:
                try:
                    if self.actual_loop_state is None:
                        self.actual_loop_state = -1
                    payload = {
                        "buffer_type": "loop_cmd",
                        "actual_state": self.actual_loop_state
                    }
                    self.logger.debug(f"ReceptionPoint::sendRequest - actual_state: {self.actual_loop_state} ")
                    self.socket.send(pickle.dumps(payload))

                    if self.poller.poll(timeout=self.timeout):
                        data = self.socket.recv()
                        response = pickle.loads(data)
                        self.logger.debug("ReceptionPoint::sendRequest - loop_cmd answer received ")
                        result = int(response["desired_state"] )  # 0/1
                        self.actual_loop_state = result
                        self.logger.debug(f"ReceptionPoint::sendRequest - desired_state: {self.actual_loop_state}")
                        break
                    else:
                        result = None
                        self._reset_socket()
                        self.logger.error("ReceptionPoint::sendRequest - loop_cmd timeout.")
                        break
                except Exception as e:
                    self.logger.error(f"ReceptionPoint::sendRequest - loop_cmd error: {e}")
                    result = None
                    break
        
        else:
            result = None
            self.logger.warning('ReceptionPoint::sendRequest - Unsupported buffer type.')
        return result
    
    
    def _reset_socket(self):
        """
        Replace the REQ socket after a timeout (a REQ socket that sent a request without a reply cannot send again): the old
        socket is removed from the poller before being closed, so that the poller never waits on closed sockets.
        """
        try:
            self.poller.unregister(self.socket)
        except KeyError:
            pass
        try:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
        except Exception:
            pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.protocol + '://' + self.ip + ':' + str(self.port))
        self.poller.register(self.socket, zmq.POLLIN)

    def __del__(self):
        if self.socket is not None and self.context is not None:
            self.socket.close()
            self.context.term()
