"""
Logging of SAOS: one queue and one listener thread per process, shared by every module created without an external logger.

The listener is started the first time a logger is requested (get_logger) and stopped at interpreter exit (atexit), so that no
object has to join the listener thread in __del__ (which hangs at interpreter shutdown) and every message is printed once,
whatever the number of SAOS objects.
"""
import atexit
import logging
import logging.handlers
import threading
from queue import Queue

_listener = None
_lock = threading.Lock()


def _stop_listener():
    if _listener is not None:
        _listener.stop()                                   # idempotent


def get_logger(logging_level=None):
    """
    Root logger fed through the shared queue and listener of SAOS (console output, "time - level - message").

    Parameters
    ----------
    logging_level : int, optional
        Minimum level of the root logger. None keeps the current one (WARNING when the listener is created here).

    Returns
    -------
    logging.Logger
        The root logger.
    """
    global _listener
    root_logger = logging.getLogger()
    with _lock:
        if _listener is None:
            log_queue = Queue()
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            _listener = logging.handlers.QueueListener(log_queue, console_handler)
            root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
            root_logger.setLevel(logging.WARNING if logging_level is None else logging_level)
            atexit.register(_stop_listener)
        elif logging_level is not None:
            root_logger.setLevel(logging_level)
        if _listener._thread is None:                     # not started yet, or stopped by the user: (re)start it
            _listener.start()
    return root_logger


class LoggingHelper:
    """
    Logger for scripts: LoggingHelper(level).logger is the shared SAOS root logger at the given level. queue_listerner is the shared
    listener (kept for compatibility; stopping it is optional, it is stopped at exit).
    """
    def __init__(self, logging_level=logging.WARNING):
        self.logger = get_logger(logging_level)
        self.queue_listerner = _listener
