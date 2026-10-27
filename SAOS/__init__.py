# -*- coding: utf-8 -*-
"""
Created on Thu May 8 2025

@author: nlinares
"""

import logging
import sys

__version__ = "5.1.0.dev0"

# SAOS propagates the light paths in threads: with the GIL enabled they do not run in parallel (results are the same, only slower)
if getattr(sys, '_is_gil_enabled', lambda: True)():
    logging.getLogger(__name__).info('SAOS: the GIL is enabled; a free-threaded Python (e.g. 3.13t/3.14t with PYTHON_GIL=0) is '
                                     'recommended to run the light paths in parallel.')

# Only print the banner when running interactively (not when imported by documentation tools)
if sys.stdout.isatty():
    print("""
      ███████  █████   ██████  ███████ 
      ██      ██   ██ ██    ██ ██      
      ███████ ███████ ██    ██ ███████ 
           ██ ██   ██ ██    ██      ██ 
      ███████ ██   ██  ██████  ███████ 
                                 
                     Legacy from OOPAO""")
