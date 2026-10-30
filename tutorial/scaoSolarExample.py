"""
SCAO loop on an extended solar source for the EST: correlating Shack-Hartmann (red, 36x36),
37x37 ASM, leaky integrator. A point-like on-axis source feeds the science camera.

Cache: the atmosphere, modal basis and interaction matrix are generated on the first run and
reused afterwards. Together with the Savepoint output they live in
~/simulations/tutorials/scaoSolarExample/ (delete the files there to regenerate).
Usage: python tutorial/scaoSolarExample.py [--iterations N] [--dry-run]
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
import datetime
import logging
import numpy as np
import gc
import matplotlib.pyplot as plt
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from joblib import Parallel, delayed

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Atmosphere import Atmosphere
from SAOS.Telescope import Telescope
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Source import Source
from SAOS.ShackHartmann import ShackHartmann
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.ScienceCam import ScienceCam
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Vibration import Vibration
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint

# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--iterations', type=int, default=3000, help='Number of AO loop iterations (default: %(default)s)')
parser.add_argument('--dry-run', action='store_true', help='Build every object up to the light paths and exit (no IM, no loop)')
args = parser.parse_args()
nIterations = args.iterations

scienceFs = 56. # Hz

nModes = None # [nModesASM, nModesM7]
im_stroke = [5e-7] # in meters

# Cache directory: files that are expensive to compute are generated once and reused
CACHE_DIR = os.path.join(os.path.expanduser("~"), 'simulations/tutorials', os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(CACHE_DIR, exist_ok=True)

atm_file = os.path.join(CACHE_DIR, 'atmosphere.h5')
modalBasis_file = os.path.join(CACHE_DIR, 'modalBasis.h5')
IM_file = os.path.join(CACHE_DIR, 'interactionMatrix.h5')

date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

## Define data sharepoint

sharepoint = Sharepoint(test_logger.logger, port=5573, atm=1, atm_per_dir=0, dm=1, dm_per_dir=0, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1)

## Define the savingpoint
savepoint = Savepoint(file_path=os.path.join(CACHE_DIR, f'savepoint_{date}.h5'), atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

## Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture_red = 36
resolution = n_subaperture_red * 6 # resolution of the phase screen in [px]
pixel_size = diameter / resolution
tel_fov = 60 # in [arcsec]

est_tel = Telescope(diameter = diameter,
                    resolution = resolution,
                    centralObstruction= obs_diameter / diameter,
                    samplingTime=sampling_time,
                    fov=tel_fov,
                    logger=test_logger.logger)

## Atmosphere:

atm = Atmosphere(r0 = 0.08,
                 L0= 25,
                 fractionalR0=[0.53, 0.37, 0.05, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[45, 90, 180, 67, 2],
                 windSpeed=[6.7, 8.5, 12.7, 23.4, 8.3],
                 telescope=est_tel,
                 zenith = 0,
                 logger=test_logger.logger)

if os.path.exists(atm_file):
    atm.load(atm_file)
else:
    atm.initializeAtmosphere()
    atm.save(atm_file)

## Sources:
ngs_red = Source(magnitude = 5,
             optBand = 'R4',
             coordinates=[0,0],
             logger=test_logger.logger)  

sun_red = ExtendedSource(optBand = 'R',
                     coordinates=[0, 0],
                     nSubDirs=3,
                     fov=9.269,
                     subDir_margin=4.0,
                     patch_padding=5.0,
                     logger=test_logger.logger)       

## Deformable mirrors:

# No DM dynamics by default; to enable a dynamic model pass the path of a discrete model file,
# e.g. 'dynamicModel': '<path to asm_discrete_model.h5>'
asm_params = {'dynamicModel': '', 'validActThreshpercentage': 0.7533}

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='cartesian',
                        mechCoupling=0.6,
                        logger=test_logger.logger,
                        **asm_params) # ASM

dms = [asm]          

## Wavefront Sensor

red_wfs = CorrelatingShackHartmann(telescope=est_tel,
                                    src=sun_red,
                                    lightRatio=0.9,
                                    nSubap=n_subaperture_red,
                                    plate_scale=0.403,
                                    fieldOfView=9.269,
                                    guardPx=2,
                                    fft_fieldOfView_oversampling=0.5,
                                    use_brightest=9,
                                    unit_in_rad=False,
                                    logger=test_logger.logger)

## Science camera

red_scicam = ScienceCam(fieldOfView=9.269, 
                         plate_scale = 0.0167,
                         samplingTime=est_tel.samplingTime,
                         telescope=est_tel,
                         integrationTime=1./scienceFs,
                         noiseFlag=False,
                         logger=test_logger.logger)

## Build the Light Path

scao_light_path_list = []

# Create red branch
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=sun_red, 
                                         atm=atm, 
                                         tel=est_tel, 
                                         dm=dms[0], 
                                         wfs=red_wfs, 
                                         vibration=None, 
                                         sci=None, 
                                         delay=1)

scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ngs_red, 
                                         atm=atm, 
                                         tel=est_tel, 
                                         dm=dms[0], 
                                         wfs=None, 
                                         vibration=None, 
                                         sci=red_scicam, 
                                         delay=1)

lightPathTasks = []
for i in range(len(scao_light_path_list)):
    lightPathTasks.append(delayed(scao_light_path_list[i].propagate)(True))

test_logger.logger.info(f'The Modules initialization took {time.time()-t0} [s]')

if args.dry_run:
    test_logger.logger.info('Dry run: every object up to the light paths was built; exiting before the IM and the loop.')
    raise SystemExit(0)

## Interaction Matrix
t0 = time.time()

im_handler = InteractionMatrixHandler(test_logger.logger)
im_handler.initialize_im_class(scao_light_path_list)

basis_cached = os.path.exists(modalBasis_file)
if basis_cached:
    im_handler.load_modalBasis(modalBasis_file)
if os.path.exists(IM_file):
    im_handler.load_IM(IM_file)
else:
    im_handler.measure(modal_basis='dh', stroke=im_stroke, nModes=nModes)
    im_handler.save_IM(IM_file)
    if not basis_cached:
        im_handler.save_modalBasis(modalBasis_file)

test_logger.logger.info(f'The IM creation took {time.time()-t0} [s]')

# Controller class
controller_kwargs = {'rcond':0.025, 
                    'beta':1e-4,
                    'gain':[0.2],
                    'decay':[0.99],
                    'ki':[0.0]}

controller = Controller(telescope=est_tel,
                        interactionMatrix=im_handler,
                        reconstructionMethod='tikhonov',
                        controllerType='leaky',
                        logger=test_logger.logger,
                        **controller_kwargs)

test_logger.logger.info('Beginning simulation')

# SCAO loop
for i in range(nIterations):
    if ((i+1)%10) == 0:
        est_tel.logger.info(f'Iteration {i+1}')
        if scao_light_path_list[1].sci_frame is not None:
            test_logger.logger.info(f'PSF peak: {scao_light_path_list[1].sci_frame.max()}')
    # Update the atmosphere
    atm.update()
    # Propagate the light
    Parallel(n_jobs=2, prefer="threads")(lightPathTasks)
    # Compute command
    cmd = controller.computeControlAction(scao_light_path_list)
    # Update the DM shape
    for j in range(len(dms)):
        dms[j].updateDMShape(cmd[j])
    # Share data with the GUI
    # sharepoint.shareData(scao_light_path_list, i, [atm], dms)              
 
    # Save Data
    savepoint.save([atm], i)
    savepoint.save(dms, i)
    savepoint.save(scao_light_path_list, i)   

    if (i + 1) % 100 == 0:
        if HAS_TORCH:
            torch.cuda.empty_cache()
        gc.collect()

test_logger.logger.info('Simulation ended')
    
# Force destructor
del est_tel, atm, asm, red_wfs, controller, im_handler, scao_light_path_list, savepoint
test_logger = None

if HAS_TORCH:
    torch.cuda.empty_cache()
gc.collect()
