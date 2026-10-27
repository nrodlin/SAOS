"""
Solar GLAO for the EST: 7 extended-source directions (on axis + ring of 6 at 15 arcsec) sensed by a
correlating Shack-Hartmann, one radial ASM, and point-like evaluation sources at several field angles.

Cache: the atmosphere, modal basis and interaction matrix are generated on the first run and
reused afterwards. Together with the Savepoint output they live in
~/simulations/tutorials/glaoExampleSun/ (delete the files there to regenerate).
Usage: python tutorial/glaoExampleSun.py [--iterations N] [--dry-run]
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
parser.add_argument('--iterations', type=int, default=1000, help='Number of AO loop iterations (default: %(default)s)')
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

# sharepoint = Sharepoint(test_logger.logger, port=5574, atm=1, atm_per_dir=0, dm=1, dm_per_dir=0, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1)

## Define the savingpoint
savepoint = Savepoint(file_path=os.path.join(CACHE_DIR, f'savepoint_{date}.h5'), atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

## Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture_red = 36
resolution = n_subaperture_red * 4 # resolution of the phase screen in [px]
pixel_size = diameter / resolution
tel_fov = 60 # in [arcsec]

est_tel = Telescope(diameter = diameter,
                    resolution = resolution,
                    centralObstruction= obs_diameter / diameter,
                    samplingTime=sampling_time,
                    fov=tel_fov,
                    logger=test_logger.logger)

spider_angle = [0, 90, 180, 270] # in [º]
spider_thickness = 0.060 # in [m]

# est_tel.apply_spiders(spider_angle, spider_thickness)

## Atmosphere:

atm = Atmosphere(r0 = 0.12,
                 L0= 25,
                 fractionalR0=[0.53, 0.37, 0.05, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[312, 47, 198, 123, 276],
                 windSpeed=[2.8, 5.3, 9.6, 18.4, 17.8],
                 telescope=est_tel,
                 zenith = 0,
                 logger=test_logger.logger)

if os.path.exists(atm_file):
    atm.load(atm_file)
else:
    atm.initializeAtmosphere()
    atm.save(atm_file)

## Sources:
# On axis
# 6 NGS cicurlarly distributed at a radious of 15" and rotated 45º
coord_list = [[0, 0]] + [[15, i * 360 / 6 + 45] for i in range(6)]

sun_list = []

for i in range(len(coord_list)):
    sun_list.append(ExtendedSource(optBand = 'R',
                     coordinates=coord_list[i],
                     nSubDirs=3,
                     fov=9.269,
                     subDir_margin=4.0,
                     patch_padding=5.0,
                     logger=test_logger.logger))
    

ext_sci_ngs_0 = Source(magnitude=5, optBand='R4', coordinates=[0,0], logger=test_logger.logger)
ext_sci_ngs_5 = Source(magnitude=5, optBand='R4', coordinates=[5,0], logger=test_logger.logger)
ext_sci_ngs_10 = Source(magnitude=5, optBand='R4', coordinates=[10,0], logger=test_logger.logger)
ext_sci_ngs_15 = Source(magnitude=5, optBand='R4', coordinates=[15,0], logger=test_logger.logger)
ext_sci_ngs_20 = Source(magnitude=5, optBand='R4', coordinates=[20,0], logger=test_logger.logger)
ext_sci_ngs_30 = Source(magnitude=5, optBand='R4', coordinates=[30,0], logger=test_logger.logger)

## Deformable mirrors:

# No DM dynamics by default; to enable a dynamic model pass the path of a discrete model file,
# e.g. 'dynamicModel': '<path to asm_discrete_model.h5>'
asm_params = {'dynamicModel': '', 'validActThreshpercentage': 0.5}

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='radial',
                        logger=test_logger.logger,
                        **asm_params) # ASM


m3 = DeformableMirror(telescope=est_tel,
                        nActs=25,
                        altitude=20000,
                        typeDM='cartesian',
                        logger=test_logger.logger) # M3

m6 = DeformableMirror(telescope=est_tel,
                        nActs=25,
                        altitude=5000,
                        typeDM='cartesian',
                        logger=test_logger.logger) # M6

dms = [asm]

## Vibration

# No vibrations by default. To enable them, pass a Vibration file (data in arcsec on sky), e.g.:
# red_vibrations = Vibration(est_tel, '<path to vibration file>.h5', test_logger.logger)
red_vibrations = None
vis_vibrations = None

## Wavefront Sensor

red_wfs = CorrelatingShackHartmann(telescope=est_tel,
                                    src=sun_list[0],
                                    lightRatio=0.9,
                                    nSubap=n_subaperture_red,
                                    plate_scale=0.403,
                                    fieldOfView=9.269,
                                    guardPx=2,
                                    fft_fieldOfView_oversampling=0.5,
                                    use_brightest=5,
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
for i in range(len(sun_list)):
    if (i == 0) or (i == 6): # Add science camera
        scao_light_path_list.append(LightPath(test_logger.logger))
        scao_light_path_list[-1].initialize_path(src=sun_list[i], atm=atm, tel=est_tel, dm=dms, wfs=red_wfs, vibration=red_vibrations, sci=red_scicam, delay=0)
    else:
        scao_light_path_list.append(LightPath(test_logger.logger))
        scao_light_path_list[-1].initialize_path(src=sun_list[i], atm=atm, tel=est_tel, dm=dms, wfs=red_wfs, vibration=red_vibrations, sci=None, delay=0)

scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ext_sci_ngs_5,  atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=red_vibrations, sci=red_scicam, delay=0)
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ext_sci_ngs_10, atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=red_vibrations, sci=red_scicam, delay=0)
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ext_sci_ngs_15, atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=red_vibrations, sci=red_scicam, delay=0)
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ext_sci_ngs_20, atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=red_vibrations, sci=red_scicam, delay=0)
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ext_sci_ngs_30, atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=red_vibrations, sci=red_scicam, delay=0)


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
                    'gain':[0.25],
                    'decay':[0.999],
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
    est_tel.logger.info(f'Iteration {i+1}')
    # Update the atmosphere
    atm.update()
    # Propagate the light
    Parallel(n_jobs=1, prefer="threads")(lightPathTasks)
    # Compute command
    cmd = controller.computeControlAction(scao_light_path_list)
    # Update the DM shape
    for j in range(len(dms)):
        dms[j].updateDMShape(cmd[j])
    # Share data with the GUI
    # sharepoint.shareData(scao_light_path_list, i, [atm], dms)   
    test_logger.logger.info(f'Atm OPD: {np.sqrt(np.mean(scao_light_path_list[7].atmosphere_opd[est_tel.pupil]**2))},\
                              DM OPD: {np.sqrt(np.mean(scao_light_path_list[7].dm_opd[0][est_tel.pupil]**2))},\
                              WFS OPD: {np.sqrt(np.mean(scao_light_path_list[7].wfs_opd[est_tel.pupil]**2))}')           
    test_logger.logger.info(f'Slopes axis: {np.std(scao_light_path_list[0].slopes_1D[:red_wfs.nSignal//2])}, {np.std(scao_light_path_list[0].slopes_1D[red_wfs.nSignal//2:])},\
        Slopes ring: {np.std(scao_light_path_list[1].slopes_1D[:red_wfs.nSignal//2])}, {np.std(scao_light_path_list[1].slopes_1D[red_wfs.nSignal//2:])}')
    # Save Data
    savepoint.save([atm], i)
    savepoint.save(dms, i)
    savepoint.save(scao_light_path_list, i)

test_logger.logger.info('Simulation ended.')

# Force destructor
del est_tel, atm, asm, red_wfs, controller, im_handler, scao_light_path_list, savepoint
test_logger = None

if HAS_TORCH:
    torch.cuda.empty_cache()
gc.collect()
