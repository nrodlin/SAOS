from dataclasses import dataclass
import numpy as np


@dataclass
class WFSGeometry:
    """
    WFS Geometry.

    field_angle_arcsec :
        (theta_x, theta_y) w.r.t optical axis.
    """
    name: str
    field_angle_arcsec: tuple[float, float]

    diameter: float
    n_subaps: int

    valid_subaps: np.ndarray | None = None

    is_lgs: bool = False
    lgs_altitude: float | None = None


@dataclass
class DMGeometry:
    """
    DM geometry
    """

    name: str
    altitude: float

    n_actuators: int

    # TODO:
    # actuator_coordinates: np.ndarray | None = None
    # influence_functions: np.ndarray | None = None


@dataclass
class AtmosphereProfile:
    """
    PAtmospheric profile used by Learn/Apply.
    """

    layer_altitudes: np.ndarray

    cn2_weights: np.ndarray

    r0: float
    L0: float

    wind_vx: np.ndarray
    wind_vy: np.ndarray


@dataclass
class TomographyConfig:
    """
    Reconstructor configuration
    """

    measured_wfs: list[WFSGeometry]

    # If None -> target_wfs = measured_wfs
    target_wfs: list[WFSGeometry] | None

    dms: list[DMGeometry]

    regularization: float = 1e-6
    delay: float = 0.0