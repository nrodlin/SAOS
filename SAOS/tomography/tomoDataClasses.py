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

    @classmethod
    def from_saos_wfs(cls, wfs, name: str = None) -> "WFSGeometry":
        r_arcsec = wfs.src.coordinates[0]
        theta_rad = np.deg2rad(wfs.src.coordinates[1])
        x_arcsec = r_arcsec * np.cos(theta_rad)
        y_arcsec = r_arcsec * np.sin(theta_rad)
        
        is_lgs = (wfs.src.tag == 'LGS')
        lgs_altitude = wfs.src.altitude if is_lgs else None
        
        valid_subaps = None
        if hasattr(wfs, 'valid_subapertures'):
            valid_subaps = wfs.valid_subapertures.ravel()
            
        return cls(
            name=name if name else getattr(wfs, 'tag', 'WFS'),
            field_angle_arcsec=(x_arcsec, y_arcsec),
            diameter=wfs.telescope.D,
            n_subaps=wfs.nSubap,
            valid_subaps=valid_subaps,
            is_lgs=is_lgs,
            lgs_altitude=lgs_altitude
        )


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

    @classmethod
    def from_saos_dm(cls, dm, name: str = None) -> "DMGeometry":
        return cls(
            name=name if name else getattr(dm, 'tag', 'DM'),
            altitude=dm.altitude,
            n_actuators=dm.nActs
        )


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

    @classmethod
    def from_saos_lightpaths(
        cls, 
        measured_lps: list, 
        target_lps: list | None = None,
        regularization: float = 1e-6
    ) -> "TomographyConfig":
        
        measured_wfs = [WFSGeometry.from_saos_wfs(lp.wfs, name=f"WFS_meas_{i}") for i, lp in enumerate(measured_lps)]
        
        target_wfs = None
        if target_lps is not None:
            target_wfs = [WFSGeometry.from_saos_wfs(lp.wfs, name=f"WFS_target_{i}") for i, lp in enumerate(target_lps)]
            
        all_lps = measured_lps + (target_lps if target_lps else [])
        unique_dms = []
        for lp in all_lps:
            if lp.dm is not None:
                for dm in lp.dm:
                    if dm not in unique_dms:
                        unique_dms.append(dm)
                        
        dms = [DMGeometry.from_saos_dm(dm, name=f"DM_{i}") for i, dm in enumerate(unique_dms)]
        
        # Assume delay is same for all measured LPs.
        # delay in SAOS LightPath is in frames. Convert to seconds.
        delay_seconds = 0.0
        if len(measured_lps) > 0:
            delay_seconds = measured_lps[0].delay * measured_lps[0].tel.samplingTime
            
        return cls(
            measured_wfs=measured_wfs,
            target_wfs=target_wfs,
            dms=dms,
            regularization=regularization,
            delay=delay_seconds
        )