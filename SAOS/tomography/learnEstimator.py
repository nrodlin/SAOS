from dataclasses import dataclass
import numpy as np

from .tomoDataClasses import AtmosphereProfile, TomographyConfig
from .telemetryReader import TelemetryReader
from .covarianceBuilder import CovarianceBuilder


@dataclass
class EmpiricalCovariances:
    """
    Empirical covariance matrices estimated from telemetry.
    """

    Css: np.ndarray
    Cdt: np.ndarray | None = None
    n_frames: int | None = None
    delay_frames: int | None = None


class LearnEstimator:
    """
    Estimate atmospheric parameters from WFS telemetry.

    First stage:
        - Read empirical covariance matrices from HDF5.
        - Validate compatibility with analytical covariance matrices.
    """

    def __init__(
        self,
        config: TomographyConfig,
        reader: TelemetryReader,
        datasets_by_wfs: dict[str, str],
    ):
        self.config = config
        self.reader = reader
        self.datasets_by_wfs = datasets_by_wfs

        self.builder = CovarianceBuilder(config)

        self._validate_wfs_dataset_mapping()

    def compute_empirical_covariances(
        self,
        frame_slice: slice | None = None,
        delay_frames: int | None = None,
        chunk_size: int = 5000,
    ) -> EmpiricalCovariances:
        """
        Compute empirical zero-delay and delayed covariance matrices.
        """

        Css_emp = compute_empirical_covariance_from_h5(
            reader=self.reader,
            datasets_by_wfs=self.datasets_by_wfs,
            frame_slice=frame_slice,
            chunk_size=chunk_size,
        )

        Cdt_emp = None

        if delay_frames is not None and delay_frames > 0:
            Cdt_emp = compute_empirical_delayed_covariance_from_h5(
                reader=self.reader,
                datasets_by_wfs=self.datasets_by_wfs,
                delay_frames=delay_frames,
                frame_slice=frame_slice,
                chunk_size=chunk_size,
            )

        return EmpiricalCovariances(
            Css=Css_emp,
            Cdt=Cdt_emp,
            delay_frames=delay_frames,
        )

    def validate_against_model(
        self,
        empirical: EmpiricalCovariances,
        atmosphere: AtmosphereProfile,
    ) -> None:
        """
        Validate empirical covariance dimensions against model dimensions.
        """

        Css_model = self.builder.build_css(atmosphere)

        if empirical.Css.shape != Css_model.shape:
            raise ValueError(
                "Empirical Css shape does not match model Css shape: "
                f"{empirical.Css.shape} != {Css_model.shape}"
            )

        if empirical.Cdt is not None:
            Cdt_model = self.builder.build_covariance(
                output_wfs=self.config.measured_wfs,
                input_wfs=self.config.measured_wfs,
                atmosphere=atmosphere,
                predictive_delay=self.config.delay,
                shift_output=True,
            )

            if empirical.Cdt.shape != Cdt_model.shape:
                raise ValueError(
                    "Empirical Cdt shape does not match model Cdt shape: "
                    f"{empirical.Cdt.shape} != {Cdt_model.shape}"
                )

    def _validate_wfs_dataset_mapping(self) -> None:
        """
        Validate that all measured WFSs have an associated HDF5 dataset.
        """

        measured_names = [wfs.name for wfs in self.config.measured_wfs]

        missing = [
            name for name in measured_names
            if name not in self.datasets_by_wfs
        ]

        if missing:
            raise ValueError(
                "Missing HDF5 dataset paths for measured WFSs: "
                f"{missing}"
            )

        extra = [
            name for name in self.datasets_by_wfs
            if name not in measured_names
        ]

        if extra:
            raise ValueError(
                "HDF5 dataset paths were provided for unknown WFSs: "
                f"{extra}"
            )
    def validate_dataset_shapes_against_geometry(self) -> None:
        """
        Validate each HDF5 dataset slope count against the WFS geometry.
        """

        shapes = self.reader.get_dataset_shapes(self.datasets_by_wfs)

        for wfs in self.config.measured_wfs:
            expected_n_slopes = 2 * self.builder._get_n_valid_subaps(wfs)
            actual_n_slopes = shapes[wfs.name][1]

            if actual_n_slopes != expected_n_slopes:
                raise ValueError(
                    f"WFS '{wfs.name}' has incompatible slope count: "
                    f"HDF5 has {actual_n_slopes}, model expects {expected_n_slopes}."
                )        


def compute_empirical_covariances_from_array(
    slopes: np.ndarray,
    delay_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute empirical zero-delay and delayed covariance matrices from RAM data.
    """

    if slopes.ndim != 2:
        raise ValueError("slopes must have shape (n_frames, n_slopes).")

    n_frames = slopes.shape[0]

    if n_frames < 2:
        raise ValueError("At least two frames are required.")

    mean_s = np.mean(slopes, axis=0)

    Css_emp = slopes.T @ slopes / n_frames
    Css_emp -= np.outer(mean_s, mean_s)

    if delay_frames <= 0:
        return Css_emp, None

    if delay_frames >= n_frames:
        raise ValueError("delay_frames must be smaller than n_frames.")

    S0 = slopes[:-delay_frames]
    S1 = slopes[delay_frames:]

    Cdt_emp = S1.T @ S0 / S0.shape[0]
    Cdt_emp -= np.outer(
        np.mean(S1, axis=0),
        np.mean(S0, axis=0),
    )

    return Css_emp, Cdt_emp


def compute_empirical_covariance_from_h5(
    reader: TelemetryReader,
    datasets_by_wfs: dict[str, str],
    frame_slice: slice | None = None,
    chunk_size: int = 5000,
) -> np.ndarray:
    """
    Compute Css_emp = cov(s(t), s(t)) from HDF5 telemetry using chunks.

    This avoids loading the full telemetry buffer into RAM.
    """

    n_total_frames = 0
    sum_s = None
    sum_ss = None

    for chunk in reader.iter_wfs_slopes(
        datasets_by_wfs=datasets_by_wfs,
        frame_slice=frame_slice,
        chunk_size=chunk_size,
    ):
        n_chunk = chunk.shape[0]

        if sum_s is None:
            n_slopes = chunk.shape[1]
            sum_s = np.zeros(n_slopes, dtype=float)
            sum_ss = np.zeros((n_slopes, n_slopes), dtype=float)

        sum_s += np.sum(chunk, axis=0)
        sum_ss += chunk.T @ chunk
        n_total_frames += n_chunk

    if n_total_frames == 0:
        raise ValueError("No frames were read from telemetry.")

    mean_s = sum_s / n_total_frames

    Css_emp = sum_ss / n_total_frames
    Css_emp -= np.outer(mean_s, mean_s)

    return Css_emp


def compute_empirical_delayed_covariance_from_h5(
    reader: TelemetryReader,
    datasets_by_wfs: dict[str, str],
    delay_frames: int,
    frame_slice: slice | None = None,
    chunk_size: int = 5000,
) -> np.ndarray:
    """
    Compute Cdt_emp = cov(s(t + delay), s(t)) from HDF5 telemetry.

    This version reads aligned chunks directly:
        S0 = s(t)
        S1 = s(t + delay)
    """

    if delay_frames <= 0:
        raise ValueError("delay_frames must be positive.")

    shapes = reader.get_dataset_shapes(datasets_by_wfs)
    n_frames = next(iter(shapes.values()))[0]

    if frame_slice is None:
        start, stop, step = 0, n_frames, 1
    else:
        start, stop, step = frame_slice.indices(n_frames)

    if step != 1:
        raise ValueError("Only frame slices with step=1 are supported.")

    if stop - start <= delay_frames:
        raise ValueError("Frame slice is too short for the requested delay.")

    pair_start = start
    pair_stop = stop - delay_frames

    n_pairs = 0
    sum_s0 = None
    sum_s1 = None
    sum_s1s0 = None

    for chunk_start in range(pair_start, pair_stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, pair_stop)

        S0 = reader.read_wfs_slopes(
            datasets_by_wfs=datasets_by_wfs,
            frame_slice=slice(chunk_start, chunk_stop),
            remove_mean=False,
        )

        S1 = reader.read_wfs_slopes(
            datasets_by_wfs=datasets_by_wfs,
            frame_slice=slice(chunk_start + delay_frames, chunk_stop + delay_frames),
            remove_mean=False,
        )

        if sum_s0 is None:
            n_slopes = S0.shape[1]
            sum_s0 = np.zeros(n_slopes, dtype=float)
            sum_s1 = np.zeros(n_slopes, dtype=float)
            sum_s1s0 = np.zeros((n_slopes, n_slopes), dtype=float)

        sum_s0 += np.sum(S0, axis=0)
        sum_s1 += np.sum(S1, axis=0)
        sum_s1s0 += S1.T @ S0
        n_pairs += S0.shape[0]

    mean_s0 = sum_s0 / n_pairs
    mean_s1 = sum_s1 / n_pairs

    Cdt_emp = sum_s1s0 / n_pairs
    Cdt_emp -= np.outer(mean_s1, mean_s0)

    return Cdt_emp