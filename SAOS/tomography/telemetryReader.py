import h5py
import numpy as np


class TelemetryReader:
    """
    Read WFS slope telemetry from an HDF5 file.

    Each WFS dataset is expected to have shape:
        (n_frames, n_slopes)

    Slope layout inside each dataset:
        [x slopes..., y slopes...]

    Output layout after concatenating WFS datasets:
        [WFS0_x, WFS0_y, WFS1_x, WFS1_y, ...]
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path

    def read_wfs_slopes(
        self,
        datasets_by_wfs: dict[str, str],
        frame_slice: slice | None = None,
        remove_mean: bool = True,
    ) -> np.ndarray:
        """
        Read selected WFS slope datasets into memory.

        Use this only for reasonably sized buffers.
        For large HDF5 files, prefer iter_wfs_slopes().
        """

        slopes_per_wfs = []

        with h5py.File(self.h5_path, "r") as h5_file:
            for wfs_name, dataset_path in datasets_by_wfs.items():
                dataset = self._get_dataset(
                    h5_file=h5_file,
                    dataset_path=dataset_path,
                    wfs_name=wfs_name,
                )

                slopes = self._read_dataset_slice(
                    dataset=dataset,
                    frame_slice=frame_slice,
                )

                slopes_per_wfs.append(slopes)

        slopes = self._concatenate_wfs_slopes(slopes_per_wfs)

        if remove_mean:
            slopes -= np.mean(slopes, axis=0, keepdims=True)

        return slopes

    def iter_wfs_slopes(
        self,
        datasets_by_wfs: dict[str, str],
        frame_slice: slice | None = None,
        chunk_size: int = 5000,
    ):
        """
        Iterate over WFS slope telemetry chunks.

        This method does not load the full telemetry buffer into RAM.
        It yields arrays with shape:
            (chunk_n_frames, n_total_slopes)
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        with h5py.File(self.h5_path, "r") as h5_file:
            datasets = []

            for wfs_name, dataset_path in datasets_by_wfs.items():
                dataset = self._get_dataset(
                    h5_file=h5_file,
                    dataset_path=dataset_path,
                    wfs_name=wfs_name,
                )
                datasets.append(dataset)

            n_frames = self._validate_dataset_compatibility(datasets)

            start, stop, step = self._normalize_frame_slice(
                frame_slice=frame_slice,
                n_frames=n_frames,
            )

            if step != 1:
                raise ValueError(
                    "iter_wfs_slopes only supports frame slices with step=1."
                )

            for chunk_start in range(start, stop, chunk_size):
                chunk_stop = min(chunk_start + chunk_size, stop)

                slopes_per_wfs = [
                    np.asarray(dataset[chunk_start:chunk_stop], dtype=float)
                    for dataset in datasets
                ]

                yield self._concatenate_wfs_slopes(slopes_per_wfs)

    def get_dataset_shapes(
        self,
        datasets_by_wfs: dict[str, str],
    ) -> dict[str, tuple[int, int]]:
        """
        Return dataset shapes without loading the datasets into RAM.
        """

        shapes = {}

        with h5py.File(self.h5_path, "r") as h5_file:
            for wfs_name, dataset_path in datasets_by_wfs.items():
                dataset = self._get_dataset(
                    h5_file=h5_file,
                    dataset_path=dataset_path,
                    wfs_name=wfs_name,
                )
                shapes[wfs_name] = tuple(dataset.shape)

        return shapes

    @staticmethod
    def _get_dataset(
        h5_file: h5py.File,
        dataset_path: str,
        wfs_name: str,
    ) -> h5py.Dataset:
        """
        Retrieve and validate a WFS dataset.
        """

        dataset_path = dataset_path.rstrip("/")

        if dataset_path not in h5_file:
            raise KeyError(
                f"Dataset '{dataset_path}' for WFS '{wfs_name}' "
                "was not found in the HDF5 file."
            )

        dataset = h5_file[dataset_path]

        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(
                f"Path '{dataset_path}' for WFS '{wfs_name}' "
                "is not an HDF5 dataset."
            )

        if dataset.ndim != 2:
            raise ValueError(
                f"Dataset '{dataset_path}' for WFS '{wfs_name}' "
                f"must have shape (n_frames, n_slopes), got {dataset.shape}."
            )

        return dataset

    @staticmethod
    def _read_dataset_slice(
        dataset: h5py.Dataset,
        frame_slice: slice | None,
    ) -> np.ndarray:
        """
        Read only the requested frame slice from a dataset.
        """

        if frame_slice is None:
            return np.asarray(dataset[...], dtype=float)

        return np.asarray(dataset[frame_slice], dtype=float)

    @staticmethod
    def _validate_dataset_compatibility(
        datasets: list[h5py.Dataset],
    ) -> int:
        """
        Validate that all datasets share the same number of frames.
        """

        if len(datasets) == 0:
            raise ValueError("No WFS datasets were provided.")

        n_frames = datasets[0].shape[0]

        for dataset in datasets:
            if dataset.shape[0] != n_frames:
                raise ValueError(
                    "All WFS datasets must have the same number of frames."
                )

        return n_frames

    @staticmethod
    def _concatenate_wfs_slopes(
        slopes_per_wfs: list[np.ndarray],
    ) -> np.ndarray:
        """
        Concatenate WFS slope arrays along the slope axis.
        """

        if len(slopes_per_wfs) == 0:
            raise ValueError("No WFS slope arrays were provided.")

        n_frames = slopes_per_wfs[0].shape[0]

        for slopes in slopes_per_wfs:
            if slopes.ndim != 2:
                raise ValueError("Each slope array must be 2D.")

            if slopes.shape[0] != n_frames:
                raise ValueError(
                    "All WFS slope arrays must have the same number of frames."
                )

        return np.concatenate(slopes_per_wfs, axis=1)

    @staticmethod
    def _normalize_frame_slice(
        frame_slice: slice | None,
        n_frames: int,
    ) -> tuple[int, int, int]:
        """
        Convert a Python slice into explicit start, stop, step values.
        """

        if frame_slice is None:
            return 0, n_frames, 1

        start, stop, step = frame_slice.indices(n_frames)

        return start, stop, step