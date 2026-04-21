# -*- coding: utf-8 -*-
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, NoReturn, List

import pandas as pd

from ts_benchmark.common.constant import (
    FORECASTING_DATASET_PATH,
    ANOMALY_DETECT_DATASET_PATH,
    ST_FORECASTING_DATASET_PATH,
)
from ts_benchmark.data.dataset import Dataset
from ts_benchmark.data.utils import load_series_info, read_data, read_covariates, get_covariate

logger = logging.getLogger(__name__)


class DataSource:
    """
    A class that manages and reads from data sources

    A data source is responsible for loading data into the internal dataset object,
    as well as detecting and updating data in the source storage.
    """

    # The class for the internal dataset object
    DATASET_CLASS = Dataset

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        initializer

        :param data_dict: A dictionary of time series, where the keys are the names and
            the values are DataFrames following the OTB protocol.
        :param covariate_dict: A dictionary of time series' covariates.
        :param metadata: A DataFrame where the index contains series names and columns
            contains meta-info fields.
        """
        self._dataset = self.DATASET_CLASS()
        self._dataset.set_data(data_dict, covariate_dict, metadata)

    @property
    def dataset(self) -> Dataset:
        """
        Returns the internally maintained dataset object

        This dataset is where the DataSource loads data into.
        """
        return self._dataset

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        """
        Loads a list of time series from the source

        The series data and (optionally) meta information are loaded into the internal dataset.

        :param series_list: The list of series names.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading series at runtime."
        )


class LocalDataSource(DataSource):
    """
    The data source that manages data files in a local directory
    """

    #: index column name of the metadata
    _INDEX_COL = "file_name"

    #: name of the data folder
    _DATA_FOLDER_NAME = "data"

    #: name of the covariates folder
    _COVARIATES_FOLDER_NAME = "covariates"

    def __init__(self, local_dataset_path: str, metadata_file_name: str):
        """
        initializer

        Only the metadata is loaded during initialization, while all series data are
        loaded on demand.

        :param local_dataset_path: the directory that contains csv data files, metadata and covariates files.
        :param metadata_file_name: name of the metadata file.
        """
        self.local_data_path = os.path.join(local_dataset_path, self._DATA_FOLDER_NAME)
        self.local_covariates_path = os.path.join(
            local_dataset_path, self._COVARIATES_FOLDER_NAME
        )
        self.metadata_path = os.path.join(local_dataset_path, metadata_file_name)
        metadata = self.update_meta_index()
        super().__init__({}, {}, metadata)

    def update_meta_index(self) -> pd.DataFrame:
        """
        Check if there are any user-added dataset files in the dataset folder

        Attempt to register them in the metadata and load metadata from the metadata file

        :return: metadata
        :rtype: pd.DataFrame
        """

        metadata = self._load_metadata()
        # csv_files = {
        #     f
        #     for f in os.listdir(self.local_data_path)
        #     if f.endswith(".csv") and f != os.path.basename(self.metadata_path)
        # }
        # user_csv_files = set(csv_files).difference(metadata.index)
        # if not user_csv_files:
        #     return metadata
        # data_info_list = []
        # for user_csv in user_csv_files:
        #     try:
        #         data_info_list.append(
        #             load_series_info(os.path.join(self.local_data_path, user_csv))
        #         )
        #     except Exception as e:
        #         raise RuntimeError(f"Error loading series info from {user_csv}: {e}")
        # new_metadata = pd.DataFrame(data_info_list)
        # new_metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        # metadata = pd.concat([metadata, new_metadata])
        # with open(self.metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        #     metadata.to_csv(csvfile, index=False)
        # logger.info(
        #     "Detected %s new user datasets, registered in the metadata",
        #     len(user_csv_files),
        # )
        return metadata

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        logger.info("Start loading %s series in parallel", len(series_list))
        data_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_series, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            data_dict[series_name] = future.result()

        covariate_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_covariates, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            covariate_dict[series_name] = future.result()
        logger.info("Data loading finished.")
        self.dataset.update_data(data_dict, covariate_dict)

    def _load_metadata(self) -> pd.DataFrame:
        """
        Loads metadata from a local csv file
        """
        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def _load_series(self, series_name: str) -> pd.DataFrame:
        """
        Loads a time series from a single data file

        :param series_name: Series name.
        :return: A time series in DataFrame format.
        """
        datafile_path = os.path.join(self.local_data_path, series_name)
        data = read_data(datafile_path)
        return data

    def _load_covariates(self, series_name: str) -> Optional[Dict]:
        """
        Loads a time series from a single data file

        :param series_name: Series name.
        :return: A time series in DataFrame format.
        """
        series_name_without_extension = os.path.splitext(series_name)[0]
        covariates_folder_path = os.path.join(self.local_covariates_path, series_name_without_extension)
        covariates = read_covariates(covariates_folder_path)
        return covariates


class LocalForecastingDataSource(LocalDataSource):
    """
    The local data source of the forecasting task
    """

    def __init__(self):
        super().__init__(FORECASTING_DATASET_PATH, "FORECAST_META.csv")


class LocalStForecastingDataSource(LocalDataSource):
    """
    The local data source of the st_forecasting task
    """

    def __init__(self):
        super().__init__(ST_FORECASTING_DATASET_PATH, "ST_FORECAST_META.csv")


class LocalAnomalyDetectDataSource(LocalDataSource):
    """
    The local data source of the anomaly detection task
    """

    def __init__(self):
        super().__init__(
            ANOMALY_DETECT_DATASET_PATH,
            "DETECT_META.csv",
        )


class S3DataSource(DataSource):
    """
    A data source that reads time-series data directly from an S3 bucket.

    Series metadata and CSV data files are streamed from S3 without writing
    them to the local filesystem.  Binary covariate files (e.g. ``.npz``) are
    downloaded to a temporary directory because the underlying loaders require
    a file path.

    Authentication and endpoint configuration are taken from the same
    environment variables used by :class:`scripts.dih_connect.s3_data_access.S3DataAccess`:
    ``DIH_S3_ENDPOINT``, ``DIH_S3_TOKEN``, ``DIH_S3_SECRET``, ``DIH_S3_REGION``,
    ``DIH_S3_BUCKET_NAME``.  The S3 prefix for the dataset is read from
    ``DIH_S3_DATASET_PREFIX`` (default ``data/tab/dataset``).
    """

    #: index column name of the metadata
    _INDEX_COL = "file_name"

    #: name of the data folder inside the S3 prefix
    _DATA_FOLDER_NAME = "data"

    #: name of the covariates folder inside the S3 prefix
    _COVARIATES_FOLDER_NAME = "covariates"

    def __init__(self, s3_dataset_subpath: str, metadata_file_name: str):
        """
        Initializer.

        Only metadata is loaded during construction; series data are loaded on demand.

        :param s3_dataset_subpath: Sub-path appended to ``DIH_S3_DATASET_PREFIX`` to
            locate the dataset, e.g. ``"forecasting"``.
        :param metadata_file_name: Name of the metadata CSV file, e.g.
            ``"FORECAST_META.csv"``.
        """
        from scripts.dih_connect.s3_data_access import S3DataAccess

        self._s3 = S3DataAccess()
        dataset_prefix = self._s3.dataset_prefix.strip("/")
        self._s3_prefix = f"{dataset_prefix}/{s3_dataset_subpath}"
        self._metadata_file_name = metadata_file_name
        metadata = self._load_metadata()
        super().__init__({}, {}, metadata)

    def _s3_key(self, *parts: str) -> str:
        """Builds an S3 key by joining the dataset prefix with *parts*."""
        return "/".join([self._s3_prefix.rstrip("/")] + list(parts))

    def _load_metadata(self) -> pd.DataFrame:
        """Loads metadata from S3."""
        key = self._s3_key(self._metadata_file_name)
        metadata = self._s3.read_csv(key)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        logger.info("Start loading %s series in parallel from S3", len(series_list))
        data_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_series, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            data_dict[series_name] = future.result()

        covariate_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_covariates, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            covariate_dict[series_name] = future.result()
        logger.info("Data loading from S3 finished.")
        self.dataset.update_data(data_dict, covariate_dict)

    def _load_series(self, series_name: str) -> pd.DataFrame:
        """Streams a series CSV directly from S3."""
        key = self._s3_key(self._DATA_FOLDER_NAME, series_name)
        raw_df = self._s3.read_csv(key)
        from ts_benchmark.data.utils import is_st, process_data_df, process_data_np
        if is_st(raw_df):
            return process_data_np(raw_df)
        return process_data_df(raw_df)

    def _load_covariates(self, series_name: str) -> Optional[Dict]:
        """Downloads binary covariate files from S3 into a temp directory."""
        series_name_without_extension = os.path.splitext(series_name)[0]
        prefix = self._s3_key(self._COVARIATES_FOLDER_NAME, series_name_without_extension)

        objects = self._s3.list_objects(prefix + "/")

        if not objects:
            return None

        covariates = {}
        with tempfile.TemporaryDirectory(prefix="tab_covariates_") as tmp_dir:
            for obj in objects:
                key = obj["Key"]
                filename = os.path.basename(key)
                local_file = os.path.join(tmp_dir, filename)
                self._s3.download_file(key, local_file)
                try:
                    # get_covariate() loads the data fully into memory (e.g. scipy
                    # sparse matrix), so the temp file can be safely deleted when
                    # the context manager exits.
                    covariates[filename] = get_covariate(local_file)
                except Exception as exc:
                    logger.warning("Error reading covariate %s: %s", filename, exc)

        return covariates if covariates else None


class S3ForecastingDataSource(S3DataSource):
    """S3-backed data source for the forecasting task."""

    def __init__(self):
        super().__init__("forecasting", "FORECAST_META.csv")


class S3StForecastingDataSource(S3DataSource):
    """S3-backed data source for the spatial-temporal forecasting task."""

    def __init__(self):
        super().__init__("st_forecasting", "ST_FORECAST_META.csv")


class S3AnomalyDetectDataSource(S3DataSource):
    """S3-backed data source for the anomaly detection task."""

    def __init__(self):
        super().__init__("anomaly_detect", "DETECT_META.csv")
