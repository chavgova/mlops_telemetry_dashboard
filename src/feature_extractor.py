import os
import pickle
import tempfile
import typing

from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import mlflow
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    StringIndexerModel,
    OneHotEncoderModel,
)

from configs.configs import run_config, PathsConfig


class _FeatureExtractorData:
    """
    Class for holding the data for FeatureExtractor.

    It is important to separate data from logic.
    """

    def __init__(self):
        self._indexer_per_column: dict[str, StringIndexerModel] = {}
        self._encoder_per_column: dict[str, OneHotEncoderModel] = {}
        self._mean_amount_all_categories: typing.Optional[float] = None
        self._std_amount_all_categories: typing.Optional[float] = None
        self._mean_amount_per_category: typing.Optional[dict[str, float]] = {}
        self._std_amount_per_category: typing.Optional[dict[str, float]] = {}

    def is_set(self) -> bool:
        return (
            True
            if None not in self.__getstate__().values()
            and {} not in self.__getstate__().values()
            else False
        )

    def save(self, directory_path: str):
        with open(
            os.path.join(
                directory_path, PathsConfig.mean_amount_all_categories_file_name
            ),
            "wb",
        ) as f:
            pickle.dump(self._mean_amount_all_categories, f)
        with open(
            os.path.join(
                directory_path, PathsConfig.std_amount_all_categories_file_name
            ),
            "wb",
        ) as f:
            pickle.dump(self._std_amount_all_categories, f)
        with open(
            os.path.join(
                directory_path, PathsConfig.mean_amount_per_category_file_name
            ),
            "wb",
        ) as f:
            pickle.dump(self._mean_amount_per_category, f)
        with open(
            os.path.join(directory_path, PathsConfig.std_amount_per_category_file_name),
            "wb",
        ) as f:
            pickle.dump(self._std_amount_per_category, f)

        # Spark objects tend to be non-pickle'able due to their distributed nature. Because of that, we need to use its
        # own write function.
        # See also: https://stackoverflow.com/questions/49891686/cant-pickle-thread-lock-objects-pyspark-send-request-to-elasticseach
        column_path_map = {"indexer": {}, "encoder": {}}
        for key, value in self._indexer_per_column.items():
            column_path_map["indexer"][key] = (
                f"{PathsConfig.indexer_per_column_file_name_prefix}{key}"
            )
            value.save(os.path.join(directory_path, column_path_map["indexer"][key]))
        for key, value in self._encoder_per_column.items():
            column_path_map["encoder"][key] = (
                f"{PathsConfig.encoder_per_column_file_name_prefix}{key}"
            )
            value.save(os.path.join(directory_path, column_path_map["encoder"][key]))
        with open(
            os.path.join(directory_path, PathsConfig.column_path_map_file_name), "wb"
        ) as f:
            pickle.dump(column_path_map, f)

    def load_from_mlflow(self, run_id: str):  # pragma: no cover
        """
        Loads the state from mlflow from the artifacts of the Run of given RunID

        :raises RuntimeError: if experiment does not exist.
        :raises Exception: if given run_id is not in the expected mlflow experiment.
        """
        # WARNING: Beware that I realized in WSL, the temp directory is not cleaned as it should. This may result in
        # unexpected issues with loading and saving.
        with tempfile.TemporaryDirectory() as dir_name:
            mlflow_experiment = mlflow.get_experiment_by_name(
                run_config.experiment_name
            )
            if mlflow_experiment is None:
                raise RuntimeError(
                    f"Experiment {run_config.experiment_name} does not exist in MLFlow."
                )
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=run_config.run_name, dst_path=dir_name
            )

            with open(
                os.path.join(
                    dir_name,
                    run_config.run_name,
                    PathsConfig.mean_amount_all_categories_file_name,
                ),
                "rb",
            ) as f:
                self._mean_amount_all_categories = pickle.load(f)
            with open(
                os.path.join(
                    dir_name,
                    run_config.run_name,
                    PathsConfig.std_amount_all_categories_file_name,
                ),
                "rb",
            ) as f:
                self._std_amount_all_categories = pickle.load(f)
            with open(
                os.path.join(
                    dir_name,
                    run_config.run_name,
                    PathsConfig.mean_amount_per_category_file_name,
                ),
                "rb",
            ) as f:
                self._mean_amount_per_category = pickle.load(f)
            with open(
                os.path.join(
                    dir_name,
                    run_config.run_name,
                    PathsConfig.std_amount_per_category_file_name,
                ),
                "rb",
            ) as f:
                self._std_amount_per_category = pickle.load(f)

            with open(
                os.path.join(
                    dir_name, run_config.run_name, PathsConfig.column_path_map_file_name
                ),
                "rb",
            ) as f:
                column_path_map = pickle.load(f)

            # StringIndexerModel, and OneHotEncoderModel requires a spark session initialized to load.
            # Therefore we need to make sure there is an already active Spark session before we attempt to load.
            SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
            for key, value in column_path_map["indexer"].items():
                self._indexer_per_column[key] = StringIndexerModel.load(
                    os.path.join(dir_name, run_config.run_name, value)
                )
            for key, value in column_path_map["encoder"].items():
                self._encoder_per_column[key] = OneHotEncoderModel.load(
                    os.path.join(dir_name, run_config.run_name, value)
                )

    def save_to_mlflow(self, run_id: str):  # pragma: no cover
        """
        Saves the state as an artifact to mlflow, inside the given Run with RunID

        :raises RuntimeError: if experiment does not exist.
        :raises Exception: if given run_id is not in the expected mlflow experiment.
        """
        with tempfile.TemporaryDirectory() as dir_name:
            mlflow_experiment = mlflow.get_experiment_by_name(
                run_config.experiment_name
            )
            if mlflow_experiment is None:
                raise RuntimeError(
                    f"Experiment {run_config.experiment_name} does not exist in MLFlow."
                )
            self.save(dir_name)
            mlflow.log_artifacts(dir_name, run_config.run_name, run_id)


class FeatureExtractor:
    """
    Class for end-to-end feature extraction.

    Class that will store the objects for feature extraction that depends on training data distributions. These objects
    will be both used for feature extraction during training, but also in inference.

    As in inference, we need training data statistics/distributions, this object is required to be created during
    training feature calculations, and stored for later use in inference.
    """

    def __init__(self):
        self._state = _FeatureExtractorData()

    def save_to_mlflow(self, run_id: str):
        self._state.save_to_mlflow(run_id)

    def load_from_mlflow(self, run_id: str):
        self._state.load_from_mlflow(run_id)

    def get_features(self, data: DataFrame) -> DataFrame:
        """
        Entrypoint for feature calculation pipeline. Will calculate and return all features.

        For the features that requires state/data from training-time, it will first check if the required state/data
        is already stored in the class. If so, it will calculate features using those stored state. If not, it will
        assume that the dataset is the full trianin-set and set the mentioned states for later usage.

        This way, if the class is loaded in the inference-time (which we expect state to be there), it will use the
        state saved. If not, it will assume that the call is from training-time and set the state.
        """
        is_inference_time = self._state.is_set()
        return (
            self._get_features_inference(data)
            if is_inference_time
            else self._get_features_training(data)
        )

    def get_known_categories(self) -> typing.Optional[typing.Set[str]]:
        """
        Returns the values of known categories from the training data.
        Raises Value Error if the mean or std amount per category is not saved.
        """
        categories = set(self._state._mean_amount_per_category.keys()).union(
            set(self._state._std_amount_per_category.keys())
        )
        if not categories:
            # Most likely the state, mean or std amount per category is not saved. Instead of returning empty iterable,
            # better to crash here so that the error does not propagate further down in the program.
            raise ValueError("The feature extractor has no information on categories!")
        return categories

    def _get_features_training(self, data: DataFrame) -> DataFrame:
        """
        Feature extraction for training-time.
        """
        stats = data.groupby("category").agg(
            F.mean("amount").alias("mean"), F.stddev("amount").alias("std")
        )
        for row in stats.collect():
            self._set_mean_std_for_category(row.mean, row.std, row.category)
        data = self._get_z_score_per_category(data)

        self._set_mean_std_for_all_categories(
            data.select(F.mean(F.col("Amount"))).collect()[0][0],
            data.select(F.std(F.col("Amount"))).collect()[0][0],
        )
        data = self._get_z_score_all_categories(data)

        for col_name in run_config.categorical_cols:
            self._set_fit_string_indexer(data, col_name)
            data = self._get_string_indexed(data, col_name)

            self._set_fit_encoder(data, col_name)
            data = self._get_encoded(data, col_name)

            data = data.withColumn(
                "enc_" + col_name, vector_to_array(F.col("enc_" + col_name))
            )
            data = data.drop(F.col(col_name), F.col("idx_" + col_name))

        data = self._get_is_large_transaction(data)
        return data

    def _get_features_inference(self, data: DataFrame) -> DataFrame:
        """
        Feature extraction for inference-time.
        """
        data = self._get_z_score_per_category(data)
        data = self._get_z_score_all_categories(data)
        for col_name in run_config.categorical_cols:
            data = self._get_string_indexed(data, col_name)
            data = self._get_encoded(data, col_name)
            data = data.withColumn(
                "enc_" + col_name, vector_to_array(F.col("enc_" + col_name))
            )
            data = data.drop(F.col(col_name), F.col("idx_" + col_name))

        data = self._get_is_large_transaction(data)
        return data

    def _get_z_score_all_categories(self, data: DataFrame) -> DataFrame:
        """
        Calculates and returns z-score on the "amount" throughout the dataset as an added column "z_score_amount".

        WARNING: The statistics must be set in the object prior to usage.

        see also: set_mean_std_for_all_categories()

        :raises: ValueError: if statistics for all categories is not set
        (i.e. set_mean_std_for_all_categories not called with proper values).
        """
        if not (
            self._state._mean_amount_all_categories
            and self._state._std_amount_all_categories
        ):
            raise ValueError("Mean or std for overall amount is not set")

        return data.withColumn(
            "z_score_amount",
            (data["amount"] - self._state._mean_amount_all_categories)
            / self._state._std_amount_all_categories,
        )

    def _get_z_score_per_category(self, data: DataFrame) -> DataFrame:
        """
        Calculates and returns z-score on the "amount" per category as an added column "z_score_amount_by_cat".

        WARNING: The statistics must be set in the object prior to usage.

        see also: set_mean_std_for_category()

        :raises: Value Error: If data's categories differ from the categories that statistics are set.
        """
        # Another example why training and inference code must be the exact same implementation:
        # https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
        # F.stddev_pop("amount")

        data_categories = set(
            row.category for row in data.select(F.col("category")).distinct().collect()
        )
        expected_categories = set(self._state._mean_amount_per_category.keys()).union(
            set(self._state._std_amount_per_category.keys())
        )
        if not data_categories.issubset(expected_categories):
            raise ValueError(
                f"z_score cannot be calculated as expected categories differ "
                f"from passed data's categories: {data_categories} != {expected_categories}"
            )

        mean_list = [
            {"category": cat, "cat_mean": mean}
            for cat, mean in self._state._mean_amount_per_category.items()
        ]
        std_list = [
            {"category": cat, "cat_std": std}
            for cat, std in self._state._std_amount_per_category.items()
        ]

        spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
        mean_df = spark.createDataFrame(mean_list)
        std_df = spark.createDataFrame(std_list)

        data = data.join(mean_df, on="category")
        data = data.join(std_df, on="category")

        data = data.withColumn(
            "z_score_amount_by_cat",
            (data["amount"] - data["cat_mean"]) / data["cat_std"],
        )

        data = data.drop("cat_mean", "cat_std")
        return data

    def _set_mean_std_for_category(self, mean: float, std: float, category_name: str):
        """
        Stores the mean and std of the Amount for the passed category .
        """
        self._state._mean_amount_per_category[category_name] = mean
        self._state._std_amount_per_category[category_name] = std

    def _set_mean_std_for_all_categories(self, mean: float, std: float):
        """
        Stores the mean and std of the Amount for the whole distribution (regardless of category).
        """
        self._state._mean_amount_all_categories = mean
        self._state._std_amount_all_categories = std

    def _set_fit_string_indexer(self, data: DataFrame, col_name: str):
        """
        Fits using the passed data and stores the String Indexer for the passed column name
        """
        self._state._indexer_per_column[col_name] = StringIndexer(
            inputCol=col_name,
            outputCol="idx_" + col_name,
        ).fit(data)

    def _get_string_indexed(self, data: DataFrame, col_name: str) -> DataFrame:
        """
        Returns the data with a new indexed column for the passed data.

        WARNING: The col_name column should already be fit via the set_fit_string_indexer().

        see also: set_fit_string_indexer

        :raises ValueError: if the col_name is not fit with the string indexer.
        """
        if col_name not in self._state._indexer_per_column.keys():
            raise ValueError(f"{col_name} is not found in fit string indexer of {self}")

        return self._state._indexer_per_column[col_name].transform(data)

    def _set_fit_encoder(self, data: DataFrame, col_name: str) -> OneHotEncoderModel:
        """
        Fits using the passed data and stores the One Hot Encoder for the passed column name.

        WARNING: The column col_name in the passed data must already be indexed

        see also: set_fit_string_indexer
        """
        # This is an example why we need the same implementation in training and inference. Check dropLast in the signature.
        self._state._encoder_per_column[col_name] = OneHotEncoder(
            inputCol="idx_" + col_name,
            outputCol="enc_" + col_name,
        ).fit(data)

    def _get_encoded(self, data: DataFrame, col_name: str) -> DataFrame:
        """
        Returns the data with a new encoded column for the passed data.

        WARNING: The col_name column should already be fit via the set_fit_encoder.

        see also: set_fit_encoder

        :raises ValueError: if the col_name is not fit with the one hot encoder.
        """
        if col_name not in self._state._encoder_per_column.keys():
            raise ValueError(
                f"{col_name} is not found in fit one-hot encoder of {self}"
            )

        return self._state._encoder_per_column[col_name].transform(data)

    @staticmethod
    def _get_is_large_transaction(data: DataFrame) -> DataFrame:
        """
        Calculates the "is_large_transaction" column, which is a boolean flag indicating if a traction is large
        according to a business rule.

        :return: data with "is_large_transaction"
        """
        return data.withColumn(
            "is_large_transaction",
            F.col("amount")
            > run_config.threshold_in_eur_to_consider_large_transactions,
        )
