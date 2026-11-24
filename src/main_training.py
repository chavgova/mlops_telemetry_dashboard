import datetime
import json

import click
import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from mlflow.entities import RunStatus

from src.model import Model
from configs.configs import PathsConfig, run_config, ModelConfig
from src.feature_extractor import FeatureExtractor
from src.data_preprocessing import get_preprocessed_data, get_nested_column_extended
from src.mlflow_utils import (
    create_mlflow_experiment_if_not_exist,
    save_artifacts_to_mlflow,
    get_latest_run_id,
    create_mlflow_run_if_not_exists,
)


# note: Changed the main to multi-step process. See the "Fault Tolerance" and recovery at W3.
@click.command()
@click.option(
    "--preprocess", is_flag=True, help="Flag argument for running Preprocessing step."
)
@click.option(
    "--feat-eng",
    is_flag=True,
    help="Flag argument for running Feature Engineering step.",
)
@click.option(
    "--training", is_flag=True, help="Flag argument for running Training step."
)
def main(preprocess: bool, feat_eng: bool, training: bool):  # pragma: no cover
    if preprocess:
        data = get_preprocessed_data(datetime.date(year=2019, month=6, day=1))
        data = data.drop(F.col("chck_date"))
        data.write.parquet(PathsConfig.preprocessing_data_path, mode="overwrite")

    if feat_eng:
        spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
        data = spark.read.parquet(PathsConfig.preprocessing_data_path)
        feature_bundle = FeatureExtractor()
        data = feature_bundle.get_features(data)

        mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
        create_mlflow_experiment_if_not_exist()
        create_mlflow_run_if_not_exists(run_config.run_name)
        feature_bundle.save_to_mlflow(get_latest_run_id(run_config.run_name))
        data.write.parquet(PathsConfig.features_data_path, mode="overwrite")

    if training:
        spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
        data = spark.read.parquet(PathsConfig.features_data_path).toPandas()
        for col_name in ["enc_category", "enc_payment_type"]:
            data = get_nested_column_extended(data, col_name)

        model = Model()
        model.train_model(data)
        print(model.get_cv_scores())
        mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
        create_mlflow_experiment_if_not_exist()
        create_mlflow_run_if_not_exists(run_config.run_name)
        save_artifacts_to_mlflow(
            {ModelConfig.model_name: model, "cv_scores": model.get_cv_scores()},
            get_latest_run_id(run_config.run_name),
        )
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

        telemetry_data = {
            "is_large_transaction": {
                False: sum(data["is_large_transaction"] == False),
                True: sum(data["is_large_transaction"]),
            },
            "is_fraud": {False: sum(data["fraud"] == False), True: sum(data["fraud"])},
        }
        with open(PathsConfig.telemetry_training_data_path, "w") as file:
            json.dump(telemetry_data, file)


if __name__ == "__main__":  # pragma: no cover
    main()
