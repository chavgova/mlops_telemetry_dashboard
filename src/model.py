import os
import pickle
import tempfile

import mlflow
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from configs.configs import ModelConfig, run_config


class Model:
    def __init__(self):
        self._model = None
        self._cv_scores = None

    @staticmethod
    def _get_model_object() -> RandomForestClassifier:
        return RandomForestClassifier(
            max_depth=ModelConfig.max_depth, class_weight=ModelConfig.class_weight
        )

    @staticmethod
    def _get_x_y(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return data.drop("fraud", axis=1), data["fraud"]

    def train_model(self, data: pd.DataFrame) -> None:  # pragma: no cover
        x, y = self._get_x_y(data)
        classifier = self._get_model_object()
        self._cv_scores = cross_validate(
            classifier,
            x,
            y,
            cv=run_config.num_folds,
            return_train_score=True,
            n_jobs=-1,
            scoring=["precision", "recall", "f1"],
        )
        self._model = classifier.fit(x, y)

    def get_cv_scores(self) -> dict[str, float]:
        return self._cv_scores

    def load_model_from_mlflow(self, run_id: str):  # pragma: no cover
        mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
        if mlflow_experiment is None:
            raise RuntimeError(
                f"Experiment {run_config.experiment_name} does not exist in MLFlow."
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                dst_path=temp_dir,
                tracking_uri=run_config.mlflow_tracking_uri,
            )
            artifacts_path = temp_dir + "/" + run_config.run_name
            model_file_name = ModelConfig.model_name + ".pkl"
            if model_file_name not in os.listdir(artifacts_path):
                raise RuntimeError(
                    f"Model {ModelConfig.model_name} is not among MLFLow artifacts."
                )
            with open(artifacts_path + "/" + model_file_name, "rb") as file:
                self._model = pickle.load(file)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(features)
