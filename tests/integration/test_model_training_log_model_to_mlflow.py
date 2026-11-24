# import mock
# import mlflow
# import time
# import numpy as np
#
# from configs.configs import run_config
# from src.model import Model
#
#
# def test_model_log_model_to_mlflow():
#     class MockModelConfig:
#         model_name = "donono"
#
#     class MockRunConfig:
#         mlflow_tracking_uri: str = "http://localhost:8080"
#         experiment_name: str = "TextExp"
#
#     with mock.patch("src.model.run_config", MockRunConfig):
#         with mock.patch("src.model.ModelConfig", MockModelConfig):
#             model = Model()
#             model._model = "foo"
#             model._cv_scores = {"test_f1": np.array([0.1, 0.2, 0.3])}
#             model.log_model_to_mlflow()
#
#     experiment = mlflow.get_experiment_by_name(MockRunConfig.experiment_name)
#     assert experiment
#
#     run_id = mlflow.search_runs(experiment.experiment_id).iloc[0]["run_id"]
#     run = mlflow.get_run(run_id)
#     artifacts = mlflow.artifacts.list_artifacts(run.info.artifact_uri)
#     assert artifacts[0].path in ["artifacts/cv_scores.json", "artifacts/donono"]
#     assert artifacts[1].path in ["artifacts/cv_scores.json", "artifacts/donono"]
#
#
# def test_model_load_model_from_mlflow():
#     class MockModelConfig:
#         model_name = str(time.time())
#
#     class MockRunConfig(run_config):
#         experiment_name: str = str(time.time())
#
#     with mock.patch("src.model.run_config", MockRunConfig):
#         with mock.patch("src.model.ModelConfig", MockModelConfig):
#             model = Model()
#             model._model = "foo"
#             model._cv_scores = {"test_f1": np.array([0.1, 0.2, 0.3])}
#             model.log_model_to_mlflow()
#
#             model_to_load = Model()
#             model_to_load.load_model_from_mlflow()
#             assert model_to_load._model == "foo"
