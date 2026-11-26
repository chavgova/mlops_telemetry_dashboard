import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import prometheus_client
from prometheus_client import CollectorRegistry, Counter
from pyspark.sql import SparkSession


@pytest.fixture
def api(monkeypatch):
    registry = CollectorRegistry()
    spark = Mock()
    model = Mock()
    model.predict.return_value = np.array([False])
    extractor = Mock()
    extractor.get_known_categories.return_value = {"food"}
    extractor.get_features.return_value.toPandas.return_value = pd.DataFrame(
        {"amount": [10]}
    )

    def counter(*args, **kwargs):
        kwargs.setdefault("registry", registry)
        return Counter(*args, **kwargs)

    # mocking startup dependencies during import
    with (
        patch("src.model.Model.load_model_from_mlflow"),
        patch("src.feature_extractor.FeatureExtractor.load_from_mlflow"),
        patch("src.mlflow_utils.get_latest_run_id", return_value="test-run"),
        patch.object(SparkSession.Builder, "getOrCreate", return_value=spark),
        patch.object(prometheus_client, "REGISTRY", registry),
        patch.object(prometheus_client, "Counter", side_effect=counter),
    ):
        name = "src._monitoring_api_test"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[2] / "src/main_api.py"
        )
        
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        
        monkeypatch.setattr(module, "get_model", lambda: model)
        monkeypatch.setattr(module, "get_feature_extractor_loaded", lambda: extractor)
        monkeypatch.setattr( module, "get_nested_column_extended", lambda data, column: data )
        
        module.app.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
        
        yield module.app.test_client(), model, registry


@pytest.mark.parametrize("prediction,label", [(True, "true"), (False, "false")])
def test_success_records_prediction_and_http_metrics(api, prediction, label):
    client, model, registry = api
    model.predict.return_value = np.array([prediction])
    response = client.post( "/predict", json={"amount": 10, "category": "food", "payment_type": "1"} )
    assert response.status_code == 200
    assert response.get_json() is prediction
    
    model.predict.assert_called_once()
    
    assert registry.get_sample_value("nova_predictions_total", {"pred": label}) == 1
    assert ( registry.get_sample_value("nova_http_request_total",{"method": "POST","status":"200"})== 1)
    assert ( registry.get_sample_value( "nova_http_request_duration_seconds_count", 
                                       {"method": "POST", "path": "/predict", "status": "200"}, ) == 1 )


@pytest.mark.parametrize(
    "payload,rule",
    [
        ([], "invalid_payload"),
        ({"amount": 1}, "missing_fields"),
        ({"amount": None, "category": "food", "payment_type": "1"}, "missing_fields"),
        ({"amount": "10", "category": "food", "payment_type": "1"}, "invalid_amount"),
        ({"amount": True, "category": "food", "payment_type": "1"}, "invalid_amount"),
        (
            {"amount": float("inf"), "category": "food", "payment_type": "1"},
            "invalid_amount",
        ),
        ({"amount": 10, "category": "new", "payment_type": "1"}, "unknown_category"),
    ],
)
def test_bad_input_is_counted_without_prediction(api, payload, rule):
    client, model, registry = api
    assert client.post("/predict", json=payload).status_code == 422
    
    model.predict.assert_not_called()
    assert ( registry.get_sample_value("nova_data_quality_total", {"quality_rule": rule}) == 1 )


def test_malformed_json(api):
    client, model, registry = api
    assert ( client.post( "/predict", data="{bad", content_type="application/json" ).status_code == 422 )
    model.predict.assert_not_called()
    assert (registry.get_sample_value( "nova_data_quality_total", {"quality_rule": "invalid_payload"})==1)

def test_negative_amount_keeps_soft_warning(api):
    client, model, registry = api
    response = client.post( "/predict", json={"amount": -10, "category": "food", "payment_type": "1"} )
    assert response.status_code == 200
    model.predict.assert_called_once()
    
    assert ( registry.get_sample_value( "nova_data_quality_total", {"quality_rule": "neg_amount"})== 1)

def test_model_failure_is_counted_as_500(api):
    client, model, registry = api
    model.predict.side_effect = RuntimeError("Model failed")
    
    assert ( client.post( "/predict", json={"amount": 10, "category": "food", "payment_type": "1"} ).status_code == 500 )
    assert ( registry.get_sample_value( "nova_http_request_total", {"method": "POST", "status": "500"} ) == 1 )
    assert registry.get_sample_value("nova_predictions_total", {"pred": "false"}) == 0


def test_metrics_endpoint_and_excluded_health(api):
    client, blabla, registry = api   
    assert client.get("/health").status_code ==200   
    response = client.get("/metrics")
    
    assert response.status_code ==200      
    assert b"nova_predictions_total" in response.data
    assert ( registry.get_sample_value( "nova_http_request_total", {"method": "GET", "status": "200"} ) is None)
