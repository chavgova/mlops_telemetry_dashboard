import os
import tempfile

import pytest
from mock import mock
import pyspark.sql.functions as F
from pyspark.ml.linalg import SparseVector
from pyspark.testing.utils import assertDataFrameEqual

from src.feature_extractor import _FeatureExtractorData, FeatureExtractor


def test_get_known_categories(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1", "amount": 123.1},
                    {"category": "foo_1", "amount": 123.2},
                    {"category": "foo_1", "amount": 345.3},
                    {"category": "foo_1", "amount": 111.4},
                    {"category": "foo_2", "amount": 10.0},
                    {"category": "foo_2", "amount": 20.0},
                ]
            ),
            "expected": {"foo_1", "foo_2"},
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        stats = (
            case["data"]
            .groupby("category")
            .agg(F.mean("amount").alias("mean"), F.stddev("amount").alias("std"))
        )
        for row in stats.collect():
            feature_extractor._set_mean_std_for_category(
                row.mean, row.std, row.category
            )
        out = feature_extractor.get_known_categories()
        assert out == case["expected"]


def test_feature_extractor_data_save():
    class MockPathsConfig:
        mean_amount_all_categories_file_name = "foo1"
        std_amount_all_categories_file_name = "foo2"
        mean_amount_per_category_file_name = "foo3"
        std_amount_per_category_file_name = "foo4"
        indexer_per_column_file_name_prefix = "foo_indexer"
        encoder_per_column_file_name_prefix = "foo_encoder"
        column_path_map_file_name = "foo_col_map"

    feature_extractor_data = _FeatureExtractorData()
    with mock.patch("src.feature_extractor.PathsConfig", MockPathsConfig()):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_extractor_data.save(tmpdir)
            assert {
                MockPathsConfig.mean_amount_all_categories_file_name,
                MockPathsConfig.std_amount_all_categories_file_name,
                MockPathsConfig.mean_amount_per_category_file_name,
                MockPathsConfig.std_amount_per_category_file_name,
                MockPathsConfig.column_path_map_file_name,
            } == set(os.listdir(tmpdir))


def test_feature_extractor_get_features_training_time(spark_fixture):
    with mock.patch(
        "src.feature_extractor.FeatureExtractor._get_features_training"
    ) as mock_get_feat:
        feature_extractor = FeatureExtractor()
        # Expecting FeatureExtractor to treat this as training-time as no state has been set
        feature_extractor.get_features(spark_fixture.createDataFrame([{}]))
        mock_get_feat.assert_called_once()

    with mock.patch(
        "src.feature_extractor.FeatureExtractor._get_features_training"
    ) as mock_get_feat:
        feature_extractor = FeatureExtractor()
        feature_extractor._mean_amount_all_categories = 999.0
        # Expecting FeatureExtractor to treat this as training-time as not all the state has been set
        feature_extractor.get_features(spark_fixture.createDataFrame([{}]))
        mock_get_feat.assert_called_once()


def test_feature_extractor_get_features_training(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1", "amount": 123.1},
                    {"category": "foo_1", "amount": 123.2},
                    {"category": "foo_1", "amount": 345.3},
                    {"category": "foo_1", "amount": 111.4},
                    {"category": "foo_2", "amount": 10.0},
                    {"category": "foo_2", "amount": 20.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {
                        "amount": 123.1,
                        "z_score_amount_by_cat": -0.4652335190351876,
                        "z_score_amount": 0.0077282538029900165,
                        "is_large_transaction": True,
                        "enc_category": [1.0],
                    },
                    {
                        "amount": 123.2,
                        "z_score_amount_by_cat": -0.46434988462106563,
                        "z_score_amount": 0.008556280996167598,
                        "is_large_transaction": True,
                        "enc_category": [1.0],
                    },
                    {
                        "amount": 345.3,
                        "z_score_amount_by_cat": 1.4982021491437048,
                        "z_score_amount": 1.8476046770434196,
                        "is_large_transaction": True,
                        "enc_category": [1.0],
                    },
                    {
                        "amount": 111.4,
                        "z_score_amount_by_cat": -0.5686187454874514,
                        "z_score_amount": -0.08915092779877867,
                        "is_large_transaction": False,
                        "enc_category": [1.0],
                    },
                    {
                        "amount": 10.0,
                        "z_score_amount_by_cat": -0.7071067811865475,
                        "z_score_amount": -0.9287705016807749,
                        "is_large_transaction": False,
                        "enc_category": [0.0],
                    },
                    {
                        "amount": 20.0,
                        "z_score_amount_by_cat": 0.7071067811865475,
                        "z_score_amount": -0.8459677823630238,
                        "is_large_transaction": False,
                        "enc_category": [0.0],
                    },
                ]
            ),
        }
    ]

    class MockRunConfig:
        threshold_in_eur_to_consider_large_transactions: float = 120.0
        categorical_cols: list = ["category"]
        spark_master_url: str = "local[2]"

    for case in cases:
        feature_extractor = FeatureExtractor()
        with mock.patch("src.feature_extractor.run_config", MockRunConfig()):
            out = feature_extractor._get_features_training(case["data"])
            assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_feature_extractor_get_z_score_all_categories(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1", "amount": 123.1},
                    {"category": "foo_1", "amount": 123.2},
                    {"category": "foo_1", "amount": 345.3},
                    {"category": "foo_1", "amount": 111.4},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {
                        "category": "foo_1",
                        "amount": 123.1,
                        "z_score_amount": -0.4652335190351876,
                    },
                    {
                        "category": "foo_1",
                        "amount": 123.2,
                        "z_score_amount": -0.46434988462106563,
                    },
                    {
                        "category": "foo_1",
                        "amount": 345.3,
                        "z_score_amount": 1.4982021491437048,
                    },
                    {
                        "category": "foo_1",
                        "amount": 111.4,
                        "z_score_amount": -0.5686187454874514,
                    },
                ]
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._set_mean_std_for_all_categories(
            case["data"].select(F.mean(F.col("Amount"))).collect()[0][0],
            case["data"].select(F.std(F.col("Amount"))).collect()[0][0],
        )
        out = feature_extractor._get_z_score_all_categories(case["data"])
        assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_feature_extractor_get_z_score_per_category(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1", "amount": 123.1},
                    {"category": "foo_1", "amount": 123.2},
                    {"category": "foo_1", "amount": 345.3},
                    {"category": "foo_1", "amount": 111.4},
                    {"category": "foo_2", "amount": 10.0},
                    {"category": "foo_2", "amount": 20.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {
                        "category": "foo_1",
                        "amount": 123.1,
                        "z_score_amount_by_cat": -0.4652335190351876,
                    },
                    {
                        "category": "foo_1",
                        "amount": 123.2,
                        "z_score_amount_by_cat": -0.46434988462106563,
                    },
                    {
                        "category": "foo_1",
                        "amount": 345.3,
                        "z_score_amount_by_cat": 1.4982021491437048,
                    },
                    {
                        "category": "foo_1",
                        "amount": 111.4,
                        "z_score_amount_by_cat": -0.5686187454874514,
                    },
                    {
                        "category": "foo_2",
                        "amount": 10.0,
                        "z_score_amount_by_cat": -0.7071067811865475,
                    },
                    {
                        "category": "foo_2",
                        "amount": 20.0,
                        "z_score_amount_by_cat": 0.7071067811865475,
                    },
                ]
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        stats = (
            case["data"]
            .groupby("category")
            .agg(F.mean("amount").alias("mean"), F.stddev("amount").alias("std"))
        )
        for row in stats.collect():
            feature_extractor._set_mean_std_for_category(
                row.mean, row.std, row.category
            )
        out = feature_extractor._get_z_score_per_category(case["data"])
        assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_feature_extractor_set_mean_std_for_category():
    feature_extractor = FeatureExtractor()
    feature_extractor._set_mean_std_for_category(1.0, 2.0, "foo")
    feature_extractor._set_mean_std_for_category(3.0, 4.0, "dodo")
    assert feature_extractor._state._mean_amount_per_category["foo"] == 1.0
    assert feature_extractor._state._mean_amount_per_category["dodo"] == 3.0
    assert feature_extractor._state._std_amount_per_category["foo"] == 2.0
    assert feature_extractor._state._std_amount_per_category["dodo"] == 4.0


def test_feature_extractor_set_mean_std_for_all_categories():
    feature_extractor = FeatureExtractor()
    feature_extractor._set_mean_std_for_all_categories(1.0, 2.0)
    assert feature_extractor._state._mean_amount_all_categories == 1.0
    assert feature_extractor._state._std_amount_all_categories == 2.0


def test_set_fit_string_indexed(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1"},
                    {"category": "foo_2"},
                    {"category": "foo_2"},
                    {"category": "foo_3"},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1", "idx_category": 1.0},
                    {"category": "foo_2", "idx_category": 0.0},
                    {"category": "foo_2", "idx_category": 0.0},
                    {"category": "foo_3", "idx_category": 2.0},
                ]
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._set_fit_string_indexer(case["data"], "category")
        assert (
            feature_extractor._state._indexer_per_column["category"].getOutputCol()
            == "idx_category"
        )


def test_get_string_indexed(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1"},
                    {"category": "foo_2"},
                    {"category": "foo_2"},
                    {"category": "foo_3"},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1", "idx_category": 1.0},
                    {"category": "foo_2", "idx_category": 0.0},
                    {"category": "foo_2", "idx_category": 0.0},
                    {"category": "foo_3", "idx_category": 2.0},
                ]
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._set_fit_string_indexer(case["data"], "category")
        out = feature_extractor._get_string_indexed(case["data"], "category")
        assertDataFrameEqual(
            out, case["expected"], ignoreNullable=True, ignoreColumnOrder=True
        )


def test_feature_extractor_set_fit_encoder(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1"},
                    {"category": "foo_2"},
                    {"category": "foo_2"},
                    {"category": "foo_3"},
                ]
            )
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._set_fit_string_indexer(case["data"], "category")
        data = feature_extractor._get_string_indexed(case["data"], "category")
        feature_extractor._set_fit_encoder(data, "category")
        assert (
            feature_extractor._state._encoder_per_column["category"].getOutputCol()
            == "enc_category"
        )


def test_feature_extractor_get_encoded(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"category": "foo_1"},
                    {"category": "foo_2"},
                    {"category": "foo_2"},
                    {"category": "foo_3"},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {
                        "category": "foo_1",
                        "enc_category": SparseVector(2, [1], [1.0]),
                        "idx_category": 1.0,
                    },
                    {
                        "category": "foo_2",
                        "enc_category": SparseVector(2, [0], [1.0]),
                        "idx_category": 0.0,
                    },
                    {
                        "category": "foo_2",
                        "enc_category": SparseVector(2, [0], [1.0]),
                        "idx_category": 0.0,
                    },
                    {
                        "category": "foo_3",
                        "enc_category": SparseVector(2, {}),
                        "idx_category": 2.0,
                    },
                ]
            ),
        }
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        feature_extractor._set_fit_string_indexer(case["data"], "category")
        data = feature_extractor._get_string_indexed(case["data"], "category")
        feature_extractor._set_fit_encoder(data, "category")
        out = feature_extractor._get_encoded(data, "category")
        assertDataFrameEqual(
            out, case["expected"], ignoreNullable=True, ignoreColumnOrder=True
        )


def test_feature_extractor_get_encoded_encoder_not_set(spark_fixture):
    feature_extractor = FeatureExtractor()
    with pytest.raises(ValueError):
        feature_extractor._get_encoded(None, "")


def test_feature_extractor_get_is_large_transaction(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"amount": 123.45},
                    {"amount": 9.123},
                    {"amount": 999.999},
                    {"amount": 10.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"amount": 123.45, "is_large_transaction": True},
                    {"amount": 9.123, "is_large_transaction": False},
                    {"amount": 999.999, "is_large_transaction": True},
                    {"amount": 10.0, "is_large_transaction": False},
                ]
            ),
        }
    ]

    class MockRunConfig:
        threshold_in_eur_to_consider_large_transactions: float = 10.0

    for case in cases:
        feature_extractor = FeatureExtractor()
        with mock.patch("src.feature_extractor.run_config", MockRunConfig()):
            out = feature_extractor._get_is_large_transaction(case["data"])
            assertDataFrameEqual(out, case["expected"])
