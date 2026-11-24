import datetime
import tempfile

import mock
import pandas as pd
import pytest
from pyspark.testing.utils import assertDataFrameEqual

from src.data_preprocessing import (
    _get_data_after,
    get_preprocessed_data,
    get_nested_column_extended,
)


def test_get_data_after(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {
                "amount": 1,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2020, 1, 1, 1, 1),
            },
            {
                "amount": 2,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 1, 1, 1, 1),
            },
            {
                "amount": 3,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 2, 1, 1, 1),
            },
            {
                "amount": 4,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(
                    2021,
                    1,
                    2,
                ),
            },
        ]
    )
    expected = spark_fixture.createDataFrame(
        [
            {
                "amount": 2,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 1, 1, 1, 1),
            },
            {
                "amount": 3,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 2, 1, 1, 1),
            },
            {
                "amount": 4,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(
                    2021,
                    1,
                    2,
                ),
            },
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        class MockPathsConfig:
            raw_data_path = tmpdir + "/donono"

        with mock.patch("src.data_preprocessing.PathsConfig", MockPathsConfig):
            data.write.parquet(MockPathsConfig.raw_data_path)

            out = _get_data_after(datetime.date(2021, 1, 1))
            assertDataFrameEqual(out, expected, ignoreColumnOrder=True)


def test_get_preprocessed_data(spark_fixture):
    # Am I actually testing something with this test, or am I being lazy trying to increase my coverage?
    data = spark_fixture.createDataFrame(
        [
            {
                "amount": 1,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2020, 1, 1, 1, 1),
            },
            {
                "amount": 2,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 1, 1, 1, 1),
            },
            {
                "amount": 3,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 2, 1, 1, 1),
            },
            {
                "amount": 4,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(
                    2021,
                    1,
                    2,
                ),
            },
        ]
    )
    expected = spark_fixture.createDataFrame(
        [
            {
                "amount": 2,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 1, 1, 1, 1),
            },
            {
                "amount": 3,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(2021, 2, 1, 1, 1),
            },
            {
                "amount": 4,
                "payment_type": 1,
                "category": 1,
                "fraud": 1,
                "chck_date": datetime.datetime(
                    2021,
                    1,
                    2,
                ),
            },
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        class MockPathsConfig:
            raw_data_path = tmpdir + "/donono"

        with mock.patch("src.data_preprocessing.PathsConfig", MockPathsConfig):
            data.write.parquet(MockPathsConfig.raw_data_path)

            out = get_preprocessed_data(datetime.date(2021, 1, 1))
            assertDataFrameEqual(out, expected, ignoreColumnOrder=True)


def test_get_nested_column_extended():
    cases = [
        {
            "data": pd.DataFrame(
                {"foo": [1, 2, 3], "listed_col": [[1, 2], [3, 4], [5, 6]]}
            ),
            "col_name": "listed_col",
            "expected": pd.DataFrame(
                {
                    "foo": [1, 2, 3],
                    "listed_col_0": [1, 3, 5],
                    "listed_col_1": [2, 4, 6],
                }
            ),
        },
        {
            "data": pd.DataFrame(
                {
                    "foo": [1, 2, 3],
                    "listed_col": [[1, 2, "x"], [3, 4, "y"], [5, 6, "z"]],
                }
            ),
            "col_name": "listed_col",
            "expected": pd.DataFrame(
                {
                    "foo": [1, 2, 3],
                    "listed_col_0": [1, 3, 5],
                    "listed_col_1": [2, 4, 6],
                    "listed_col_2": ["x", "y", "z"],
                }
            ),
        },
    ]

    for case in cases:
        out = get_nested_column_extended(case["data"], case["col_name"])
        pd.testing.assert_frame_equal(out, case["expected"], check_like=True)


def test_get_nested_column_extended_nonuniform_length():
    cases = [
        {
            "data": pd.DataFrame(
                {"foo": [1, 2, 3], "listed_col": [[1], [3, 4, "y"], [5, 6, "z"]]}
            ),
            "col_name": "listed_col",
            "expected": pd.DataFrame(
                {
                    "foo": [1, 2, 3],
                    "listed_col_0": [1, 3, 5],
                    "listed_col_1": [2, 4, 6],
                    "listed_col_2": ["x", "y", "z"],
                }
            ),
        }
    ]

    for case in cases:
        with pytest.raises(ValueError):
            get_nested_column_extended(case["data"], case["col_name"])
