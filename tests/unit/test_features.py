# import mock
# import pytest
# import pandas as pd
# from pyspark.testing.utils import assertDataFrameEqual
#
# from src.features import (
#     get_nested_column_extended,
#     _get_encoded_features,
#     _get_z_score,
#     get_features,
# )
#
#
#
# def test_get_encoded_features(spark_fixture):
#     cases = [
#         {
#             "data": spark_fixture.createDataFrame(
#                 [
#                     {"category": "foo"},
#                     {"category": "bar"},
#                     {"category": "foo"},
#                     {"category": "baz"},
#                     {"category": "foo"},
#                 ]
#             ),
#             "expected": spark_fixture.createDataFrame(
#                 [
#                     {"enc_category": [1.0, 0.0]},
#                     {"enc_category": [0.0, 1.0]},
#                     {"enc_category": [1.0, 0.0]},
#                     {"enc_category": [0.0, 0.0]},
#                     {"enc_category": [1.0, 0.0]},
#                 ]
#             ),
#         }
#     ]
#     for case in cases:
#         out = _get_encoded_features(case["data"], "category")
#         assertDataFrameEqual(out, case["expected"])

# def test_get_z_score(spark_fixture):
#     cases = [
#         {
#             "data": spark_fixture.createDataFrame(
#                 [
#                     {"amount": 1.1},
#                     {"amount": 0.2},
#                     {"amount": -11.56},
#                     {"amount": 23.43},
#                     {"amount": 67.99},
#                 ]
#             ),
#             "group_by_cols": [],
#             "expected": spark_fixture.createDataFrame(
#                 [
#                     {"amount": 1.1, "z_score": -0.47916920013376096},
#                     {"amount": 0.2, "z_score": -0.5076685577943733},
#                     {"amount": -11.56, "z_score": -0.8800601645597069},
#                     {"amount": 23.43, "z_score": 0.2279315293789857},
#                     {"amount": 67.99, "z_score": 1.6389663931088554},
#                 ]
#             ),
#         },
#         {
#             "data": spark_fixture.createDataFrame(
#                 [
#                     {"amount": 1.1, "category": "foo"},
#                     {"amount": 0.2, "category": "foo"},
#                     {"amount": -11.56, "category": "bar"},
#                     {"amount": 23.43, "category": "bar"},
#                     {"amount": 67.99, "category": "foo"},
#                 ]
#             ),
#             "group_by_cols": ["category"],
#             "expected": spark_fixture.createDataFrame(
#                 [
#                     {
#                         "amount": 1.1,
#                         "category": "foo",
#                         "z_score_by_category": -0.5657379340615589,
#                     },
#                     {
#                         "amount": 0.2,
#                         "category": "foo",
#                         "z_score_by_category": -0.5888852658082814,
#                     },
#                     {
#                         "amount": -11.56,
#                         "category": "bar",
#                         "z_score_by_category": -0.7071067811865475,
#                     },
#                     {
#                         "amount": 23.43,
#                         "category": "bar",
#                         "z_score_by_category": 0.7071067811865475,
#                     },
#                     {
#                         "amount": 67.99,
#                         "category": "foo",
#                         "z_score_by_category": 1.1546231998698404,
#                     },
#                 ]
#             ),
#         },
#     ]
#
#     for case in cases:
#         out = _get_z_score(case["data"], case["group_by_cols"])
#         assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)
#
#
# def test_get_features(spark_fixture):
#     cases = [
#         {
#             "data": spark_fixture.createDataFrame(
#                 [
#                     {"amount": 1.1, "category": "foo", "payment_type": "donono"},
#                     {"amount": 0.2, "category": "foo", "payment_type": "abc"},
#                     {"amount": -11.56, "category": "bar", "payment_type": "donono"},
#                     {"amount": 23.43, "category": "bar", "payment_type": "abc"},
#                     {"amount": 67.99, "category": "foo", "payment_type": "donono"},
#                 ]
#             ),
#             "expected": spark_fixture.createDataFrame(
#                 [
#                     {
#                         "amount": 1.1,
#                         "z_score_by_category": -0.5657379340615589,
#                         "z_score": -0.47916920013376096,
#                         "enc_category": [1.0],
#                         "enc_payment_type": [1.0],
#                         "is_large_transaction": False,
#                     },
#                     {
#                         "amount": 0.2,
#                         "z_score_by_category": -0.5888852658082814,
#                         "z_score": -0.5076685577943733,
#                         "enc_category": [1.0],
#                         "enc_payment_type": [0.0],
#                         "is_large_transaction": False,
#                     },
#                     {
#                         "amount": -11.56,
#                         "z_score_by_category": -0.7071067811865475,
#                         "z_score": -0.8800601645597069,
#                         "enc_category": [0.0],
#                         "enc_payment_type": [1.0],
#                         "is_large_transaction": False,
#                     },
#                     {
#                         "amount": 23.43,
#                         "z_score_by_category": 0.7071067811865475,
#                         "z_score": 0.2279315293789857,
#                         "enc_category": [0.0],
#                         "enc_payment_type": [0.0],
#                         "is_large_transaction": False,
#                     },
#                     {
#                         "amount": 67.99,
#                         "z_score_by_category": 1.1546231998698404,
#                         "z_score": 1.6389663931088554,
#                         "enc_category": [1.0],
#                         "enc_payment_type": [1.0],
#                         "is_large_transaction": True,
#                     },
#                 ]
#             ),
#         }
#     ]
#     for case in cases:
#
#         class MockRunConfig:
#             threshold_in_eur_to_consider_large_transactions = 45.0
#
#         with mock.patch("src.features.run_config", MockRunConfig):
#             out = get_features(case["data"])
#             assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)
