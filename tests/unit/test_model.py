import pandas as pd

from src.model import Model


def test_model_get_x_y():
    cases = [
        {
            "data": pd.DataFrame({"fraud": [1, 0, 0, 1], "amount": [1, 1, 1, 1]}),
            "expected_x": pd.DataFrame({"amount": [1, 1, 1, 1]}),
            "expected_y": pd.Series([1, 0, 0, 1], copy=False, name="fraud"),
        }
    ]

    for case in cases:
        out_x, out_y = Model._get_x_y(case["data"])
        pd.testing.assert_frame_equal(out_x, case["expected_x"])
        pd.testing.assert_series_equal(out_y, case["expected_y"])
