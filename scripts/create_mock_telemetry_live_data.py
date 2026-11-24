from datetime import datetime

import pandas as pd

from configs.configs import PathsConfig


def main():
    predictions = pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, 13, 20),
                datetime(2025, 1, 1, 14, 12),
                datetime(2025, 1, 1, 17, 45),
                datetime(2025, 1, 2, 9, 1),
                datetime(2025, 1, 2, 1, 11),
            ],
            "is_large_transaction": [True, False, False, True, True],
            "is_fraud": [True, False, True, False, True],
        }
    )
    predictions.to_json(PathsConfig.telemetry_live_data_path)


if __name__ == "__main__":
    main()
