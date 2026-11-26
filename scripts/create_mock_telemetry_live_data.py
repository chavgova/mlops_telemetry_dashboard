from datetime import datetime
import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

from configs.configs import PathsConfig, TelemetryConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", choices=("normal", "drift", "small", "stale"), default="normal" ) args = parser.parse_args()

    with open(PathsConfig.telemetry_training_data_path) as stream:
        reference = json.load(stream)["is_large_transaction"]

    size = TelemetryConfig.num_instances_for_live_dist

    if args.mode == "small":
        size = max(1, size // 2)

    # genetaring randomized mock flags 
    rate = 0.5 if args.mode == "drift" else reference["true"] / sum(reference.values())
    positives = round(size * rate)
    flags = [True] * positives + [False] * (size - positives)
    random.Random(88).shuffle(flags)
    
    now = datetime.now().timestamp() - ( 2 * TelemetryConfig.live_window_seconds if args.mode == "stale" else 0 )
    predictions = pd.DataFrame(
        {
            "timestamp": [int((now - i * 0.1) * 1000) for i in range(size)],
            "is_large_transaction": flags,
        }
    )

    path = Path(PathsConfig.telemetry_live_data_path)
    temporary = path.with_suffix(".tmp")
    predictions.to_json(temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
