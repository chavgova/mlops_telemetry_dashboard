"""
Entrypoint for calculating drift monitoring metrics for Prometheus.
"""

import json
import math
import logging
import typing
import time

import pandas as pd
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

from configs.configs import PathsConfig, TelemetryConfig, run_config

logger = logging.getLogger(__name__)


def get_psi(
    training_percentages: typing.Tuple[float], latest_percentages: typing.Tuple[float]
) -> float:
    if len(training_percentages) != len(latest_percentages):
        raise ValueError("PSI distributions must use the same bins")

    def normalize(values):
        if (
            not values
            or any(not math.isfinite(value) or value < 0 for value in values)
            or sum(values) <= 0
        ):
            raise ValueError(
                "PSI requires finite nonnegative weights with a positive total"
            )
        probabilities = [
            max(value / sum(values), TelemetryConfig.epsilon) for value in values
        ]
        return tuple(value / sum(probabilities) for value in probabilities)

    training_percentages = normalize(training_percentages)
    latest_percentages = normalize(latest_percentages)
    psi = 0
    for training, latest in zip(training_percentages, latest_percentages):
        psi += (latest - training) * math.log(latest / training, math.e)
    return psi


def main(now=None):  # pragma: no cover
    with open(PathsConfig.telemetry_training_data_path, "r") as file:
        training_telemetry_data = json.load(file)
    with open(PathsConfig.telemetry_live_data_path, "r") as file:
        live_telemetry_data = json.load(file)
        live_telemetry_data = pd.DataFrame.from_records(live_telemetry_data)

    now = time.time() if now is None else now
    if live_telemetry_data.empty:
        live_telemetry_data = pd.DataFrame(
            columns=["timestamp", "is_large_transaction"]
        )
    if not {"timestamp", "is_large_transaction"}.issubset(live_telemetry_data.columns):
        raise ValueError("Live data requires timestamp and is_large_transaction")
    timestamps = pd.to_numeric(live_telemetry_data["timestamp"], errors="raise") / 1000
    if any(
        not math.isfinite(value) or value < 0 or value > now for value in timestamps
    ):
        raise ValueError("Invalid live timestamps")
    if not live_telemetry_data["is_large_transaction"].isin([True, False]).all():
        raise ValueError("Large-transaction flags must be Boolean or binary")
    latest_timestamp = float(timestamps.max()) if len(timestamps) else 0
    live_telemetry_data = live_telemetry_data[
        timestamps >= now - TelemetryConfig.live_window_seconds
    ].copy()
    live_telemetry_data.sort_values("timestamp", inplace=True, ascending=False)
    latest_telemetry_data = live_telemetry_data.iloc[
        0 : TelemetryConfig.num_instances_for_live_dist
    ]

    telemetry_data_count = latest_telemetry_data.shape[0]
    targets = ["is_large_transaction"]
    if telemetry_data_count < TelemetryConfig.num_instances_for_live_dist:
        logger.warning(
            f"Telemetry calculation has {telemetry_data_count} which is less than "
            f"required {TelemetryConfig.num_instances_for_live_dist} - skipping PSI"
        )
        targets = []

    psi_s = {}
    for target in targets:
        training_count = (
            training_telemetry_data[target]["true"]
            + training_telemetry_data[target]["false"]
        )
        # Adding Epsilon to avoid division by zero err for when there is no instance of a category.
        # smoothing is applied in get_psi
        training_percentages = (
            training_telemetry_data[target]["true"] / training_count,
            training_telemetry_data[target]["false"] / training_count,
        )

        latest_count = latest_telemetry_data[target].shape[0]
        # Adding Epsilon to avoid division by zero err for when there is no instance of a category.
        #  smoothing is applied in get_psi
        latest_percentages = (
            sum(latest_telemetry_data[target]) / latest_count,
            ((latest_count - sum(latest_telemetry_data[target])) / latest_count),
        )

        psi_s[target] = get_psi(training_percentages, latest_percentages)

    registry = CollectorRegistry()
    for name, description, value in (
        ("monitoring_samples", "Recent monitoring records", telemetry_data_count),
        ("live_data_timestamp_seconds", "Newest source event time", latest_timestamp),
        (
            "monitoring_last_success_timestamp_seconds",
            "Last successful monitoring snapshot",
            now,
        ),
    ):
        Gauge(f"{run_config.app_name}_{name}", description, registry=registry).set(
            value
        )
    psi_gauge = Gauge(
        f"{run_config.app_name}_psi_s",
        "PSI calculations for feature and output data",
        labelnames=["target"],
        registry=registry,
    )
    for target in targets:
        psi_gauge.labels(target=target).set(psi_s[target])

    push_to_gateway(
        TelemetryConfig.push_gateway_uri,
        job="telemetryBatch",
        registry=registry,
        timeout=5,
    )


if __name__ == "__main__":
    main()
