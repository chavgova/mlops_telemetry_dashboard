import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import main_drift_calculation as batch

NOW = 1_700_000_000  # used for tests time


@pytest.fixture
def job(tmp_path, monkeypatch):
    training = tmp_path / "training.json"
    live = tmp_path / "live.json"
    training.write_text(json.dumps({"is_large_transaction": {"true": 50, "false": 50}}))
    monkeypatch.setattr(
        batch,
        "PathsConfig",
        SimpleNamespace(
            telemetry_training_data_path=str(training),
            telemetry_live_data_path=str(live),
        ),
    )
    monkeypatch.setattr(
        batch,
        "TelemetryConfig",
        SimpleNamespace(
            num_instances_for_live_dist=4,
            live_window_seconds=300,
            epsilon=1e-6,
            push_gateway_uri="http://unused:9091",
        ),
    )
    # mocking the gateway
    gateway = Mock()
    monkeypatch.setattr(batch, "push_to_gateway", gateway)

    def run(rows):
        live.write_text(json.dumps(rows))
        batch.main(now=NOW)
        return gateway.call_args.kwargs["registry"]

    return run, gateway, live


def row(age, flag):
    return {"timestamp": (NOW - age) * 1000, "is_large_transaction": flag}


def test_identical_distributions_have_zero_psi():
    assert batch.get_psi((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0)


def test_known_shift():
    assert batch.get_psi((0.9, 0.1), (0.1, 0.9)) == pytest.approx(1.6 * math.log(9))


def test_empty_bins_are_finite():
    assert math.isfinite(batch.get_psi((1, 0), (0, 1)))


@pytest.mark.parametrize(
    "reference,current",
    [
        ((0, 0), (1, 1)),
        ((-1, 2), (1, 1)),
        ((float("nan"), 1), (1, 1)),
        ((1, 1), (1,)),
    ],
)
def test_invalid_distributions(reference, current):
    with pytest.raises(ValueError):
        batch.get_psi(reference, current)


def test_latest_window_and_exported_metrics(job):
    run, gateway, bla = job
    registry = run(
        [
            row(100, True),
            row(2, True),
            row(0, False),
            row(3, False),
            row(1, True),
            row(400, True),
        ]
    )
    # asserting that the metric values are what is expected
    assert registry.get_sample_value("nova_monitoring_samples") == 4
    assert registry.get_sample_value(
        "nova_psi_s", {"target": "is_large_transaction"}
    ) == pytest.approx(0)
    assert registry.get_sample_value("nova_live_data_timestamp_seconds") == NOW
    assert (
        registry.get_sample_value("nova_monitoring_last_success_timestamp_seconds")
        == NOW
    )
    gateway.assert_called_once()
    assert gateway.call_args.args[0] == "http://unused:9091"
    assert gateway.call_args.kwargs["job"] == "telemetryBatch"


@pytest.mark.parametrize(  # small data sample for tests
    "rows,count,timestamp",
    [
        ([row(0, True)], 1, NOW),
        ([row(600, True)] * 4, 0, NOW - 600),
        ([], 0, 0),
    ],
)
def test_insufficient_or_stale_data(job, rows, count, timestamp):
    registry = job[0](rows)
    assert registry.get_sample_value("nova_monitoring_samples") == count
    assert registry.get_sample_value("nova_live_data_timestamp_seconds") == timestamp
    assert (
        registry.get_sample_value("nova_psi_s", {"target": "is_large_transaction"})
        is None
    )


def test_next_snapshot_removes_old_psi(job):
    run, bla, blabla = job
    first = run([row(0, True), row(1, False), row(2, True), row(3, False)])
    assert first.get_sample_value(
        "nova_psi_s", {"target": "is_large_transaction"}
    ) == pytest.approx(0)
    second = run([row(0, True)])
    assert (
        second.get_sample_value("nova_psi_s", {"target": "is_large_transaction"})
        is None
    )


@pytest.mark.parametrize(
    "rows", [[row(-1, True)], [row(0, "false")], [{"timestamp": NOW * 1000}]]
)
def test_bad_data_is_not_pushed(job, rows):
    run, gateway, blabla = job
    with pytest.raises(ValueError):
        run(rows)
    gateway.assert_not_called()


def test_push_failure_propagates(job):
    # making sure that if it fails, the problem is propagated and not being silenced
    run, gateway, blabla = job
    gateway.side_effect = ConnectionError("Unavailable")

    with pytest.raises(ConnectionError):
        run([row(0, True)])


def test_invalid_json_is_not_pushed(job):
    # this test helped me catch an issue
    blabla, gateway, live = job
    live.write_text("{bad")

    with pytest.raises(json.JSONDecodeError):
        batch.main(now=NOW)

    gateway.assert_not_called()
