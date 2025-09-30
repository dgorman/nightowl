from prometheus_client import CollectorRegistry

from nightowl_monitor.client import DeviceSnapshot
from nightowl_monitor.metrics import MetricsService


def _get_metric_sample(registry: CollectorRegistry, name: str):
    for metric in registry.collect():
        if metric.name == name:
            return metric
    raise AssertionError(f"Metric {name} not found")


def test_metrics_service_records_success():
    registry = CollectorRegistry()
    service = MetricsService("127.0.0.1", 0, registry=registry, auto_start=False)

    snapshot = DeviceSnapshot(
        device_id="device-1",
        name="Pump 1",
        attributes={"active": "true", "Note": "ignored"},
        timeseries={"P1V1": "119.8", "S1": ""},
    )

    service.record_success([snapshot])

    success_total = registry.get_sample_value(
        "nightowl_poll_attempts_total", {"status": "success"}
    )
    assert success_total == 1.0

    telemetry = _get_metric_sample(registry, "nightowl_telemetry_value")
    telemetry_samples = {sample.labels["key"]: sample.value for sample in telemetry.samples}
    assert telemetry_samples["P1V1"] == 119.8
    assert "S1" not in telemetry_samples  # empty value skipped

    attributes = _get_metric_sample(registry, "nightowl_attribute_value")
    attribute_samples = {sample.labels["key"]: sample.value for sample in attributes.samples}
    assert attribute_samples["active"] == 1.0
    assert "Note" not in attribute_samples

    info = _get_metric_sample(registry, "nightowl_device_info")
    info_sample = info.samples[0]
    assert info_sample.labels["device_id"] == "device-1"
    assert info_sample.value == 1.0

    success_gauge = _get_metric_sample(registry, "nightowl_last_poll_success")
    assert success_gauge.samples[0].value == 1.0


def test_metrics_service_records_failure():
    registry = CollectorRegistry()
    service = MetricsService("127.0.0.1", 0, registry=registry, auto_start=False)

    service.record_failure()

    failure_total = registry.get_sample_value(
        "nightowl_poll_attempts_total", {"status": "failure"}
    )
    assert failure_total == 1.0

    success_gauge = _get_metric_sample(registry, "nightowl_last_poll_success")
    assert success_gauge.samples[0].value == 0.0
