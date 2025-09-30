from nightowl_monitor.client import DeviceSnapshot
from nightowl_monitor.poller import _format_device_report


def test_format_device_report_with_summary_and_payloads():
    snapshot = DeviceSnapshot(
        device_id="device-1",
        name="Pump 1",
        attributes={"CustomerInfo": "Dan Gorman", "active": "true"},
        timeseries={"P1V1": "119.8", "Pulse_TotalGallons": ""},
    )

    report = _format_device_report(snapshot, ["P1V1", "Pulse_TotalGallons"])

    assert "Device: Pump 1 (device-1)" in report
    assert "Summary:" in report
    assert "P1V1" in report and "119.8" in report
    assert "Pulse_TotalGallons" in report and "-" in report
    assert "Telemetry:" in report and "P1V1" in report
    assert "Attributes:" in report and "CustomerInfo" in report


def test_format_device_report_falls_back_when_summary_empty():
    snapshot = DeviceSnapshot(
        device_id="device-2",
        name=None,
        attributes={},
        timeseries={"S1": "12"},
    )

    report = _format_device_report(snapshot, ["UnknownKey"])

    assert "Device: device-2" in report
    assert "Summary:" in report and "S1" in report
    assert "Telemetry:" in report and "S1" in report


def test_format_device_report_handles_missing_payloads():
    snapshot = DeviceSnapshot(
        device_id=None,
        name=None,
        attributes={},
        timeseries={},
    )

    report = _format_device_report(snapshot, [])

    assert "Device: <unknown>" in report
    assert "Summary: (none)" in report
    assert "Telemetry: (none)" in report
    assert "Attributes: (none)" in report
