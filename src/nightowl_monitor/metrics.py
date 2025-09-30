"""Prometheus metrics integration for the NightOwl monitor."""

from __future__ import annotations

import time
from typing import Iterable, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from .client import DeviceSnapshot


class MetricsService:
    """Publishes NightOwl device data to Prometheus."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        registry: Optional[CollectorRegistry] = None,
        auto_start: bool = True,
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self._telemetry_gauge = Gauge(
            "nightowl_telemetry_value",
            "Numeric NightOwl telemetry values",
            labelnames=("device_id", "device_name", "key"),
            registry=self.registry,
        )
        self._attribute_gauge = Gauge(
            "nightowl_attribute_value",
            "Numeric NightOwl attribute values",
            labelnames=("device_id", "device_name", "key"),
            registry=self.registry,
        )
        self._device_info_gauge = Gauge(
            "nightowl_device_info",
            "NightOwl device presence flag",
            labelnames=("device_id", "device_name"),
            registry=self.registry,
        )
        self._poll_success_gauge = Gauge(
            "nightowl_last_poll_success",
            "Whether the most recent poll succeeded (1=success, 0=failure)",
            registry=self.registry,
        )
        self._poll_timestamp_gauge = Gauge(
            "nightowl_last_poll_timestamp",
            "Unix timestamp of the most recent successful poll",
            registry=self.registry,
        )
        self._poll_counter = Counter(
            "nightowl_poll_attempts",
            "Total NightOwl poll attempts", labelnames=("status",), registry=self.registry
        )

        if auto_start:
            start_http_server(port, addr=host, registry=self.registry)

    def record_success(self, snapshots: Iterable[DeviceSnapshot]) -> None:
        """Record a successful poll and update telemetry/attribute gauges."""

        now = time.time()
        self._poll_counter.labels("success").inc()
        self._poll_success_gauge.set(1)
        self._poll_timestamp_gauge.set(now)

        for snapshot in snapshots:
            device_id = snapshot.device_id or "unknown"
            device_name = snapshot.name or ""
            self._device_info_gauge.labels(device_id=device_id, device_name=device_name).set(1)

            for key, value in snapshot.timeseries.items():
                numeric = _to_float(value)
                if numeric is None:
                    continue
                self._telemetry_gauge.labels(
                    device_id=device_id, device_name=device_name, key=key
                ).set(numeric)

            for key, value in snapshot.attributes.items():
                numeric = _to_float(value)
                if numeric is None:
                    continue
                self._attribute_gauge.labels(
                    device_id=device_id, device_name=device_name, key=key
                ).set(numeric)

    def record_failure(self) -> None:
        """Record an unsuccessful poll."""

        self._poll_counter.labels("failure").inc()
        self._poll_success_gauge.set(0)


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        if text.lower() in {"true", "false"}:
            return 1.0 if text.lower() == "true" else 0.0
    return None
