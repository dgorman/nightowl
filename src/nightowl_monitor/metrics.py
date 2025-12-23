"""Prometheus metrics integration for the NightOwl monitor."""

from __future__ import annotations

import time
from typing import Iterable, Optional, TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from .client import DeviceSnapshot

if TYPE_CHECKING:
    from .ml_leak_detector import MLLeakDetectionResult


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
        
        # ML Leak Detection Metrics
        self._ml_anomaly_score = Gauge(
            "nightowl_ml_anomaly_score",
            "ML model anomaly score (negative = anomaly, positive = normal)",
            labelnames=("device_id", "device_name"),
            registry=self.registry,
        )
        self._ml_leak_probability = Gauge(
            "nightowl_ml_leak_probability",
            "ML model leak probability (0-100%)",
            labelnames=("device_id", "device_name"),
            registry=self.registry,
        )
        self._ml_is_anomaly = Gauge(
            "nightowl_ml_is_anomaly",
            "ML model anomaly flag (1=anomaly detected, 0=normal)",
            labelnames=("device_id", "device_name"),
            registry=self.registry,
        )
        self._ml_model_confidence = Gauge(
            "nightowl_ml_model_confidence",
            "ML model confidence level (0=low, 1=medium, 2=high)",
            labelnames=("device_id", "device_name"),
            registry=self.registry,
        )
        self._ml_feature_value = Gauge(
            "nightowl_ml_feature_value",
            "ML model feature values (top contributors)",
            labelnames=("device_id", "device_name", "feature"),
            registry=self.registry,
        )
        self._ml_feature_contribution = Gauge(
            "nightowl_ml_feature_contribution",
            "ML feature contribution to anomaly score",
            labelnames=("device_id", "device_name", "feature"),
            registry=self.registry,
        )
        self._ml_inference_timestamp = Gauge(
            "nightowl_ml_inference_timestamp",
            "Unix timestamp of the last ML inference",
            labelnames=("device_id", "device_name"),
            registry=self.registry,
        )
        self._ml_inference_counter = Counter(
            "nightowl_ml_inferences_total",
            "Total ML inference runs",
            labelnames=("device_id", "status"),
            registry=self.registry,
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

    def record_ml_prediction(
        self,
        device_id: str,
        device_name: str,
        result: "MLLeakDetectionResult",
    ) -> None:
        """Record ML leak detection prediction metrics."""
        
        # Map confidence to numeric value
        confidence_map = {"low": 0, "medium": 1, "high": 2}
        confidence_value = confidence_map.get(result.confidence, 0)
        
        # Core ML metrics
        self._ml_anomaly_score.labels(
            device_id=device_id, device_name=device_name
        ).set(result.anomaly_score)
        
        self._ml_leak_probability.labels(
            device_id=device_id, device_name=device_name
        ).set(result.leak_probability * 100)
        
        self._ml_is_anomaly.labels(
            device_id=device_id, device_name=device_name
        ).set(1 if result.is_anomaly else 0)
        
        self._ml_model_confidence.labels(
            device_id=device_id, device_name=device_name
        ).set(confidence_value)
        
        self._ml_inference_timestamp.labels(
            device_id=device_id, device_name=device_name
        ).set(result.timestamp.timestamp())
        
        # Record top feature contributions
        for feature, contribution in result.feature_contributions.items():
            self._ml_feature_contribution.labels(
                device_id=device_id, device_name=device_name, feature=feature
            ).set(contribution)
        
        # Record feature values (limited to avoid cardinality explosion)
        for feature, value in list(result.raw_features.items())[:20]:
            self._ml_feature_value.labels(
                device_id=device_id, device_name=device_name, feature=feature
            ).set(value)
        
        # Increment inference counter
        self._ml_inference_counter.labels(
            device_id=device_id, status="success"
        ).inc()

    def record_ml_failure(self, device_id: str) -> None:
        """Record a failed ML inference."""
        self._ml_inference_counter.labels(
            device_id=device_id, status="failure"
        ).inc()


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
