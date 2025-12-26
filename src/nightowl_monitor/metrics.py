"""Prometheus metrics integration for the NightOwl monitor."""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .client import DeviceSnapshot

if TYPE_CHECKING:
    from .ml_leak_detector import MLLeakDetectionResult

logger = logging.getLogger(__name__)


class MetricsService:
    """Publishes NightOwl device data to Prometheus and handles LLM queries."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        registry: Optional[CollectorRegistry] = None,
        auto_start: bool = True,
        llm_handler: Optional[Callable[[str], dict]] = None,
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self._llm_handler = llm_handler
        self._host = host
        self._port = port
        
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
        self._ml_model_trained_at = Gauge(
            "nightowl_ml_model_trained_at",
            "Unix timestamp when the ML model was last trained",
            registry=self.registry,
        )
        self._ml_model_training_samples = Gauge(
            "nightowl_ml_model_training_samples",
            "Number of samples used to train the ML model",
            registry=self.registry,
        )
        self._ml_model_version = Gauge(
            "nightowl_ml_model_version_info",
            "ML model version (as info metric, value always 1)",
            labelnames=("version",),
            registry=self.registry,
        )
        
        # LLM query counter
        self._llm_query_counter = Counter(
            "nightowl_llm_queries_total",
            "Total LLM queries",
            labelnames=("status",),
            registry=self.registry,
        )

        if auto_start:
            self._start_server()
    
    def set_llm_handler(self, handler: Callable[[str], dict]) -> None:
        """Set the LLM handler function after initialization."""
        self._llm_handler = handler
    
    def _start_server(self) -> None:
        """Start the custom HTTP server with metrics and LLM endpoints."""
        metrics_service = self
        
        class NightOwlHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Suppress default logging
                pass
            
            def do_GET(self):
                if self.path == "/metrics" or self.path == "/":
                    self._handle_metrics()
                elif self.path == "/ask/status":
                    self._handle_llm_status()
                elif self.path.startswith("/health"):
                    self._handle_health()
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == "/ask":
                    self._handle_llm_query()
                else:
                    self.send_error(404)
            
            def _handle_metrics(self):
                output = generate_latest(metrics_service.registry)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(output)
            
            def _handle_health(self):
                response = {"status": "healthy", "timestamp": time.time()}
                self._send_json(response)
            
            def _handle_llm_status(self):
                if metrics_service._llm_handler is None:
                    response = {
                        "enabled": False,
                        "message": "LLM analyzer not configured"
                    }
                else:
                    try:
                        from .llm_analyzer import LLMConfig, NightOwlLLMAnalyzer
                        config = LLMConfig.from_env()
                        analyzer = NightOwlLLMAnalyzer(config)
                        status = analyzer.is_available()
                        response = {
                            "enabled": True,
                            "ollama_available": status["ollama"],
                            "grafana_cloud_configured": status["grafana_cloud"],
                            "model": config.ollama_model,
                        }
                    except Exception as e:
                        response = {
                            "enabled": True,
                            "error": str(e)
                        }
                self._send_json(response)
            
            def _handle_llm_query(self):
                if metrics_service._llm_handler is None:
                    metrics_service._llm_query_counter.labels(status="disabled").inc()
                    self._send_json({
                        "success": False,
                        "error": "LLM analyzer not configured"
                    }, status=503)
                    return
                
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length).decode("utf-8")
                    data = json.loads(body) if body else {}
                    question = data.get("question", "")
                    
                    result = metrics_service._llm_handler(question)
                    
                    if result.get("success"):
                        metrics_service._llm_query_counter.labels(status="success").inc()
                    else:
                        metrics_service._llm_query_counter.labels(status="failure").inc()
                    
                    self._send_json(result)
                except json.JSONDecodeError:
                    metrics_service._llm_query_counter.labels(status="error").inc()
                    self._send_json({
                        "success": False,
                        "error": "Invalid JSON"
                    }, status=400)
                except Exception as e:
                    metrics_service._llm_query_counter.labels(status="error").inc()
                    logger.exception("LLM query error")
                    self._send_json({
                        "success": False,
                        "error": str(e)
                    }, status=500)
            
            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        
        server = HTTPServer((self._host, self._port), NightOwlHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"NightOwl HTTP server started on {self._host}:{self._port}")

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

    def record_ml_model_info(
        self,
        trained_at: Optional[str] = None,
        training_samples: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Record ML model metadata when loaded."""
        if trained_at:
            try:
                from datetime import datetime
                # Parse ISO format timestamp
                if trained_at.endswith('Z'):
                    trained_at = trained_at[:-1]
                dt = datetime.fromisoformat(trained_at)
                self._ml_model_trained_at.set(dt.timestamp())
            except (ValueError, TypeError):
                pass
        if training_samples is not None:
            self._ml_model_training_samples.set(training_samples)
        if model_version:
            self._ml_model_version.labels(version=model_version).set(1)


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
