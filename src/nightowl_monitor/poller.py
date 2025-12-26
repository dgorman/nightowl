"""Long-running poller that refreshes NightOwl tokens periodically."""

from __future__ import annotations

import logging
import signal
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .client import ApiError, AuthError, DeviceSnapshot, NightOwlClient
from .config import Settings, SettingsError
from .metrics import MetricsService

_LOGGER = logging.getLogger("nightowl.monitor")

# ML imports (optional, graceful fallback if not installed)
try:
    import pandas as pd
    from .ml_leak_detector import MLLeakDetector, PrometheusDataFetcher
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    _LOGGER.warning("ML dependencies not available. Install scikit-learn, pandas, numpy for ML features.")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handle(signum: int, _frame: Optional[object]) -> None:
        _LOGGER.info("Received signal %s; shutting down gracefully", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)


def run_once(client: NightOwlClient, settings: Settings) -> List[DeviceSnapshot]:
    """Execute a single authentication cycle and return device snapshots."""

    tokens = client.authenticate(settings.username, settings.password, settings.tenant)
    masked = tokens.token[:8] + "..." if tokens.token else "<empty>"
    _LOGGER.info("Authenticated successfully; token prefix %s", masked)

    snapshots = client.fetch_latest_device_data(
        tokens.token,
        timeseries_keys=settings.timeseries_keys,
        attribute_keys=settings.attribute_keys,
        page_size=settings.page_size,
    )

    if not snapshots:
        _LOGGER.info("No devices returned by the NightOwl API")
        return []

    for snapshot in snapshots:
        report = _format_device_report(snapshot, settings.summary_keys)
        _LOGGER.info("%s", report)

    return snapshots


def run_polling_loop(settings: Settings) -> None:
    """Run the polling loop until interrupted."""

    _configure_logging()
    client = NightOwlClient(base_url=settings.base_url)
    metrics = MetricsService(
        host=settings.metrics_host,
        port=settings.metrics_port,
    )

    _LOGGER.info(
        "Prometheus metrics listening on %s:%s",
        settings.metrics_host,
        settings.metrics_port,
    )

    # Initialize ML detector if enabled
    ml_detector: Optional["MLLeakDetector"] = None
    data_fetcher: Optional["PrometheusDataFetcher"] = None
    poll_count = 0
    
    if settings.ml_enabled and ML_AVAILABLE:
        _LOGGER.info("ML leak detection enabled")
        model_path = Path(settings.ml_model_path) if settings.ml_model_path else None
        ml_detector = MLLeakDetector(model_path=model_path)
        
        # Try to load existing model
        if ml_detector.load():
            _LOGGER.info("Loaded pre-trained ML model")
            # Record model metadata in metrics
            stats = ml_detector.training_stats
            metrics.record_ml_model_info(
                trained_at=stats.get("trained_at"),
                training_samples=stats.get("training_samples"),
                model_version=stats.get("model_version"),
            )
        else:
            _LOGGER.warning("No pre-trained ML model found. Run training script to enable ML predictions.")
            ml_detector = None
            
        if settings.ml_prometheus_url:
            data_fetcher = PrometheusDataFetcher(prometheus_url=settings.ml_prometheus_url)
    elif settings.ml_enabled and not ML_AVAILABLE:
        _LOGGER.error("ML enabled but dependencies not installed. Run: pip install scikit-learn pandas numpy")

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    _LOGGER.info(
        "Starting NightOwl authentication poller (interval=%ss, base_url=%s)",
        settings.poll_interval_seconds,
        settings.base_url,
    )

    while not stop_event.is_set():
        try:
            snapshots = run_once(client, settings)
            metrics.record_success(snapshots)
            poll_count += 1
            
            # Run ML inference periodically (data_fetcher is optional)
            if (
                ml_detector is not None 
                and ml_detector.is_trained 
                and poll_count % settings.ml_inference_interval == 0
            ):
                _run_ml_inference(ml_detector, data_fetcher, snapshots, metrics)
                
        except AuthError as exc:
            metrics.record_failure()
            _LOGGER.error("Authentication attempt failed: %s", exc)
        except ApiError as exc:
            metrics.record_failure()
            _LOGGER.error("Failed to retrieve latest device data: %s", exc)
        except Exception:  # pragma: no cover - safety net for unexpected errors
            metrics.record_failure()
            _LOGGER.exception("Unexpected error while polling NightOwl")

        if stop_event.wait(settings.poll_interval_seconds):
            break

    _LOGGER.info("NightOwl poller stopped")


def _run_ml_inference(
    ml_detector: "MLLeakDetector",
    data_fetcher: Optional["PrometheusDataFetcher"],
    snapshots: List[DeviceSnapshot],
    metrics: MetricsService,
) -> None:
    """Run ML inference for each device."""
    for snapshot in snapshots:
        device_id = snapshot.device_id
        device_name = snapshot.name or ""
        
        if not device_id:
            continue
            
        try:
            # Try to fetch recent data from Prometheus for full feature calculation
            recent_data = None
            if data_fetcher is not None:
                try:
                    recent_data = data_fetcher.fetch_training_data(
                        device_id=device_id,
                        days=0.1,  # ~2.4 hours
                    )
                except Exception as e:
                    _LOGGER.debug(f"Could not fetch Prometheus data: {e}")
            
            # If no Prometheus data, create a simple dataframe from current telemetry
            if recent_data is None or recent_data.empty or len(recent_data) < 10:
                # Create a minimal dataframe from current snapshot
                telemetry = snapshot.timeseries or {}
                if not telemetry:
                    _LOGGER.debug(f"No telemetry for ML inference on {device_id}")
                    continue
                    
                recent_data = pd.DataFrame([{
                    "timestamp": datetime.utcnow(),
                    **{k: v for k, v in telemetry.items() if isinstance(v, (int, float))}
                }])
                _LOGGER.debug(f"Using real-time telemetry for ML inference")
            
            # Run prediction
            result = ml_detector.predict(recent_data)
            
            # Record metrics
            metrics.record_ml_prediction(device_id, device_name, result)
            
            if result.is_anomaly:
                _LOGGER.warning(
                    f"ML ANOMALY DETECTED for {device_name} ({device_id}): "
                    f"probability={result.leak_probability:.1%}, "
                    f"score={result.anomaly_score:.3f}"
                )
            else:
                _LOGGER.debug(
                    f"ML inference for {device_name}: normal "
                    f"(probability={result.leak_probability:.1%})"
                )
                
        except Exception as e:
            _LOGGER.error(f"ML inference failed for {device_id}: {e}")
            metrics.record_ml_failure(device_id)


def _build_summary(snapshot: DeviceSnapshot, summary_keys: Iterable[str]) -> Dict[str, Optional[str]]:
    summary: Dict[str, Optional[str]] = {}
    for key in summary_keys:
        value = snapshot.timeseries.get(key)
        if value is None:
            value = snapshot.attributes.get(key)
        summary[key] = None if value is None else str(value)
    return summary


def _all_empty(values: Dict[str, Optional[str]]) -> bool:
    return all(value in (None, "") for value in values.values()) if values else True


def _stringify_dict(values: Dict[str, object]) -> Dict[str, Optional[str]]:
    return {key: None if value is None else str(value) for key, value in values.items()}


def _format_device_report(snapshot: DeviceSnapshot, summary_keys: Iterable[str]) -> str:
    ordered_summary_keys = tuple(summary_keys)
    summary = _build_summary(snapshot, ordered_summary_keys)
    if _all_empty(summary):
        if snapshot.timeseries:
            summary = _stringify_dict(snapshot.timeseries)
        elif snapshot.attributes:
            summary = _stringify_dict(snapshot.attributes)
        else:
            summary = {}

    summary_section = _format_key_values("Summary", summary, ordered_summary_keys)
    telemetry_section = _format_key_values(
        "Telemetry", _stringify_dict(snapshot.timeseries)
    )
    attributes_section = _format_key_values(
        "Attributes", _stringify_dict(snapshot.attributes)
    )

    header = _format_header(snapshot)
    underline = "-" * len(header)

    sections = [header, underline, summary_section]
    if telemetry_section:
        sections.append("")
        sections.append(telemetry_section)
    if attributes_section:
        sections.append("")
        sections.append(attributes_section)

    return "\n".join(section for section in sections if section)


def _format_header(snapshot: DeviceSnapshot) -> str:
    if snapshot.name and snapshot.device_id:
        return f"Device: {snapshot.name} ({snapshot.device_id})"
    if snapshot.name:
        return f"Device: {snapshot.name}"
    if snapshot.device_id:
        return f"Device: {snapshot.device_id}"
    return "Device: <unknown>"


def _format_key_values(
    title: str,
    values: Dict[str, Optional[str]],
    preferred_order: Sequence[str] | None = None,
) -> str:
    if not values:
        return f"{title}: (none)"

    items = _ordered_items(values, preferred_order)
    if not items:
        return f"{title}: (none)"

    width = max(len(key) for key, _ in items)
    lines = [f"{title}:"]
    for key, value in items:
        display = _display_value(value)
        lines.append(f"  {key.ljust(width)} : {display}")
    return "\n".join(lines)


def _ordered_items(
    values: Dict[str, Optional[str]], preferred_order: Sequence[str] | None
) -> Tuple[Tuple[str, Optional[str]], ...]:
    items = []
    seen = set()
    if preferred_order:
        for key in preferred_order:
            if key in values:
                items.append((key, values[key]))
                seen.add(key)
    for key in sorted(values.keys()):
        if key not in seen:
            items.append((key, values[key]))
    return tuple(items)


def _display_value(value: Optional[str]) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    return text if text else "-"


def main() -> int:
    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        _configure_logging()
        _LOGGER.error("Failed to load settings: %s", exc)
        return 1

    run_polling_loop(settings)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
