"""Long-running poller that refreshes NightOwl tokens periodically."""

from __future__ import annotations

import logging
import signal
import sys
import threading
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .client import ApiError, AuthError, DeviceSnapshot, NightOwlClient
from .config import Settings, SettingsError

_LOGGER = logging.getLogger("nightowl.monitor")


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


def run_once(client: NightOwlClient, settings: Settings) -> None:
    """Execute a single authentication cycle."""

    tokens = client.authenticate(settings.username, settings.password, settings.tenant)
    masked = tokens.token[:8] + "..." if tokens.token else "<empty>"
    _LOGGER.info("Authenticated successfully; token prefix %s", masked)

    try:
        snapshots = client.fetch_latest_device_data(
            tokens.token,
            timeseries_keys=settings.timeseries_keys,
            attribute_keys=settings.attribute_keys,
            page_size=settings.page_size,
        )
    except ApiError as exc:
        _LOGGER.error("Failed to retrieve latest device data: %s", exc)
        return

    if not snapshots:
        _LOGGER.info("No devices returned by the NightOwl API")
        return

    for snapshot in snapshots:
        report = _format_device_report(snapshot, settings.summary_keys)
        _LOGGER.info("%s", report)


def run_polling_loop(settings: Settings) -> None:
    """Run the polling loop until interrupted."""

    _configure_logging()
    client = NightOwlClient(base_url=settings.base_url)

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    _LOGGER.info(
        "Starting NightOwl authentication poller (interval=%ss, base_url=%s)",
        settings.poll_interval_seconds,
        settings.base_url,
    )

    while not stop_event.is_set():
        try:
            run_once(client, settings)
        except AuthError as exc:
            _LOGGER.error("Authentication attempt failed: %s", exc)
        except Exception:  # pragma: no cover - safety net for unexpected errors
            _LOGGER.exception("Unexpected error while polling NightOwl")

        if stop_event.wait(settings.poll_interval_seconds):
            break

    _LOGGER.info("NightOwl poller stopped")


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
