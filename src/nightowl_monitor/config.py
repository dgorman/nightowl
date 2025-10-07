"""Configuration helpers for the NightOwl monitor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

DEFAULT_TIMESERIES_KEYS: Tuple[str, ...] = (
    "Well_Level-1",
    "S1",
    "S2",
    "Level1_precent",
    "P1V1",
    "P1V2",
    "P1V3",
    "P1C1",
    "P1C2",
    "P1C3",
    "P1Hz",
    "P2V1",
    "P2V2",
    "P2V3",
    "P2C1",
    "P2C2",
    "P2C3",
    "P2Hz",
    "P3V1",
    "P3V2",
    "P3V3",
    "P3C1",
    "P3C2",
    "P3C3",
    "P3Hz",
    "Pulse_TotalGallons",
)

DEFAULT_ATTRIBUTE_KEYS: Tuple[str, ...] = (
    "CustomerInfo",
    "Location",
    "SiteInfo",
    "active",
    "edgedata_status",
    "S1_name",
    "S2_name",
    "Level1_Name",
)

DEFAULT_SUMMARY_KEYS: Tuple[str, ...] = (
    "Well_Level-1",
    "Level1_precent",
    "Pulse_TotalGallons",
)


class SettingsError(RuntimeError):
    """Raised when configuration could not be loaded."""


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    username: str
    password: str
    tenant: Optional[str]
    base_url: str
    poll_interval_seconds: int
    page_size: int
    timeseries_keys: Tuple[str, ...]
    attribute_keys: Tuple[str, ...]
    summary_keys: Tuple[str, ...]
    metrics_host: str
    metrics_port: int

    @staticmethod
    def from_env(env: Optional[dict[str, str]] = None) -> "Settings":
        """Create settings from environment variables."""

        source = env or os.environ

        def required(name: str) -> str:
            value = source.get(name)
            if value is None or not value.strip():
                raise SettingsError(f"Environment variable {name} must be set and non-empty")
            return value.strip()

        username = required("NIGHTOWL_USERNAME")
        password = required("NIGHTOWL_PASSWORD")
        tenant = source.get("NIGHTOWL_TENANT")
        base_url = source.get("NIGHTOWL_BASE_URL", "https://portal.nightowlmonitoring.com").rstrip("/")

        interval_raw = source.get("NIGHTOWL_POLL_INTERVAL_SECONDS", "60")
        try:
            interval = int(interval_raw)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise SettingsError(
                "NIGHTOWL_POLL_INTERVAL_SECONDS must be an integer"
            ) from exc

        if interval < 10:
            raise SettingsError("NIGHTOWL_POLL_INTERVAL_SECONDS must be at least 10 seconds")

        page_size_raw = source.get("NIGHTOWL_PAGE_SIZE", "5")
        try:
            page_size = int(page_size_raw)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise SettingsError("NIGHTOWL_PAGE_SIZE must be an integer") from exc

        if page_size < 1:
            raise SettingsError("NIGHTOWL_PAGE_SIZE must be at least 1")

        timeseries_keys = _parse_key_list(
            source.get("NIGHTOWL_TIMESERIES_KEYS"), DEFAULT_TIMESERIES_KEYS
        )
        attribute_keys = _parse_key_list(
            source.get("NIGHTOWL_ATTRIBUTE_KEYS"), DEFAULT_ATTRIBUTE_KEYS
        )
        summary_keys = _parse_key_list(
            source.get("NIGHTOWL_SUMMARY_KEYS"), DEFAULT_SUMMARY_KEYS
        )

        metrics_host = source.get("NIGHTOWL_METRICS_HOST", "0.0.0.0").strip() or "0.0.0.0"
        metrics_port_raw = source.get("NIGHTOWL_METRICS_PORT", "8010")
        try:
            metrics_port = int(metrics_port_raw)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise SettingsError("NIGHTOWL_METRICS_PORT must be an integer") from exc

        if not (1 <= metrics_port <= 65535):
            raise SettingsError("NIGHTOWL_METRICS_PORT must be between 1 and 65535")

        return Settings(
            username=username,
            password=password,
            tenant=tenant.strip() if tenant and tenant.strip() else None,
            base_url=base_url,
            poll_interval_seconds=interval,
            page_size=page_size,
            timeseries_keys=timeseries_keys,
            attribute_keys=attribute_keys,
            summary_keys=summary_keys,
            metrics_host=metrics_host,
            metrics_port=metrics_port,
        )


def _parse_key_list(raw: Optional[str], default: Sequence[str]) -> Tuple[str, ...]:
    if raw is None or not raw.strip():
        return tuple(default)

    items: Iterable[str] = (item.strip() for item in raw.split(","))
    filtered = tuple(item for item in items if item)
    if not filtered:
        raise SettingsError("Key lists cannot be empty when provided")
    return filtered
