"""Client for interacting with the NightOwl Monitoring API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests


class AuthError(RuntimeError):
    """Raised when authentication with NightOwl fails."""


class ApiError(RuntimeError):
    """Raised when querying the NightOwl API fails."""


@dataclass(frozen=True)
class TokenBundle:
    """Authentication tokens returned by the NightOwl login endpoint."""

    token: str
    refresh_token: str


@dataclass(frozen=True)
class DeviceSnapshot:
    """Latest data snapshot for a NightOwl device."""

    device_id: Optional[str]
    name: Optional[str]
    attributes: Dict[str, Any]
    timeseries: Dict[str, Any]

    @property
    def label(self) -> str:
        return self.name or self.device_id or "<unknown device>"


class NightOwlClient:
    """Thin wrapper around the NightOwl HTTP API."""

    def __init__(
        self,
        base_url: str = "https://portal.watersystem.live",
        *,
        session: Optional[requests.Session] = None,
        timeout_seconds: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    def authenticate(self, username: str, password: str, tenant: Optional[str] = None) -> TokenBundle:
        """Authenticate and return an access token bundle."""

        payload: Dict[str, Any] = {
            "username": username,
            "password": password,
        }
        if tenant:
            payload["tenant"] = tenant

        url = f"{self._base_url}/api/auth/login"
        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - delegated to caller
            raise AuthError("Unable to reach NightOwl authentication endpoint") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("NightOwl authentication endpoint returned invalid JSON") from exc

        token = data.get("token")
        refresh_token = data.get("refreshToken")
        if not token or not refresh_token:
            raise AuthError("NightOwl authentication response missing token fields")

        return TokenBundle(token=token, refresh_token=refresh_token)

    def fetch_latest_device_data(
        self,
        token: str,
        *,
        timeseries_keys: Sequence[str],
        attribute_keys: Sequence[str],
        page_size: int = 5,
    ) -> List[DeviceSnapshot]:
        """Retrieve the latest telemetry and attribute values for accessible devices."""

        if not token:
            raise ApiError("Access token is required to fetch device data")

        latest_values = [
            {"key": key, "type": "TIME_SERIES"} for key in timeseries_keys
        ] + [
            {"key": key, "type": "ATTRIBUTE"} for key in attribute_keys
        ]

        body: Dict[str, Any] = {
            "entityFilter": {
                "type": "entityType",
                "resolveMultiple": True,
                "entityType": "DEVICE",
            },
            "pageLink": {
                "page": 0,
                "pageSize": page_size,
                "textSearch": None,
                "dynamic": False,
                "sortOrder": {
                    "key": {"key": "name", "type": "ENTITY_FIELD"},
                    "direction": "ASC",
                },
            },
            "entityFields": [
                {"key": "type", "type": "ENTITY_FIELD"},
                {"key": "name", "type": "ENTITY_FIELD"},
            ],
            "latestValues": latest_values,
        }

        headers = {"X-Authorization": f"Bearer {token}"}
        url = f"{self._base_url}/api/entitiesQuery/find"

        try:
            response = self._session.post(
                url, json=body, headers=headers, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - delegated to caller
            raise ApiError("Unable to retrieve NightOwl device data") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ApiError("NightOwl device data response contained invalid JSON") from exc

        snapshots: List[DeviceSnapshot] = []
        for entry in data.get("data", []):
            entity_id = entry.get("entityId", {})
            latest = entry.get("latest", {})
            attributes = _extract_values(latest.get("ATTRIBUTE", {}))
            timeseries = _extract_values(latest.get("TIME_SERIES", {}))
            entity_fields = latest.get("ENTITY_FIELD", {})
            name_value = entity_fields.get("name", {}) if isinstance(entity_fields, dict) else {}

            snapshots.append(
                DeviceSnapshot(
                    device_id=entity_id.get("id"),
                    name=name_value.get("value"),
                    attributes=attributes,
                    timeseries=timeseries,
                )
            )

        return snapshots


def _extract_values(values: Dict[str, Any]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key, payload in values.items():
        if isinstance(payload, dict) and "value" in payload:
            extracted[key] = payload["value"]
    return extracted
