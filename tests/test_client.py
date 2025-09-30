import pytest
import requests

from nightowl_monitor.client import ApiError, AuthError, DeviceSnapshot, NightOwlClient


class _StubResponse:
    def __init__(self, status_code=200, json_data=None, json_exc: Exception | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._json_exc = json_exc

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._json_exc:
            raise self._json_exc
        return self._json_data


class _StubSession:
    def __init__(self, response: _StubResponse | Exception):
        self._response = response
        self.post_called_with = None

    def post(self, url, json=None, timeout=None, headers=None):
        self.post_called_with = {
            "url": url,
            "json": json,
            "timeout": timeout,
            "headers": headers,
        }
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_authenticate_success():
    response = _StubResponse(json_data={"token": "abc", "refreshToken": "def"})
    session = _StubSession(response)
    client = NightOwlClient(base_url="https://example.com", session=session)

    tokens = client.authenticate("user", "pass", tenant="tenant")

    assert tokens.token == "abc"
    assert tokens.refresh_token == "def"
    assert session.post_called_with == {
        "url": "https://example.com/api/auth/login",
        "json": {"username": "user", "password": "pass", "tenant": "tenant"},
        "timeout": 30,
        "headers": None,
    }


def test_authenticate_missing_fields():
    response = _StubResponse(json_data={"token": "abc"})
    session = _StubSession(response)
    client = NightOwlClient(session=session)

    with pytest.raises(AuthError):
        client.authenticate("user", "pass")


def test_authenticate_request_error():
    session = _StubSession(requests.ConnectionError("boom"))
    client = NightOwlClient(session=session)

    with pytest.raises(AuthError):
        client.authenticate("user", "pass")


def test_authenticate_bad_json():
    response = _StubResponse(json_exc=ValueError("bad json"))
    session = _StubSession(response)
    client = NightOwlClient(session=session)

    with pytest.raises(AuthError):
        client.authenticate("user", "pass")


def test_fetch_latest_device_data_success():
    response = _StubResponse(
        json_data={
            "data": [
                {
                    "entityId": {"id": "device-1"},
                    "latest": {
                        "ATTRIBUTE": {
                            "CustomerInfo": {"value": "Customer"},
                        },
                        "TIME_SERIES": {
                            "Well_Level-1": {"value": "10"},
                        },
                        "ENTITY_FIELD": {
                            "name": {"value": "Device One"}
                        },
                    },
                }
            ]
        }
    )
    session = _StubSession(response)
    client = NightOwlClient(session=session, base_url="https://example.com")

    snapshots = client.fetch_latest_device_data(
        "token",
        timeseries_keys=["Well_Level-1"],
        attribute_keys=["CustomerInfo"],
        page_size=2,
    )

    assert session.post_called_with["headers"] == {"X-Authorization": "Bearer token"}
    assert session.post_called_with["json"]["pageLink"]["pageSize"] == 2
    assert isinstance(snapshots[0], DeviceSnapshot)
    assert snapshots[0].timeseries["Well_Level-1"] == "10"
    assert snapshots[0].attributes["CustomerInfo"] == "Customer"
    assert snapshots[0].label == "Device One"


def test_fetch_latest_device_data_request_error():
    session = _StubSession(requests.ConnectionError("boom"))
    client = NightOwlClient(session=session)

    with pytest.raises(ApiError):
        client.fetch_latest_device_data(
            "token",
            timeseries_keys=["A"],
            attribute_keys=["B"],
        )


def test_fetch_latest_device_data_http_error():
    response = _StubResponse(status_code=500)
    session = _StubSession(response)
    client = NightOwlClient(session=session)

    with pytest.raises(ApiError):
        client.fetch_latest_device_data(
            "token",
            timeseries_keys=["A"],
            attribute_keys=["B"],
        )


def test_fetch_latest_device_data_bad_json():
    response = _StubResponse(json_exc=ValueError("bad json"))
    session = _StubSession(response)
    client = NightOwlClient(session=session)

    with pytest.raises(ApiError):
        client.fetch_latest_device_data(
            "token",
            timeseries_keys=["A"],
            attribute_keys=["B"],
        )


def test_fetch_latest_device_data_missing_token():
    client = NightOwlClient()

    with pytest.raises(ApiError):
        client.fetch_latest_device_data(
            "",
            timeseries_keys=["A"],
            attribute_keys=["B"],
        )
