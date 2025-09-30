import pytest

from nightowl_monitor.config import (
    DEFAULT_ATTRIBUTE_KEYS,
    DEFAULT_SUMMARY_KEYS,
    DEFAULT_TIMESERIES_KEYS,
    Settings,
    SettingsError,
)


def test_settings_from_env_defaults():
    env = {
        "NIGHTOWL_USERNAME": "user",
        "NIGHTOWL_PASSWORD": "pass",
    }

    settings = Settings.from_env(env)

    assert settings.username == "user"
    assert settings.password == "pass"
    assert settings.tenant is None
    assert settings.base_url == "https://portal.nightowlmonitoring.com"
    assert settings.poll_interval_seconds == 60
    assert settings.page_size == 5
    assert settings.timeseries_keys == DEFAULT_TIMESERIES_KEYS
    assert settings.attribute_keys == DEFAULT_ATTRIBUTE_KEYS
    assert settings.summary_keys == DEFAULT_SUMMARY_KEYS


def test_settings_requires_username_password():
    with pytest.raises(SettingsError):
        Settings.from_env({"NIGHTOWL_PASSWORD": "pass"})

    with pytest.raises(SettingsError):
        Settings.from_env({"NIGHTOWL_USERNAME": "user"})


def test_settings_custom_values():
    env = {
        "NIGHTOWL_USERNAME": "user",
        "NIGHTOWL_PASSWORD": "pass",
        "NIGHTOWL_TENANT": "tenant@example.com",
        "NIGHTOWL_BASE_URL": "https://example.com/",
        "NIGHTOWL_POLL_INTERVAL_SECONDS": "120",
        "NIGHTOWL_PAGE_SIZE": "3",
        "NIGHTOWL_TIMESERIES_KEYS": "A,B,C",
        "NIGHTOWL_ATTRIBUTE_KEYS": "X , Y",
        "NIGHTOWL_SUMMARY_KEYS": "A,X",
    }

    settings = Settings.from_env(env)

    assert settings.tenant == "tenant@example.com"
    assert settings.base_url == "https://example.com"
    assert settings.poll_interval_seconds == 120
    assert settings.page_size == 3
    assert settings.timeseries_keys == ("A", "B", "C")
    assert settings.attribute_keys == ("X", "Y")
    assert settings.summary_keys == ("A", "X")


def test_settings_invalid_interval():
    env = {
        "NIGHTOWL_USERNAME": "user",
        "NIGHTOWL_PASSWORD": "pass",
        "NIGHTOWL_POLL_INTERVAL_SECONDS": "5",
    }

    with pytest.raises(SettingsError):
        Settings.from_env(env)


def test_settings_invalid_page_size():
    env = {
        "NIGHTOWL_USERNAME": "user",
        "NIGHTOWL_PASSWORD": "pass",
        "NIGHTOWL_PAGE_SIZE": "0",
    }

    with pytest.raises(SettingsError):
        Settings.from_env(env)


def test_settings_empty_custom_keys():
    env = {
        "NIGHTOWL_USERNAME": "user",
        "NIGHTOWL_PASSWORD": "pass",
        "NIGHTOWL_TIMESERIES_KEYS": ", ",
    }

    with pytest.raises(SettingsError):
        Settings.from_env(env)
