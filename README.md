# NightOwl Monitor

Polling service that authenticates against the NightOwl Monitoring API every minute.

## Features

- Authenticates with the NightOwl `/api/auth/login` endpoint and logs success.
- Retrieves the latest device telemetry/attribute values and prints both a concise summary and neatly formatted key/value sections each cycle.
- Runs continuously with a configurable poll interval (defaults to 60 seconds).
- Dockerized runtime for easy deployment.

## Configuration

Provide credentials via environment variables (for local development you can store them in a `.env` file that you do **not** commit to version control).

| Variable | Required | Description |
| --- | --- | --- |
| `NIGHTOWL_USERNAME` | ✅ | NightOwl account username. |
| `NIGHTOWL_PASSWORD` | ✅ | Password for the NightOwl account. |
| `NIGHTOWL_TENANT` | ⛔️ | Optional tenant identifier if your NightOwl instance requires it. |
| `NIGHTOWL_BASE_URL` | ⛔️ | Override the API base URL (defaults to `https://portal.nightowlmonitoring.com`). |
| `NIGHTOWL_POLL_INTERVAL_SECONDS` | ⛔️ | Poll frequency in seconds (minimum 10, defaults to 60). |
| `NIGHTOWL_PAGE_SIZE` | ⛔️ | Number of devices to request per poll (minimum 1, defaults to 5). |
| `NIGHTOWL_TIMESERIES_KEYS` | ⛔️ | Comma-separated list of telemetry keys to request (defaults to the NightOwl Flex list). |
| `NIGHTOWL_ATTRIBUTE_KEYS` | ⛔️ | Comma-separated list of device attribute keys to request. |
| `NIGHTOWL_SUMMARY_KEYS` | ⛔️ | Comma-separated keys to display in the log summary (defaults to well level, level %, total gallons). |
| `NIGHTOWL_METRICS_HOST` | ⛔️ | Bind address for the Prometheus metrics server (defaults to `0.0.0.0`). |
| `NIGHTOWL_METRICS_PORT` | ⛔️ | TCP port for the Prometheus metrics server (defaults to `8000`). |

## Run with Docker

```bash
docker build -t nightowl-monitor .
docker run --rm \
  -e NIGHTOWL_USERNAME="your-nightowl-username" \
  -e NIGHTOWL_PASSWORD="your-nightowl-password" \
  nightowl-monitor
```

> ℹ️ Add `-e NIGHTOWL_TENANT=...` or use an `--env-file` if your account needs the tenant field.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Prometheus Metrics

NightOwl Monitor exposes Prometheus metrics at `http://<host>:<port>/` (by default `http://0.0.0.0:8000/`).
Key metric families:

- `nightowl_telemetry_value{device_id,device_name,key}` — numeric telemetry readings (volts, amps, levels, etc.).
- `nightowl_attribute_value{...}` — numeric device attributes (boolean strings are mapped to 1/0).
- `nightowl_device_info{device_id,device_name}` — presence indicator for each discovered device.
- `nightowl_last_poll_success` / `nightowl_last_poll_timestamp` — health markers for the polling loop.
- `nightowl_poll_attempts_total{status}` — counter of successful vs failed polls.

### Scrape configuration example

Add the following job to your Prometheus configuration (replace the target host if needed) to integrate the metrics into your SolarDashboard Prometheus instance:

```yaml
scrape_configs:
  - job_name: "nightowl-monitor"
    metrics_path: "/"
    static_configs:
      - targets: ["nightowl-monitor.local:8000"]
```

If you're running the container locally, you can replace `nightowl-monitor.local` with `localhost`. When deploying alongside Prometheus (e.g., via Docker Compose), use the service name from the shared network.

### Grafana dashboard

Import `grafana/nightowl-dashboard.json` into Grafana to get a starter view over the NightOwl data. The dashboard expects a Prometheus data source; during import, choose the data source that scrapes your NightOwl monitor.

The dashboard includes:

- **Water Level %** stat card sourced from `Level1_precent` telemetry.
- **Pump Voltages/Currents** time-series panels (keys `P1V1-3`, `P1C1-3`).
- **Device Attributes** table for metadata such as `CustomerInfo` and modem status.
- **Poll Status** indicator driven by `nightowl_last_poll_success`.

Use the *Device* dropdown (populated via `nightowl_device_info`) to focus on a specific NightOwl device.

## Sample Output

```text
2025-09-30 21:56:23,243 [INFO] nightowl.monitor: Authenticated successfully; token prefix eyJhbGci...
2025-09-30 21:56:23,310 [INFO] nightowl.monitor: Device: Pump 1 (device-1)
2025-09-30 21:56:23,310 [INFO] nightowl.monitor: ---------------------------
2025-09-30 21:56:23,310 [INFO] nightowl.monitor: Summary:
2025-09-30 21:56:23,310 [INFO] nightowl.monitor:   Well_Level-1       : -
2025-09-30 21:56:23,310 [INFO] nightowl.monitor:   Level1_precent     : -
2025-09-30 21:56:23,310 [INFO] nightowl.monitor:   Pulse_TotalGallons : -
2025-09-30 21:56:23,310 [INFO] nightowl.monitor: 
2025-09-30 21:56:23,310 [INFO] nightowl.monitor: Telemetry:
2025-09-30 21:56:23,310 [INFO] nightowl.monitor:   P1C1 : 0
2025-09-30 21:56:23,310 [INFO] nightowl.monitor:   P1C2 : 0
2025-09-30 21:56:23,310 [INFO] nightowl.monitor:   P1V1 : 119.88
...
```

## Default Telemetry Keys

Time-series keys requested by default:

```text
Well_Level-1, S1, S2, Level1_precent,
P1V1, P1V2, P1V3, P1C1, P1C2, P1C3, P1Hz,
P2V1, P2V2, P2V3, P2C1, P2C2, P2C3, P2Hz,
P3V1, P3V2, P3V3, P3C1, P3C2, P3C3, P3Hz,
Pulse_TotalGallons
```

Attribute keys requested by default:

```text
CustomerInfo, Location, SiteInfo, active, edgedata_status,
S1_name, S2_name, Level1_Name
```

Summary keys shown in the logs (override with `NIGHTOWL_SUMMARY_KEYS`):

```text
Well_Level-1, Level1_precent, Pulse_TotalGallons
```
