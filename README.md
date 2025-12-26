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
| `NIGHTOWL_BASE_URL` | ⛔️ | Override the API base URL (defaults to `https://portal.watersystem.live`). |
| `NIGHTOWL_POLL_INTERVAL_SECONDS` | ⛔️ | Poll frequency in seconds (minimum 10, defaults to 60). |
| `NIGHTOWL_PAGE_SIZE` | ⛔️ | Number of devices to request per poll (minimum 1, defaults to 5). |
| `NIGHTOWL_TIMESERIES_KEYS` | ⛔️ | Comma-separated list of telemetry keys to request (defaults to the NightOwl Flex list). |
| `NIGHTOWL_ATTRIBUTE_KEYS` | ⛔️ | Comma-separated list of device attribute keys to request. |
| `NIGHTOWL_SUMMARY_KEYS` | ⛔️ | Comma-separated keys to display in the log summary (defaults to well level, level %, total gallons). |
| `NIGHTOWL_METRICS_HOST` | ⛔️ | Bind address for the Prometheus metrics server (defaults to `0.0.0.0`). |
| `NIGHTOWL_METRICS_PORT` | ⛔️ | TCP port for the Prometheus metrics server (defaults to `8010`). Set to `8000` in production to match Kubernetes service. |

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

NightOwl Monitor exposes Prometheus metrics at `http://<host>:<port>/` (by default `http://0.0.0.0:8010/`).

> **Production Note:** In Kubernetes deployments, set `NIGHTOWL_METRICS_PORT=8000` to match the service port configuration.

### Multi-User Support

All metrics include a `user_label` dimension to isolate data between different NightOwl accounts. The label is automatically derived from the `CustomerInfo` attribute (e.g., "Dan Gorman" → "dgorman").

**Example**: Multiple deployments can run concurrently for different users, with metrics cleanly separated:
```promql
nightowl_telemetry_value{user_label="dgorman", device_name="3701155"}
nightowl_telemetry_value{user_label="demos", device_name="demo-device-1"}
```

This enables:
- **Shared Prometheus instance**: One scrape target per user deployment
- **User-specific dashboards**: Grafana variables filter by `user_label`
- **Per-user ML models**: Separate leak detection models trained on each user's data

Key metric families:

- `nightowl_telemetry_value{user_label,device_id,device_name,key}` — numeric telemetry readings (volts, amps, levels, etc.).
- `nightowl_attribute_value{user_label,...}` — numeric device attributes (boolean strings are mapped to 1/0).
- `nightowl_device_info{user_label,device_id,device_name}` — presence indicator for each discovered device.
- `nightowl_last_poll_success{user_label}` / `nightowl_last_poll_timestamp{user_label}` — health markers for the polling loop.
- `nightowl_poll_attempts_total{user_label,status}` — counter of successful vs failed polls.
- `nightowl_ml_leak_probability{user_label,device_id,device_name}` — ML leak detection probability (0-100%).
- `nightowl_ml_anomaly_score{user_label,device_id,device_name}` — Isolation Forest decision score.
- `nightowl_ml_is_anomaly{user_label,device_id,device_name}` — Binary anomaly flag.

### Scrape configuration example

> Shared Prometheus configs now live in the `monitoring/` submodule that tracks [`dgorman/infrastructure`](https://github.com/dgorman/infrastructure). After cloning this repo run `git submodule update --init --recursive` so the files are available locally.

Add the following job to your Prometheus configuration (replace the target host if needed) to integrate the metrics into your SolarDashboard Prometheus instance:

```yaml
scrape_configs:
  - job_name: "nightowl-monitor"
    metrics_path: "/"
    static_configs:
      - targets: ["nightowl-monitor.solardashboard.svc.cluster.local:8000"]
```

> **Note:** The example above shows the production Kubernetes service target on port 8000. For local development with the default app configuration, use port 8010.

If you're running the container locally, you can replace `nightowl-monitor.local` with `localhost`. When deploying alongside Prometheus (e.g., via Docker Compose), use the service name from the shared network.

### Grafana dashboard

Import `grafana/nightowl-ml-dashboard.json` into Grafana to get a comprehensive view of water system health including ML leak detection.

The dashboard supports **multi-user filtering** with cascading dropdowns:
1. **User** selector: Choose from available `user_label` values
2. **Device** selector: Automatically filtered to show only devices for the selected user

Dashboard panels include:

- **Water Level %** stat card sourced from `Level1_precent` telemetry
- **ML Leak Detection Status**: Real-time anomaly probability and confidence
- **Pump Voltages/Currents** time-series panels (keys `P1V1-3`, `P1C1-3`)
- **Pressure Trends**: Historical pressure data with anomaly overlays
- **Device Attributes** table for metadata such as `CustomerInfo` and modem status
- **Poll Status** indicator driven by `nightowl_last_poll_success`
- **Feature Contributions**: ML explainability showing which sensors triggered anomalies

All queries automatically filter by `user_label=~"$user"` to isolate data for the selected user.

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

## Machine Learning Leak Detection

NightOwl Monitor includes an intelligent leak detection system that learns what "normal" looks like for your water system and alerts you when something seems off.

### Multi-User Model Architecture

The system supports **per-user ML models** to handle different water system configurations and usage patterns:

```
models/
├── global/                         # Fallback model (generic training)
│   └── leak_detector.joblib
└── users/                          # Per-user models
    ├── dgorman/                    # Trained on dgorman's 90-day history
    │   └── leak_detector.joblib
    └── demos/                      # Demo account (placeholder)
        └── leak_detector.joblib
```

**Why per-user models?**
- Different systems have different "normal" patterns (residential vs commercial, well vs municipal)
- Usage patterns vary (family of 4 vs vacation rental)
- Sensor configurations differ (pressure-only vs pressure+flow+level)
- Reduces false positives by learning your specific system's behavior

The monitor automatically loads the model for the authenticated user based on the `user_label` derived from `CustomerInfo`.

### How It Works (Plain English)

Imagine you've been watching your water system every day for a month. After a while, you'd start to notice patterns:
- The pressure usually stays between 40-55 PSI
- The pump kicks on about 6 times during the day
- Pressure drops a little at night when no one's using water
- On weekends, usage patterns are different

**That's exactly what our ML model does—but it watches thousands of data points instead of just a few.**

Here's the process in simple terms:

1. **Learning Phase**: We show the model a month's worth of normal operation data. It studies the patterns: pressure readings, pump activity, voltage levels, time of day, day of week. It learns what "normal" looks like for YOUR specific system.

2. **Watching for Oddities**: Once trained, the model continuously compares current readings against what it learned. It asks: "Does this look like something I've seen before, or is this unusual?"

3. **Scoring Weirdness**: The model gives each reading an "anomaly score." Think of it like a weirdness meter:
   - **0-20%**: "This looks totally normal"
   - **20-50%**: "Slightly unusual, but probably fine"
   - **50-80%**: "This is odd—worth watching"
   - **80-100%**: "This doesn't match anything I've seen—possible leak!"

4. **Smart Context**: The model doesn't just look at one number. It considers:
   - Is the pressure dropping while the pump is off? (leak indicator)
   - Is this happening at 3 AM when no one should be using water?
   - How does the current reading compare to the last hour? The last 6 hours? The last day?

### Why ML Instead of Simple Thresholds?

You might ask: "Why not just alert when pressure drops below 40 PSI?"

Simple thresholds miss context. For example:
- Pressure at 38 PSI at 3 AM with the pump off = **probably a leak**
- Pressure at 38 PSI at 7 AM when everyone's showering = **completely normal**

The ML model understands this difference because it learned from real patterns, not arbitrary numbers.

### What You'll See

When the system detects something unusual, you'll see warnings like:

```
ML ANOMALY DETECTED for 3701155: probability=82.7%, score=-0.044
```

This means:
- The current readings look 82.7% "unusual" compared to normal patterns
- The negative score confirms it's flagged as an anomaly
- You should check the system—could be a leak, or could be unusual but legitimate usage

### Understanding the Probability

| Probability | What It Means | Action |
|------------|---------------|--------|
| 0-30% | Normal operation | None needed |
| 30-50% | Slightly unusual | Keep an eye on it |
| 50-70% | Moderately unusual | Check recent usage patterns |
| 70-90% | Significantly unusual | Inspect the system |
| 90-100% | Highly abnormal | Investigate immediately |

**Important**: A high probability doesn't guarantee a leak—it means the readings don't match typical patterns. It could be:
- An actual leak
- Unusual but legitimate water usage (filling a pool, guests visiting)
- A sensor issue
- Seasonal changes the model hasn't seen yet

### Model Confidence

The system also reports confidence based on how much training data it has:

- **Low confidence**: Less than 5,000 training samples—model is still learning
- **Medium confidence**: 5,000-20,000 samples—model is reasonably trained
- **High confidence**: 20,000+ samples—model has seen many patterns

---

### Technical Details (For Developers)

<details>
<summary>Click to expand technical documentation</summary>

#### How It Works (Technical)

1. **Training Phase**: The model trains on historical telemetry data from Prometheus (recommended: 30+ days)
2. **Feature Engineering**: Raw telemetry is transformed into 52 ML features:
   - Rolling window statistics (mean, std, range) at 5m, 15m, 1h, 6h, 24h windows
   - Rate-of-change features (1m, 5m, 60m deltas)
   - Cyclical time features (hour_sin, hour_cos, dow_sin, dow_cos)
   - Binary time features (is_night, is_weekend)
   - Pump-specific features (pump_running, pump_cycle_count_1h)
3. **Anomaly Detection**: Isolation Forest identifies patterns that don't match "normal" behavior
4. **Metrics Export**: Results are exposed as Prometheus metrics for dashboards and alerting

#### Algorithm: Isolation Forest

We use [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), an unsupervised anomaly detection algorithm. It works by:

1. Randomly selecting a feature and a split value
2. Recursively partitioning data until points are "isolated"
3. Anomalies are isolated faster (fewer splits needed) because they're different from the majority

Key parameters:
- `contamination=0.05` — Assumes ~5% of training data contains anomalies
- `n_estimators=100` — Uses 100 isolation trees
- `random_state=42` — Reproducible results

#### ML Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `NIGHTOWL_ML_ENABLED` | `false` | Enable ML leak detection |
| `NIGHTOWL_ML_MODEL_PATH` | `/app/models/leak_detector.joblib` | Path to trained model file |
| `NIGHTOWL_ML_PROMETHEUS_URL` | `http://prometheus:9090` | Prometheus URL for fetching inference data |
| `NIGHTOWL_ML_INFERENCE_INTERVAL` | `5` | Run ML inference every N poll cycles |

#### Training the Model

**Single-User Model:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train from Grafana Cloud (production data)
python scripts/train_ml_model.py \
  --prometheus-url https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom \
  --username 1953220 \
  --password "$GRAFANA_CLOUD_READ_TOKEN" \
  --days 90 \
  --user-label dgorman \
  --model-path models/users/dgorman/leak_detector.joblib

# Or train from local Prometheus
python scripts/train_ml_model.py \
  --prometheus-url http://localhost:9090 \
  --days 90 \
  --user-label dgorman \
  --model-path models/users/dgorman/leak_detector.joblib
```

**Adding a New User:**

1. **Create model directory:**
   ```bash
   mkdir -p models/users/newuser
   ```

2. **Fetch historical data** (if not already in Prometheus):
   ```bash
   python scripts/fetch_nightowl_data.py \
     --username "newuser@domain.com" \
     --password "password" \
     --days 90 \
     --output data/newuser_export
   ```

3. **Train user-specific model:**
   ```bash
   python scripts/train_ml_model.py \
     --prometheus-url http://localhost:9090 \
     --days 90 \
     --user-label newuser \
     --model-path models/users/newuser/leak_detector.joblib
   ```

4. **Add Kubernetes secrets** (for production deployment):
   ```yaml
   # k8s/overlays/dev/kustomization.yaml or prod equivalent
   - name: nightowl-secret-newuser
     literals:
       - NIGHTOWL_USERNAME=newuser@domain.com
       - NIGHTOWL_PASSWORD=password
       - NIGHTOWL_ML_ENABLED=true
       - NIGHTOWL_ML_MODEL_PATH=/app/models/users/newuser/leak_detector.joblib
   ```

5. **Deploy separate instance:**
   ```bash
   kubectl apply -k k8s/overlays/dev  # or prod
   ```

**Model Loading Behavior:**
- Looks for `models/users/{user_label}/leak_detector.joblib` first
- Falls back to `models/global/leak_detector.joblib` if user-specific model not found
- Logs warning if no model available (ML disabled for that user)

#### ML Prometheus Metrics

All ML metrics include `user_label` for multi-user isolation:

| Metric | Type | Description |
| --- | --- | --- |
| `nightowl_ml_leak_probability{user_label,device_id,device_name}` | Gauge | ML leak probability (0-100%) |
| `nightowl_ml_anomaly_score{user_label,device_id,device_name}` | Gauge | Isolation Forest decision score (negative = anomaly) |
| `nightowl_ml_is_anomaly{user_label,device_id,device_name}` | Gauge | Binary flag (1 = anomaly detected) |
| `nightowl_ml_model_confidence{user_label,device_id,device_name}` | Gauge | Model confidence (0=low, 1=medium, 2=high) |
| `nightowl_ml_feature_contribution{user_label,device_id,device_name,feature}` | Gauge | Per-feature contribution to anomaly score |
| `nightowl_ml_feature_value{user_label,device_id,device_name,feature}` | Gauge | Current feature values used in prediction |
| `nightowl_ml_inference_timestamp{user_label,device_id,device_name}` | Gauge | Unix timestamp of last inference |
| `nightowl_ml_inferences_total{user_label,device_id,device_name,status}` | Counter | Total inference runs by status |

#### ML vs Statistical Detection

| Aspect | Statistical (Z-Score) | ML (Isolation Forest) |
| --- | --- | --- |
| Training Required | No | Yes (30+ days recommended) |
| Labeled Data Needed | No | No |
| Multi-variate | Limited | Yes (52 features) |
| Time Awareness | Basic | Advanced (rolling windows) |
| Interpretability | High | Medium (feature contributions) |
| False Positives | Moderate | Lower with sufficient training |

</details>

## Kubernetes Deployment

### Development (Docker Desktop)

The dev overlay supports **multiple user deployments** with separate secrets for each user.

**Current Users:**
- `dgorman` (dgorman@nightowl.com) - primary user with 90-day trained model
- `demos` (demos@nightowlmonitoring.com) - demo account with placeholder model

```bash
# Build local image
docker build -t nightowl-monitor:dev .

# Deploy to Docker Desktop K8s
kubectl apply -k k8s/overlays/dev

# Check status
kubectl get pods -n nightowl
kubectl logs -n nightowl deployment/nightowl-monitor
```

**Note**: Dev deployment uses a **separate Prometheus ConfigMap** (`prometheus-configmap-dev.yaml`) that does NOT include `remote_write` to Grafana Cloud. This prevents dev test data from polluting production metrics.

### Production (MicroK8s)

Production uses a **separate Prometheus ConfigMap** (`prometheus-configmap-prod.yaml`) that includes `remote_write` to push metrics to Grafana Cloud.

```bash
# SSH to prod and pull latest changes
ssh dgorman@node01.olympusdrive.com
cd /home/dgorman/Apps/nightowl
git pull

# Apply Grafana Cloud secrets (one-time setup)
kubectl apply -f monitoring/k8s/grafana-cloud-secrets.yaml
kubectl apply -f monitoring/k8s/grafana-cloud-solardashboard-secret.yaml

# Update Prometheus deployment to mount secrets
kubectl patch deployment prometheus -n solardashboard \
  --patch-file monitoring/k8s/prometheus-secrets-patch.yaml

# Apply PRODUCTION Prometheus config (includes remote_write)
kubectl apply -f monitoring/k8s/prometheus-configmap-prod.yaml

# Deploy NightOwl
kubectl apply -k k8s/overlays/prod
kubectl rollout restart deployment/nightowl-monitor -n nightowl
```

**Dev vs Prod Prometheus Configs:**
| Config | File | remote_write | pushgateway | Use Case |
| --- | --- | --- | --- | --- |
| DEV | `prometheus-configmap-dev.yaml` | ❌ No | ✅ Yes | Local testing, historical data import |
| PROD | `prometheus-configmap-prod.yaml` | ✅ Yes | ❌ No | Live monitoring, Grafana Cloud integration |

## Grafana Cloud Integration

NightOwl metrics are pushed to Grafana Cloud via Prometheus remote_write.

### Tokens

Tokens are stored as Kubernetes secrets in the `solardashboard` namespace:

| Secret | Key | Purpose |
| --- | --- | --- |
| `grafana-cloud-nightowl` | `write-token` | Prometheus remote_write for nightowl_* metrics |
| `grafana-cloud-nightowl` | `read-token` | ML training script data queries |
| `grafana-cloud-solardashboard` | `write-token` | Prometheus remote_write for solar metrics |

### Training with Grafana Cloud

Query historical data from Grafana Cloud to train user-specific models:

```bash
# Get read token from secret
READ_TOKEN=$(kubectl get secret grafana-cloud-nightowl -n solardashboard \
  -o jsonpath='{.data.read-token}' | base64 -d)

# Train ML model for specific user using Grafana Cloud historical data
python scripts/train_ml_model.py \
  --prometheus-url "https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom" \
  --username "1953220" \
  --password "$READ_TOKEN" \
  --days 90 \
  --user-label "dgorman" \
  --model-path models/users/dgorman/leak_detector.joblib
```

**Training Recommendations:**
- **Minimum**: 30 days of data for basic model
- **Recommended**: 90 days to capture seasonal patterns and usage variations
- **Optimal**: 6+ months for production-grade models with high confidence
- Retrain models quarterly or after significant system changes (new pump, sensor upgrades)

### Available Telemetry in Grafana Cloud

| Key | Description | Data Availability |
| --- | --- | --- |
| `P1C1`, `P1C2`, `P1C3` | Pump 1 phase currents (Amps) | ✅ |
| `P1V1`, `P1V2`, `P1V3` | Pump 1 phase voltages (Volts) | ✅ |
| `P1Hz` | Pump 1 frequency (Hz) | ✅ |
| `S1` | Pressure sensor 1 (PSI) | ✅ |
| `Level1_precent` | Water level percentage | ❌ (not available on all systems) |
| `Pulse_TotalGallons` | Flow meter total | ❌ (not available on all systems) |

## Monitoring Stack Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   NightOwl API      │────▶│  NightOwl Monitor    │
│ (portal.watersystem │     │   (Python poller)    │
│       .live)        │     │  - Metrics export    │
└─────────────────────┘     │  - ML inference      │
                            └──────────┬───────────┘
                                       │ :8010
                                       ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│     Prometheus      │────▶│   Grafana Cloud      │────▶│    Grafana      │
│  (solardashboard    │     │   (remote_write)     │     │   Dashboards    │
│    namespace)       │     └──────────────────────┘     └─────────────────┘
└─────────────────────┘
        │
        ▼ query
┌─────────────────────┐
│  ML Training Script │
│  (fetch historical  │
│   data for model)   │
└─────────────────────┘
```

## Files Reference

| Path | Description |
| --- | --- |
| `src/nightowl_monitor/` | Python source code |
| `src/nightowl_monitor/ml_leak_detector.py` | ML leak detection module |
| `src/nightowl_monitor/metrics.py` | Prometheus metrics with user_label support |
| `scripts/train_ml_model.py` | ML model training script |
| `scripts/fetch_nightowl_data.py` | Historical data fetcher from NightOwl portal |
| `models/global/` | Global fallback ML models |
| `models/users/{username}/` | Per-user ML models (gitignored) |
| `k8s/base/` | Base Kubernetes manifests |
| `k8s/overlays/dev/` | Dev-specific patches (Docker Desktop, no remote_write) |
| `k8s/overlays/prod/` | Prod-specific patches (MicroK8s, with remote_write) |
| `monitoring/k8s/prometheus-configmap-dev.yaml` | DEV Prometheus config (includes pushgateway, no remote_write) |
| `monitoring/k8s/prometheus-configmap-prod.yaml` | PROD Prometheus config (includes remote_write to Grafana Cloud) |
| `grafana/nightowl-ml-dashboard.json` | Multi-user Grafana dashboard with ML panels |

## Known Issues & Notes

- **Demos User**: The `demos@nightowlmonitoring.com` credentials work on `portal.watersystem.live`. Historical data has been fetched (920,295 data points across 2 devices over 90 days). The model is currently a placeholder (copied from dgorman). To train a proper demos model, deploy the demos instance to production where live telemetry will flow, then retrain using production Prometheus data.
- **Credentials Verified** (as of 2025-12-26):
  - `dgorman@nightowl.com` / `<password from secrets>` → ✅ Works
  - `demos@nightowlmonitoring.com` / `<password from secrets>` → ✅ Works  
  - **SECURITY**: Passwords stored in Kubernetes secrets, never commit to git
- **Portal URL**: `portal.watersystem.live` (working)
- **Pushgateway Usage**: Only used in DEV for historical data import. Pushgateway retains only the latest value per label set, so it's not suitable for ML training (which needs time-series history). Use live telemetry or Grafana Cloud queries for training.
