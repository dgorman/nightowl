#!/usr/bin/env python3
"""
Training script for NightOwl ML Leak Detector.

Fetches historical data from Prometheus and trains an Isolation Forest model
for anomaly detection. The trained model is saved for use by the poller.

Usage:
    python -m scripts.train_ml_model --prometheus-url http://prometheus:9090 --days 30

Environment variables (alternative to CLI args):
    NIGHTOWL_ML_PROMETHEUS_URL - Prometheus server URL
    NIGHTOWL_ML_MODEL_PATH - Path to save the trained model
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nightowl_monitor.ml_leak_detector import (
    MLLeakDetector,
    PrometheusDataFetcher,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nightowl.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NightOwl ML Leak Detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on 30 days of data from local Prometheus
  python scripts/train_ml_model.py --prometheus-url http://localhost:9090 --days 30

  # Train on production data via SSH tunnel
  ssh -L 9090:prometheus:9090 node01.olympusdrive.com
  python scripts/train_ml_model.py --days 30

  # Train for specific device
  python scripts/train_ml_model.py --device-id "abc123" --days 14
        """,
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.environ.get("NIGHTOWL_ML_PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus server URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "NIGHTOWL_ML_MODEL_PATH", "/var/lib/nightowl/models/leak_detector.joblib"
        ),
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("PROMETHEUS_USERNAME"),
        help="Username for Prometheus basic auth (for Grafana Cloud)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("PROMETHEUS_PASSWORD"),
        help="Password/token for Prometheus basic auth (for Grafana Cloud)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to use for training (default: 30)",
    )
    parser.add_argument(
        "--device-id",
        help="Train model for specific device ID. If not specified, trains on all devices.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected fraction of anomalies in training data (default: 0.05)",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=7,
        help="Days of most recent data to use for validation (default: 7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data and show stats without training",
    )
    return parser.parse_args()


def discover_devices(fetcher: PrometheusDataFetcher, days: int = 30) -> list:
    """Discover available device IDs from Prometheus using historical data."""
    import requests
    from datetime import datetime, timedelta

    url = f"{fetcher.prometheus_url}/api/v1/query_range"
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    params = {
        "query": 'count by (device_id) (nightowl_telemetry_value)',
        "start": start_time.timestamp(),
        "end": end_time.timestamp(),
        "step": "1d",  # Daily samples are enough for discovery
    }

    try:
        response = requests.get(url, params=params, timeout=30, auth=fetcher.auth)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to discover devices: {e}")
        return []

    if data.get("status") != "success":
        logger.error(f"Query failed: {data.get('error', 'unknown')}")
        return []

    results = data.get("data", {}).get("result", [])
    devices = [r.get("metric", {}).get("device_id") for r in results]
    return [d for d in devices if d]


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NightOwl ML Leak Detector - Training Script")
    logger.info("=" * 60)
    logger.info(f"Prometheus URL: {args.prometheus_url}")
    logger.info(f"Training days: {args.days}")
    logger.info(f"Validation days: {args.validation_days}")
    logger.info(f"Contamination: {args.contamination}")
    logger.info(f"Model path: {args.model_path}")
    if args.username:
        logger.info(f"Using auth: username={args.username}")
    logger.info("=" * 60)

    # Initialize fetcher with optional authentication
    fetcher = PrometheusDataFetcher(
        prometheus_url=args.prometheus_url,
        username=args.username,
        password=args.password,
    )

    # Discover or use specified device
    if args.device_id:
        device_ids = [args.device_id]
    else:
        logger.info("Discovering devices from Prometheus...")
        device_ids = discover_devices(fetcher, days=args.days)
        if not device_ids:
            logger.error("No devices found in Prometheus. Check your Prometheus URL and data.")
            return 1
        logger.info(f"Found {len(device_ids)} devices: {device_ids}")

    # Fetch training data for each device
    all_training_data = []
    all_validation_data = []

    for device_id in device_ids:
        logger.info(f"\nFetching data for device: {device_id}")

        # Fetch full training period
        data = fetcher.fetch_training_data(
            device_id=device_id,
            days=args.days,
        )

        if data.empty:
            logger.warning(f"No data found for device {device_id}")
            continue

        logger.info(f"  Retrieved {len(data)} data points")
        logger.info(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        logger.info(f"  Columns: {list(data.columns)}")

        # Show data statistics
        for col in data.columns:
            if col != "timestamp":
                valid_count = data[col].notna().sum()
                if valid_count > 0:
                    logger.info(
                        f"    {col}: mean={data[col].mean():.2f}, "
                        f"std={data[col].std():.2f}, "
                        f"min={data[col].min():.2f}, "
                        f"max={data[col].max():.2f}"
                    )

        # Split into training and validation
        cutoff = datetime.utcnow() - timedelta(days=args.validation_days)
        training = data[data["timestamp"] < cutoff]
        validation = data[data["timestamp"] >= cutoff]

        logger.info(f"  Training samples: {len(training)}")
        logger.info(f"  Validation samples: {len(validation)}")

        if len(training) > 0:
            training["device_id"] = device_id
            all_training_data.append(training)
        if len(validation) > 0:
            validation["device_id"] = device_id
            all_validation_data.append(validation)

    if not all_training_data:
        logger.error("No training data available. Check Prometheus connection and data availability.")
        return 1

    # Combine data from all devices
    import pandas as pd

    combined_training = pd.concat(all_training_data, ignore_index=True)
    combined_validation = pd.concat(all_validation_data, ignore_index=True) if all_validation_data else pd.DataFrame()

    logger.info("\n" + "=" * 60)
    logger.info("Combined Training Data Summary")
    logger.info("=" * 60)
    logger.info(f"Total training samples: {len(combined_training)}")
    logger.info(f"Total validation samples: {len(combined_validation)}")
    logger.info(f"Devices: {combined_training['device_id'].nunique()}")

    if args.dry_run:
        logger.info("\n[DRY RUN] Skipping model training")
        return 0

    # Train the model
    logger.info("\n" + "=" * 60)
    logger.info("Training Isolation Forest Model")
    logger.info("=" * 60)

    detector = MLLeakDetector(
        contamination=args.contamination,
        model_path=Path(args.model_path),
    )

    # Drop device_id column before training (not a feature)
    training_features = combined_training.drop(columns=["device_id"], errors="ignore")
    validation_features = combined_validation.drop(columns=["device_id"], errors="ignore") if not combined_validation.empty else None

    stats = detector.train(
        training_data=training_features,
        validation_data=validation_features,
    )

    # Print training results
    logger.info("\nTraining Results:")
    logger.info(f"  Feature count: {stats['feature_count']}")
    logger.info(f"  Anomaly count (training): {stats['anomaly_count']}")
    logger.info(f"  Anomaly rate (training): {stats['anomaly_rate']:.2%}")
    logger.info(f"  Score mean: {stats['score_mean']:.4f}")
    logger.info(f"  Score std: {stats['score_std']:.4f}")
    if "validation_anomaly_rate" in stats:
        logger.info(f"  Anomaly rate (validation): {stats['validation_anomaly_rate']:.2%}")

    logger.info("\nTop features used:")
    for i, feat in enumerate(stats["feature_names"][:10], 1):
        logger.info(f"  {i}. {feat}")

    # Save the model
    logger.info(f"\nSaving model to: {args.model_path}")
    detector.save()

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info("\nTo use the trained model, set these environment variables:")
    logger.info(f"  NIGHTOWL_ML_ENABLED=true")
    logger.info(f"  NIGHTOWL_ML_MODEL_PATH={args.model_path}")
    logger.info(f"  NIGHTOWL_ML_PROMETHEUS_URL={args.prometheus_url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
