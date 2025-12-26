#!/usr/bin/env python3
"""
Push historical NightOwl CSV data to Prometheus via Remote Write.

This script reads CSV files exported by fetch_nightowl_data.py and pushes
them to a Prometheus-compatible remote write endpoint (e.g., Grafana Cloud).

The data is pushed with proper labels so it can be queried alongside
live data from the NightOwl monitor.

Usage:
    # Push to Grafana Cloud
    python scripts/push_csv_to_prometheus.py \
        --csv data/customers/customer_xyz_Device_20250101_20250331.csv \
        --url https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom/push \
        --username 1953220 \
        --password glc_xxx

    # Push multiple CSVs
    python scripts/push_csv_to_prometheus.py \
        --csv data/customers/*.csv \
        --url https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom/push \
        --username 1953220 \
        --password glc_xxx

    # Dry run (no actual push)
    python scripts/push_csv_to_prometheus.py \
        --csv data/test.csv \
        --dry-run
"""

import argparse
import csv
import gzip
import glob
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

# Prometheus remote write uses snappy compression and protobuf
# We'll use the simpler influx line protocol style for Grafana Cloud
# or the OpenMetrics text format

# Metric name mapping from CSV columns to Prometheus metric names
METRIC_MAPPING = {
    "P1C1": "nightowl_telemetry_value",
    "P1C2": "nightowl_telemetry_value",
    "P1C3": "nightowl_telemetry_value",
    "P1Hz": "nightowl_telemetry_value",
    "P1V1": "nightowl_telemetry_value",
    "P1V2": "nightowl_telemetry_value",
    "P1V3": "nightowl_telemetry_value",
    "S1": "nightowl_telemetry_value",
    "S2": "nightowl_telemetry_value",
    "Level1_precent": "nightowl_telemetry_value",
    "Pulse_TotalGallons": "nightowl_telemetry_value",
    "Well_Level-1": "nightowl_telemetry_value",
}

# Telemetry keys that should be pushed
TELEMETRY_KEYS = [
    "P1C1", "P1C2", "P1C3", "P1Hz", 
    "P1V1", "P1V2", "P1V3",
    "S1", "S2", 
    "Level1_precent", "Pulse_TotalGallons", "Well_Level-1",
]


def read_csv_rows(csv_path: str) -> Iterator[Dict[str, Any]]:
    """Read CSV file and yield rows as dicts."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def build_remote_write_payload(
    rows: List[Dict[str, Any]],
    metric_prefix: str = "nightowl_historical",
) -> List[Dict[str, Any]]:
    """
    Build a list of timeseries for remote write.
    
    Returns a structure compatible with Prometheus remote write API.
    Each timeseries has labels and samples.
    """
    # Group samples by metric + labels
    timeseries_map: Dict[str, Dict[str, Any]] = {}
    
    for row in rows:
        timestamp_ms = int(row.get("timestamp", 0))
        if not timestamp_ms:
            continue
            
        # Extract label values
        user_label = row.get("user_label", "unknown")
        device_id = row.get("device_id", "")
        device_name = row.get("device_name", "unknown")
        
        # Process each telemetry key
        for key in TELEMETRY_KEYS:
            value_str = row.get(key, "")
            if value_str == "" or value_str is None:
                continue
                
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            
            # Build unique key for this timeseries
            labels = {
                "user_label": user_label,
                "device_id": device_id,
                "device_name": device_name,
                "key": key,
                "source": "historical_import",
            }
            labels_key = tuple(sorted(labels.items()))
            metric_name = f"{metric_prefix}_telemetry_value"
            ts_key = (metric_name, labels_key)
            
            if ts_key not in timeseries_map:
                timeseries_map[ts_key] = {
                    "__name__": metric_name,
                    "labels": labels,
                    "samples": [],
                }
            
            timeseries_map[ts_key]["samples"].append({
                "timestamp": timestamp_ms,
                "value": value,
            })
    
    return list(timeseries_map.values())


def format_as_influx_line_protocol(
    timeseries: List[Dict[str, Any]],
) -> str:
    """
    Format timeseries as InfluxDB line protocol.
    
    Grafana Cloud accepts this format via their /api/v1/push/influx endpoint.
    Format: measurement,tag1=val1,tag2=val2 field=value timestamp_ns
    """
    lines = []
    
    for ts in timeseries:
        metric_name = ts["__name__"]
        labels = ts["labels"]
        
        # Build tags string
        tags = ",".join(f'{k}={v.replace(" ", "_").replace(",", "_")}' 
                        for k, v in sorted(labels.items()) if v)
        
        for sample in ts["samples"]:
            # Convert ms to ns for InfluxDB
            timestamp_ns = sample["timestamp"] * 1_000_000
            value = sample["value"]
            
            line = f"{metric_name},{tags} value={value} {timestamp_ns}"
            lines.append(line)
    
    return "\n".join(lines)


def format_as_openmetrics(
    timeseries: List[Dict[str, Any]],
) -> str:
    """
    Format timeseries as OpenMetrics text format.
    
    This can be used with promtool for backfilling or some remote write endpoints.
    """
    lines = []
    seen_metrics = set()
    
    for ts in timeseries:
        metric_name = ts["__name__"]
        labels = ts["labels"]
        
        # Add TYPE and HELP if not seen
        if metric_name not in seen_metrics:
            lines.append(f"# HELP {metric_name} Historical NightOwl telemetry data")
            lines.append(f"# TYPE {metric_name} gauge")
            seen_metrics.add(metric_name)
        
        # Build labels string
        labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()) if v)
        
        for sample in ts["samples"]:
            timestamp_ms = sample["timestamp"]
            value = sample["value"]
            
            line = f'{metric_name}{{{labels_str}}} {value} {timestamp_ms}'
            lines.append(line)
    
    lines.append("# EOF")
    return "\n".join(lines)


def push_to_grafana_cloud(
    timeseries: List[Dict[str, Any]],
    url: str,
    username: str,
    password: str,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Push timeseries to Grafana Cloud via remote write.
    
    Returns (success_count, failure_count).
    """
    success_count = 0
    failure_count = 0
    
    # Flatten all samples for batching
    all_samples = []
    for ts in timeseries:
        metric_name = ts["__name__"]
        labels = ts["labels"]
        for sample in ts["samples"]:
            all_samples.append({
                "metric": metric_name,
                "labels": labels,
                "timestamp": sample["timestamp"],
                "value": sample["value"],
            })
    
    # Sort by timestamp
    all_samples.sort(key=lambda x: x["timestamp"])
    
    print(f"  Total samples to push: {len(all_samples):,}")
    
    if dry_run:
        print("  [DRY RUN] Would push samples but skipping actual API calls")
        return len(all_samples), 0
    
    # Push in batches using Influx line protocol
    # Grafana Cloud accepts this at /api/v1/push/influx
    influx_url = url.replace("/api/prom/push", "/api/v1/push/influx")
    
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i:i + batch_size]
        
        # Format as line protocol
        lines = []
        for s in batch:
            tags = ",".join(f'{k}={v.replace(" ", "_").replace(",", "_").replace("=", "_")}' 
                           for k, v in sorted(s["labels"].items()) if v)
            timestamp_ns = s["timestamp"] * 1_000_000
            line = f'{s["metric"]},{tags} value={s["value"]} {timestamp_ns}'
            lines.append(line)
        
        payload = "\n".join(lines)
        
        try:
            response = requests.post(
                influx_url,
                data=payload,
                auth=(username, password),
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            
            if response.status_code in (200, 204):
                success_count += len(batch)
            else:
                print(f"  Warning: Batch failed with status {response.status_code}: {response.text[:200]}")
                failure_count += len(batch)
                
        except requests.RequestException as e:
            print(f"  Error pushing batch: {e}")
            failure_count += len(batch)
        
        # Progress indicator
        pct = min(100, (i + batch_size) / len(all_samples) * 100)
        print(f"  Progress: {pct:.1f}% ({success_count:,} pushed)", end="\r")
        
        # Rate limiting - be nice to the API
        time.sleep(0.1)
    
    print()  # Newline after progress
    return success_count, failure_count


def push_to_pushgateway(
    timeseries: List[Dict[str, Any]],
    url: str,
    job: str = "historical_import",
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Push timeseries to Prometheus Pushgateway.
    
    This pushes metrics that Prometheus will scrape. Note that Pushgateway
    only keeps the latest value for each unique metric/label combination,
    so historical data with different timestamps will overwrite each other.
    
    For true historical backfill, use --export and promtool backfill.
    For testing the metric format and labels, Pushgateway works well.
    
    Returns (success_count, failure_count).
    """
    # Pushgateway accepts Prometheus text exposition format
    # Group by unique label set to get latest values only
    latest_by_labels: Dict[str, Dict[str, Any]] = {}
    
    for ts in timeseries:
        metric_name = ts["__name__"]
        labels = ts["labels"]
        labels_key = tuple(sorted(labels.items()))
        
        # Get the latest sample for each unique label set
        if ts["samples"]:
            latest_sample = max(ts["samples"], key=lambda x: x["timestamp"])
            key = (metric_name, labels_key)
            latest_by_labels[key] = {
                "metric": metric_name,
                "labels": labels,
                "value": latest_sample["value"],
                "timestamp": latest_sample["timestamp"],
            }
    
    # Format as Prometheus text exposition
    lines = []
    lines.append(f"# HELP nightowl_historical_telemetry_value Historical NightOwl telemetry data")
    lines.append(f"# TYPE nightowl_historical_telemetry_value gauge")
    
    for data in latest_by_labels.values():
        labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(data["labels"].items()))
        lines.append(f'{data["metric"]}{{{labels_str}}} {data["value"]}')
    
    payload = "\n".join(lines) + "\n"
    
    print(f"  Unique metric/label combinations: {len(latest_by_labels)}")
    print(f"  Note: Pushgateway keeps only latest value per label set")
    
    if dry_run:
        print("  [DRY RUN] Would push to Pushgateway but skipping")
        print("\n  Sample of what would be pushed:")
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 10:
            print(f"    ... and {len(lines) - 10} more lines")
        return len(latest_by_labels), 0
    
    # Push to Pushgateway
    # URL format: http://pushgateway:9091/metrics/job/<job>
    push_url = f"{url.rstrip('/')}/metrics/job/{job}"
    
    try:
        response = requests.post(
            push_url,
            data=payload,
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        
        if response.status_code in (200, 202):
            print(f"  Successfully pushed to Pushgateway")
            return len(latest_by_labels), 0
        else:
            print(f"  Error: Pushgateway returned {response.status_code}: {response.text[:200]}")
            return 0, len(latest_by_labels)
            
    except requests.RequestException as e:
        print(f"  Error pushing to Pushgateway: {e}")
        return 0, len(latest_by_labels)


def export_openmetrics_file(
    timeseries: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Export timeseries to OpenMetrics format file for promtool backfill."""
    content = format_as_openmetrics(timeseries)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Push NightOwl CSV data to Prometheus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Push to Grafana Cloud
    python scripts/push_csv_to_prometheus.py \\
        --csv data/customers/customer_xyz.csv \\
        --url https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom/push \\
        --username 1953220 \\
        --password glc_xxx

    # Export to OpenMetrics for promtool backfill
    python scripts/push_csv_to_prometheus.py \\
        --csv data/customers/customer_xyz.csv \\
        --export-openmetrics data/customer_xyz.om

    # Dry run
    python scripts/push_csv_to_prometheus.py \\
        --csv data/test.csv \\
        --dry-run
        """,
    )
    
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="CSV file(s) to process (supports glob patterns)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(
            "PROMETHEUS_REMOTE_WRITE_URL",
            "https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom/push"
        ),
        help="Prometheus remote write URL",
    )
    parser.add_argument(
        "--username", "-u",
        default=os.environ.get("GRAFANA_CLOUD_USER", "1953220"),
        help="Grafana Cloud username/user ID",
    )
    parser.add_argument(
        "--password", "-p",
        default=os.environ.get("GRAFANA_CLOUD_TOKEN", ""),
        help="Grafana Cloud API token",
    )
    parser.add_argument(
        "--metric-prefix",
        default="nightowl_historical",
        help="Prefix for metric names (default: nightowl_historical)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Show N sample lines of what would be pushed (for verification)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of samples per batch (default: 1000)",
    )
    parser.add_argument(
        "--export-openmetrics",
        default=None,
        help="Export to OpenMetrics file instead of pushing (for promtool backfill)",
    )
    parser.add_argument(
        "--pushgateway",
        default=None,
        metavar="URL",
        help="Push to Pushgateway URL (e.g., http://localhost:9091) for dev testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and prepare data but don't push",
    )
    
    args = parser.parse_args()
    
    # Find CSV files
    csv_files = glob.glob(args.csv)
    if not csv_files:
        print(f"No files found matching: {args.csv}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    total_success = 0
    total_failure = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        # Read CSV
        rows = list(read_csv_rows(csv_file))
        print(f"  Read {len(rows):,} rows")
        
        if not rows:
            print("  Skipping empty file")
            continue
        
        # Build timeseries
        timeseries = build_remote_write_payload(rows, metric_prefix=args.metric_prefix)
        total_samples = sum(len(ts["samples"]) for ts in timeseries)
        print(f"  Built {len(timeseries)} timeseries with {total_samples:,} samples")
        
        # Preview mode - show sample data and exit
        if args.preview > 0:
            print(f"\n  === PREVIEW (first {args.preview} samples per timeseries) ===")
            print(f"  Metric prefix: {args.metric_prefix}")
            print(f"  This data will NOT mix with live 'nightowl_telemetry_value' metrics\n")
            
            for ts in timeseries[:5]:  # Show first 5 timeseries
                metric_name = ts["__name__"]
                labels = ts["labels"]
                labels_str = ", ".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                
                print(f"  Timeseries: {metric_name}{{{labels_str}}}")
                for sample in ts["samples"][:args.preview]:
                    dt = datetime.fromtimestamp(sample["timestamp"] / 1000).isoformat()
                    print(f"    {dt} -> {sample['value']}")
                print()
            
            if len(timeseries) > 5:
                print(f"  ... and {len(timeseries) - 5} more timeseries")
            continue
        
        # Export or push
        if args.export_openmetrics:
            output_path = args.export_openmetrics
            if len(csv_files) > 1:
                # Generate unique filename for each CSV
                base = Path(csv_file).stem
                output_path = str(Path(args.export_openmetrics).parent / f"{base}.om")
            
            export_openmetrics_file(timeseries, output_path)
            print(f"  Exported to: {output_path}")
        elif args.pushgateway:
            success, failure = push_to_pushgateway(
                timeseries,
                url=args.pushgateway,
                job="historical_import",
                dry_run=args.dry_run,
            )
            total_success += success
            total_failure += failure
        else:
            if not args.password and not args.dry_run:
                print("  Error: --password required for pushing (or set GRAFANA_CLOUD_TOKEN)")
                continue
            
            success, failure = push_to_grafana_cloud(
                timeseries,
                url=args.url,
                username=args.username,
                password=args.password,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            total_success += success
            total_failure += failure
            print(f"  Pushed: {success:,} success, {failure:,} failed")
    
    if not args.export_openmetrics:
        print(f"\nTotal: {total_success:,} samples pushed, {total_failure:,} failed")


if __name__ == "__main__":
    main()
