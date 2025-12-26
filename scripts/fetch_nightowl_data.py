#!/usr/bin/env python3
"""
Fetch historical telemetry data from NightOwl API for a given user.

This script authenticates with the NightOwl API and downloads historical
telemetry data for all devices accessible to the user. Data is saved as
CSV files for use in ML model training.

Usage:
    python scripts/fetch_nightowl_data.py --username user@example.com --password secret
    python scripts/fetch_nightowl_data.py --username user@example.com --password secret --days 30
    python scripts/fetch_nightowl_data.py --username user@example.com --password secret --output data/user1/
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class NightOwlDataFetcher:
    """Fetches historical telemetry data from NightOwl API."""

    # Standard telemetry keys to fetch
    TELEMETRY_KEYS = [
        "P1C1", "P1C2", "P1C3",  # Pump currents
        "P1Hz",                   # Pump frequency
        "P1V1", "P1V2", "P1V3",  # Pump voltages
        "S1", "S2",              # Pressure sensors
        "Level1_precent",        # Tank level
        "Pulse_TotalGallons",    # Flow meter
        "Well_Level-1",          # Well level
    ]

    ATTRIBUTE_KEYS = [
        "CustomerInfo",
        "SiteInfo", 
        "Location",
        "active",
    ]

    def __init__(
        self,
        base_url: str = "https://portal.watersystem.live",
        timeout_seconds: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.session = requests.Session()
        self.token: Optional[str] = None

    def authenticate(self, username: str, password: str, tenant: str = "") -> bool:
        """Authenticate with NightOwl and store the access token."""
        payload = {
            "username": username,
            "password": password,
        }
        if tenant:
            payload["tenant"] = tenant

        url = f"{self.base_url}/api/auth/login"
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self.token = data.get("token")
            return bool(self.token)
        except requests.RequestException as e:
            print(f"Authentication failed: {e}")
            return False

    def _headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        if not self.token:
            raise RuntimeError("Not authenticated")
        return {"X-Authorization": f"Bearer {self.token}"}

    def list_devices(self) -> List[Dict[str, Any]]:
        """List all accessible devices with their info."""
        body = {
            "entityFilter": {
                "type": "entityType",
                "resolveMultiple": True,
                "entityType": "DEVICE",
            },
            "pageLink": {
                "page": 0,
                "pageSize": 100,
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
            "latestValues": [
                {"key": key, "type": "ATTRIBUTE"} for key in self.ATTRIBUTE_KEYS
            ],
        }

        url = f"{self.base_url}/api/entitiesQuery/find"
        response = self.session.post(
            url, json=body, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        devices = []
        for entry in data.get("data", []):
            entity_id = entry.get("entityId", {})
            latest = entry.get("latest", {})
            entity_fields = latest.get("ENTITY_FIELD", {})
            attributes = latest.get("ATTRIBUTE", {})

            device = {
                "device_id": entity_id.get("id"),
                "name": entity_fields.get("name", {}).get("value", "Unknown"),
                "attributes": {
                    k: v.get("value") for k, v in attributes.items() if isinstance(v, dict)
                },
            }
            devices.append(device)

        return devices

    def fetch_timeseries(
        self,
        device_id: str,
        keys: List[str],
        start_ts: int,
        end_ts: int,
        limit: int = 50000,
        agg_type: str = "NONE",
        interval: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical timeseries data for a device.
        
        Args:
            device_id: The device UUID
            keys: List of telemetry keys to fetch
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            limit: Maximum number of points per key
            agg_type: Aggregation type (NONE, AVG, MIN, MAX, SUM, COUNT)
            interval: Aggregation interval in milliseconds (0 for raw data)
            
        Returns:
            Dict mapping key names to lists of {ts, value} records
        """
        # The NightOwl API uses ThingsBoard's timeseries endpoint
        keys_param = ",".join(keys)
        url = (
            f"{self.base_url}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
            f"?keys={keys_param}"
            f"&startTs={start_ts}"
            f"&endTs={end_ts}"
            f"&limit={limit}"
            f"&agg={agg_type}"
            f"&interval={interval}"
            f"&orderBy=ASC"
            f"&useStrictDataTypes=true"
        )

        try:
            response = self.session.get(url, headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch timeseries for device {device_id}: {e}")
            return {}

    def fetch_device_history(
        self,
        device_id: str,
        days: int = 30,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical data for a device over a specified time range.
        
        Args:
            device_id: The device UUID
            days: Number of days of history to fetch
            keys: Telemetry keys to fetch (defaults to TELEMETRY_KEYS)
            
        Returns:
            Dict mapping key names to lists of {ts, value} records
        """
        if keys is None:
            keys = self.TELEMETRY_KEYS

        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        print(f"  Fetching {days} days of data ({datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)})")

        # Fetch in chunks to avoid API limits
        all_data: Dict[str, List[Dict[str, Any]]] = {key: [] for key in keys}
        chunk_days = 7  # Fetch one week at a time
        
        current_start = start_ts
        while current_start < end_ts:
            current_end = min(current_start + (chunk_days * 24 * 60 * 60 * 1000), end_ts)
            
            chunk_data = self.fetch_timeseries(
                device_id=device_id,
                keys=keys,
                start_ts=current_start,
                end_ts=current_end,
                limit=50000,
            )
            
            for key, values in chunk_data.items():
                if key in all_data:
                    all_data[key].extend(values)
            
            current_start = current_end
            
        # Sort by timestamp
        for key in all_data:
            all_data[key].sort(key=lambda x: x.get("ts", 0))
            
        return all_data


def save_to_csv(
    device_info: Dict[str, Any],
    timeseries_data: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    user_label: Optional[str] = None,
) -> str:
    """
    Save timeseries data to a CSV file.
    
    Creates a wide-format CSV with timestamp as the first column and
    each telemetry key as additional columns. Includes user/customer
    labels for multi-user dataset analysis.
    """
    device_id = device_info["device_id"]
    device_name = device_info["name"].replace(" ", "_").replace("/", "-")
    customer_info = device_info.get("attributes", {}).get("CustomerInfo", "Unknown")
    site_info = device_info.get("attributes", {}).get("SiteInfo", "Unknown")
    
    # Use provided user_label or derive from customer info
    if not user_label:
        user_label = customer_info.replace(" ", "_").replace("/", "-")
    
    # Collect all unique timestamps
    all_timestamps = set()
    for values in timeseries_data.values():
        for point in values:
            all_timestamps.add(point.get("ts"))
    
    if not all_timestamps:
        print(f"  No data found for device {device_name}")
        return ""
    
    # Sort timestamps
    sorted_ts = sorted(all_timestamps)
    
    # Build lookup dicts for each key
    key_lookups = {}
    for key, values in timeseries_data.items():
        key_lookups[key] = {p["ts"]: p.get("value") for p in values}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with date range and user label
    min_ts = min(sorted_ts)
    max_ts = max(sorted_ts)
    start_date = datetime.fromtimestamp(min_ts / 1000).strftime("%Y%m%d")
    end_date = datetime.fromtimestamp(max_ts / 1000).strftime("%Y%m%d")
    
    filename = f"{user_label}_{device_name}_{start_date}_{end_date}.csv"
    filepath = output_dir / filename
    
    # Write CSV with user/device label columns for Prometheus-style labeling
    keys = list(timeseries_data.keys())
    header = ["timestamp", "datetime", "user_label", "device_id", "device_name", "customer", "site"] + keys
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for ts in sorted_ts:
            dt = datetime.fromtimestamp(ts / 1000).isoformat()
            row = [ts, dt, user_label, device_id, device_name, customer_info, site_info]
            for key in keys:
                row.append(key_lookups[key].get(ts, ""))
            writer.writerow(row)
    
    return str(filepath)


def save_device_info(devices: List[Dict[str, Any]], output_dir: Path) -> str:
    """Save device metadata to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "devices.json"
    
    with open(filepath, "w") as f:
        json.dump(devices, f, indent=2)
    
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical data from NightOwl API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch 30 days of data for a user
    python scripts/fetch_nightowl_data.py \\
        --username user@example.com \\
        --password secret123

    # Fetch 90 days of data to a specific directory
    python scripts/fetch_nightowl_data.py \\
        --username user@example.com \\
        --password secret123 \\
        --days 90 \\
        --output data/customer_xyz/

    # Fetch only specific telemetry keys
    python scripts/fetch_nightowl_data.py \\
        --username user@example.com \\
        --password secret123 \\
        --keys S1,P1C1,P1Hz
        """,
    )
    
    parser.add_argument(
        "--username", "-u",
        required=True,
        help="NightOwl username (email)",
    )
    parser.add_argument(
        "--password", "-p",
        required=True,
        help="NightOwl password",
    )
    parser.add_argument(
        "--tenant", "-t",
        default="",
        help="NightOwl tenant (optional)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Number of days of history to fetch (default: 30)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/nightowl_export",
        help="Output directory for CSV files (default: data/nightowl_export)",
    )
    parser.add_argument(
        "--keys", "-k",
        default=None,
        help="Comma-separated list of telemetry keys (default: all standard keys)",
    )
    parser.add_argument(
        "--device-id",
        default=None,
        help="Fetch data for a specific device ID only",
    )
    parser.add_argument(
        "--label", "-l",
        default=None,
        help="User/customer label for the data (used in CSV columns and filename). "
             "Defaults to CustomerInfo from device attributes.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List devices and exit without fetching data",
    )
    
    args = parser.parse_args()
    
    # Parse keys if provided
    keys = None
    if args.keys:
        keys = [k.strip() for k in args.keys.split(",")]
    
    output_dir = Path(args.output)
    
    # Initialize fetcher
    fetcher = NightOwlDataFetcher()
    
    # Authenticate
    print(f"Authenticating as {args.username}...")
    if not fetcher.authenticate(args.username, args.password, args.tenant):
        print("Authentication failed!")
        sys.exit(1)
    print("Authentication successful!")
    
    # List devices
    print("\nDiscovering devices...")
    devices = fetcher.list_devices()
    
    if not devices:
        print("No devices found for this user.")
        sys.exit(0)
    
    print(f"Found {len(devices)} device(s):")
    for d in devices:
        attrs = d.get("attributes", {})
        customer = attrs.get("CustomerInfo", "N/A")
        site = attrs.get("SiteInfo", "N/A")
        print(f"  - {d['name']} (ID: {d['device_id'][:8]}...)")
        print(f"    Customer: {customer}, Site: {site}")
    
    if args.list_devices:
        # Save device info and exit
        info_file = save_device_info(devices, output_dir)
        print(f"\nDevice info saved to: {info_file}")
        sys.exit(0)
    
    # Filter to specific device if requested
    if args.device_id:
        devices = [d for d in devices if d["device_id"] == args.device_id]
        if not devices:
            print(f"Device ID {args.device_id} not found!")
            sys.exit(1)
    
    # Fetch data for each device
    print(f"\nFetching {args.days} days of telemetry data...")
    
    for device in devices:
        device_id = device["device_id"]
        device_name = device["name"]
        
        print(f"\nProcessing device: {device_name}")
        
        data = fetcher.fetch_device_history(
            device_id=device_id,
            days=args.days,
            keys=keys,
        )
        
        # Count total points
        total_points = sum(len(v) for v in data.values())
        print(f"  Retrieved {total_points:,} data points")
        
        if total_points > 0:
            filepath = save_to_csv(device, data, output_dir, user_label=args.label)
            if filepath:
                print(f"  Saved to: {filepath}")
    
    # Save device metadata
    info_file = save_device_info(devices, output_dir)
    print(f"\nDevice metadata saved to: {info_file}")
    print(f"\nExport complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
