#!/usr/bin/env python3
"""
Import NightOwl dashboard into local Grafana instance.
Usage: ./import_dashboard.py
"""

import json
import os
import requests
import sys

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
USERNAME = os.environ.get("GRAFANA_USERNAME", "admin")
PASSWORD = os.environ.get("GRAFANA_PASSWORD")
DASHBOARD_FILE = "grafana/nightowl-dashboard.json"

if not PASSWORD:
    print("Error: GRAFANA_PASSWORD environment variable is required")
    sys.exit(1)

def import_dashboard():
    """Import the NightOwl dashboard into Grafana."""
    
    # Load dashboard JSON
    try:
        with open(DASHBOARD_FILE, 'r') as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dashboard file not found: {DASHBOARD_FILE}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dashboard file: {e}")
        return False
    
    # Get data sources to map DS_PROMETHEUS
    print("Fetching Prometheus data source...")
    try:
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(USERNAME, PASSWORD),
            timeout=10
        )
        response.raise_for_status()
        datasources = response.json()
        
        # Find Prometheus data source
        prometheus_ds = None
        for ds in datasources:
            if ds.get('type') == 'prometheus':
                prometheus_ds = ds
                print(f"Found Prometheus data source: {ds['name']} (UID: {ds['uid']})")
                break
        
        if not prometheus_ds:
            print("Warning: No Prometheus data source found. Dashboard may not work correctly.")
            print("Available data sources:")
            for ds in datasources:
                print(f"  - {ds['name']} ({ds['type']})")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data sources: {e}")
        print("Continuing with dashboard import...")
    
    # Prepare dashboard for import
    # Remove properties that should not be included in import
    dashboard_json.pop('id', None)
    dashboard_json.pop('uid', None)
    dashboard_json.pop('version', None)
    
    # Prepare the import payload
    payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Imported NightOwl dashboard with leak detection",
        "folderId": 0  # General folder
    }
    
    # Import dashboard
    print(f"Importing dashboard to {GRAFANA_URL}...")
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            auth=(USERNAME, PASSWORD),
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"{GRAFANA_URL}{result.get('url', '')}"
            print(f"✅ Dashboard imported successfully!")
            print(f"   URL: {dashboard_url}")
            print(f"   UID: {result.get('uid', 'N/A')}")
            print(f"   Version: {result.get('version', 'N/A')}")
            return True
        elif response.status_code == 401:
            print(f"❌ Authentication failed. Check username/password.")
            print(f"   Try accessing {GRAFANA_URL} and verify credentials.")
            return False
        elif response.status_code == 412:
            # Precondition failed - usually means dashboard already exists
            print(f"⚠️  Dashboard already exists. Updating...")
            return True
        else:
            print(f"❌ Import failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Grafana at {GRAFANA_URL}")
        print(f"   Make sure Grafana is running (check with: curl {GRAFANA_URL}/api/health)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error importing dashboard: {e}")
        return False

if __name__ == "__main__":
    success = import_dashboard()
    sys.exit(0 if success else 1)
