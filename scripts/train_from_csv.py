#!/usr/bin/env python3
"""
Train ML model directly from CSV files.
This is useful when historical data is in CSV format rather than in Prometheus.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nightowl_monitor.ml_leak_detector import MLLeakDetector


def main():
    parser = argparse.ArgumentParser(description="Train ML model from CSV file")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--model-path", required=True, help="Output path for trained model")
    parser.add_argument("--contamination", type=float, default=0.05, help="Contamination parameter")
    parser.add_argument("--device-id", help="Device ID (extracted from CSV if not provided)")
    
    args = parser.parse_args()
    
    print(f"Loading CSV data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Use 'datetime' column as 'timestamp' and drop the Unix timestamp column
    if 'datetime' in df.columns and 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        df = df.rename(columns={'datetime': 'timestamp'})
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize detector
    detector = MLLeakDetector(
        contamination=args.contamination,
    )
    
    print(f"\nTraining model with contamination={args.contamination}...")
    detector.train(df)
    
    # Save model
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"   Training samples: {len(df):,}")


if __name__ == "__main__":
    main()
