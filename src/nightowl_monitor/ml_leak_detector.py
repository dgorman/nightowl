"""
ML-based Leak Detection for NightOwl water monitoring system.

Uses Isolation Forest for unsupervised anomaly detection, trained on historical
telemetry data from Prometheus. Provides more sophisticated leak detection compared
to pure statistical z-score analysis.

Features:
- Isolation Forest for multi-variate anomaly detection
- Rolling window feature engineering
- Time-based features (hour of day, day of week)
- Model persistence for production deployment
- Integration with Prometheus for data retrieval and metric export
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class MLLeakDetectionResult:
    """Result of ML-based leak detection analysis."""

    anomaly_score: float  # -1 (anomaly) to 1 (normal) from Isolation Forest
    is_anomaly: bool  # True if classified as anomaly
    leak_probability: float  # 0.0 to 1.0, normalized probability
    feature_contributions: Dict[str, float]  # Feature importance for this prediction
    raw_features: Dict[str, float]  # Input features used for prediction
    model_version: str  # Model identifier/version
    timestamp: datetime
    confidence: str  # "low", "medium", "high" based on training data quality

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomaly": self.is_anomaly,
            "leak_probability": round(self.leak_probability * 100, 2),
            "feature_contributions": {
                k: round(v, 4) for k, v in self.feature_contributions.items()
            },
            "raw_features": {k: round(v, 4) for k, v in self.raw_features.items()},
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Rolling window sizes (in minutes)
    rolling_windows: List[int] = field(default_factory=lambda: [5, 15, 60, 360, 1440])
    
    # Feature names from telemetry
    telemetry_features: List[str] = field(
        default_factory=lambda: [
            "Level1_precent",
            "Pulse_TotalGallons",
            "P1C1",  # Pump 1 current
            "P1V1",  # Pump 1 voltage
            "Pressure",
        ]
    )
    
    # Whether to include time-based features
    include_time_features: bool = True
    
    # Whether to include rate-of-change features
    include_rate_features: bool = True


class FeatureEngineer:
    """
    Transforms raw telemetry data into ML features.
    
    Creates time-based, rolling window, and rate-of-change features
    suitable for anomaly detection.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time-based features."""
        if "timestamp" not in df.columns:
            return df

        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])

        # Hour of day (cyclical encoding)
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

        # Day of week (cyclical encoding)
        df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        # Is night time (11pm - 5am) - leaks are more obvious at night
        df["is_night"] = ((ts.dt.hour >= 23) | (ts.dt.hour < 5)).astype(float)

        # Is weekend
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        df = df.copy()

        for feature in self.config.telemetry_features:
            if feature not in df.columns:
                continue

            for window in self.config.rolling_windows:
                # Rolling mean
                df[f"{feature}_roll_{window}m_mean"] = (
                    df[feature].rolling(window=window, min_periods=1).mean()
                )
                # Rolling std
                df[f"{feature}_roll_{window}m_std"] = (
                    df[feature].rolling(window=window, min_periods=1).std().fillna(0)
                )
                # Rolling min/max range
                df[f"{feature}_roll_{window}m_range"] = (
                    df[feature].rolling(window=window, min_periods=1).max()
                    - df[feature].rolling(window=window, min_periods=1).min()
                )

        return df

    def _add_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate-of-change features."""
        df = df.copy()

        for feature in self.config.telemetry_features:
            if feature not in df.columns:
                continue

            # 1-minute rate of change
            df[f"{feature}_rate_1m"] = df[feature].diff().fillna(0)

            # 5-minute rate of change
            df[f"{feature}_rate_5m"] = df[feature].diff(periods=5).fillna(0)

            # 1-hour rate of change
            df[f"{feature}_rate_60m"] = df[feature].diff(periods=60).fillna(0)

        return df

    def _add_pump_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pump-specific features for leak detection."""
        df = df.copy()

        # Pump running indicator (current > 0.5A)
        if "P1C1" in df.columns:
            df["pump_running"] = (df["P1C1"] > 0.5).astype(float)

            # Time since last pump cycle (approximate)
            pump_changes = df["pump_running"].diff().abs()
            df["pump_cycle_count_1h"] = (
                pump_changes.rolling(window=60, min_periods=1).sum()
            )

        # Level drop while pump is off (strong leak indicator)
        if "Level1_precent" in df.columns and "pump_running" in df.columns:
            level_drop = df["Level1_precent"].diff().clip(upper=0).abs()
            df["level_drop_pump_off"] = level_drop * (1 - df["pump_running"])

        return df

    def transform(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform raw telemetry dataframe into feature array.
        
        Args:
            df: DataFrame with timestamp and telemetry columns
            fit_scaler: Whether to fit the scaler (True for training)
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        # Add engineered features
        if self.config.include_time_features:
            df = self._add_time_features(df)
        if self.config.include_rate_features:
            df = self._add_rate_features(df)
        df = self._add_rolling_features(df)
        df = self._add_pump_features(df)

        # Select only numeric columns (excluding timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "timestamp" in numeric_cols:
            numeric_cols.remove("timestamp")

        # Remove columns with all NaN
        valid_cols = [c for c in numeric_cols if df[c].notna().any()]

        # Fill remaining NaN with 0 (or forward fill for time series)
        feature_df = df[valid_cols].fillna(method="ffill").fillna(0)

        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(feature_df)
            self.feature_names = valid_cols
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call transform with fit_scaler=True first.")
            # Handle case where columns might differ
            available_cols = [c for c in self.feature_names if c in feature_df.columns]
            missing_cols = [c for c in self.feature_names if c not in feature_df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns during transform: {missing_cols}")
                for col in missing_cols:
                    feature_df[col] = 0
                    
            feature_df = feature_df[self.feature_names]
            features = self.scaler.transform(feature_df)

        return features, self.feature_names


class MLLeakDetector:
    """
    ML-based leak detector using Isolation Forest.
    
    Isolation Forest is ideal for this use case because:
    1. Works with unlabeled data (leaks are rare, hard to get labeled examples)
    2. Detects anomalies by isolating outliers
    3. Fast training and inference
    4. Works well with multivariate data
    """

    MODEL_VERSION = "1.0.0"
    DEFAULT_MODEL_PATH = Path("/var/lib/nightowl/models/leak_detector.joblib")

    def __init__(
        self,
        contamination: float = 0.05,  # Expected fraction of anomalies
        n_estimators: int = 100,
        random_state: int = 42,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize the ML leak detector.
        
        Args:
            contamination: Expected proportion of anomalies in training data
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            model_path: Path to save/load the model
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_path = model_path or self.DEFAULT_MODEL_PATH

        self.model: Optional[IsolationForest] = None
        self.feature_engineer = FeatureEngineer()
        self.training_stats: Dict[str, Any] = {}
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained or loaded."""
        return self._is_trained and self.model is not None

    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Train the Isolation Forest model on historical data.
        
        Args:
            training_data: DataFrame with timestamp and telemetry columns
            validation_data: Optional validation set for evaluation
            
        Returns:
            Training statistics and metrics
        """
        logger.info(f"Training ML leak detector on {len(training_data)} samples")

        # Feature engineering (fit scaler on training data)
        features, feature_names = self.feature_engineer.transform(
            training_data, fit_scaler=True
        )

        logger.info(f"Generated {len(feature_names)} features")

        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0,
        )
        self.model.fit(features)
        self._is_trained = True

        # Calculate training statistics
        train_predictions = self.model.predict(features)
        train_scores = self.model.decision_function(features)

        anomaly_count = (train_predictions == -1).sum()
        anomaly_rate = anomaly_count / len(train_predictions)

        self.training_stats = {
            "training_samples": len(training_data),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "anomaly_count": int(anomaly_count),
            "anomaly_rate": float(anomaly_rate),
            "contamination": self.contamination,
            "score_mean": float(np.mean(train_scores)),
            "score_std": float(np.std(train_scores)),
            "trained_at": datetime.utcnow().isoformat(),
            "model_version": self.MODEL_VERSION,
        }

        # Validation if provided
        if validation_data is not None and len(validation_data) > 0:
            val_features, _ = self.feature_engineer.transform(
                validation_data, fit_scaler=False
            )
            val_predictions = self.model.predict(val_features)
            val_anomaly_rate = (val_predictions == -1).sum() / len(val_predictions)
            self.training_stats["validation_samples"] = len(validation_data)
            self.training_stats["validation_anomaly_rate"] = float(val_anomaly_rate)

        logger.info(f"Training complete. Anomaly rate: {anomaly_rate:.2%}")
        return self.training_stats

    def predict(self, current_data: pd.DataFrame) -> MLLeakDetectionResult:
        """
        Predict leak probability for current telemetry data.
        
        Args:
            current_data: Recent telemetry data (last few hours recommended)
            
        Returns:
            MLLeakDetectionResult with anomaly score and probability
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Use the most recent row for prediction
        if len(current_data) == 0:
            raise ValueError("Empty dataframe provided")

        # Feature engineering
        features, feature_names = self.feature_engineer.transform(
            current_data, fit_scaler=False
        )

        # Get the last row (most recent data point)
        latest_features = features[-1:, :]

        # Predict
        prediction = self.model.predict(latest_features)[0]
        decision_score = self.model.decision_function(latest_features)[0]

        # Normalize score to probability (0 to 1, where 1 = definitely anomaly)
        # Decision function returns negative values for anomalies
        # Typical range is roughly -0.5 to 0.5, but can vary
        # We normalize based on training statistics
        score_mean = self.training_stats.get("score_mean", 0)
        score_std = self.training_stats.get("score_std", 0.1)
        
        # Z-score normalized probability
        z_score = (score_mean - decision_score) / max(score_std, 0.01)
        leak_probability = min(max(z_score / 4, 0), 1)  # Clip to 0-1

        # Calculate feature contributions (approximate via perturbation)
        feature_contributions = self._calculate_feature_contributions(
            latest_features, feature_names
        )

        # Build raw features dict
        raw_features = {
            name: float(latest_features[0, i])
            for i, name in enumerate(feature_names)
            if i < latest_features.shape[1]
        }

        # Determine confidence based on training data quality
        if self.training_stats.get("training_samples", 0) >= 20000:
            confidence = "high"
        elif self.training_stats.get("training_samples", 0) >= 5000:
            confidence = "medium"
        else:
            confidence = "low"

        return MLLeakDetectionResult(
            anomaly_score=float(decision_score),
            is_anomaly=(prediction == -1),
            leak_probability=leak_probability,
            feature_contributions=feature_contributions,
            raw_features=raw_features,
            model_version=self.MODEL_VERSION,
            timestamp=datetime.utcnow(),
            confidence=confidence,
        )

    def _calculate_feature_contributions(
        self, features: np.ndarray, feature_names: List[str], n_top: int = 10
    ) -> Dict[str, float]:
        """
        Approximate feature contributions to the anomaly score.
        
        Uses a simple perturbation-based approach: for each feature,
        calculate how much the score changes when that feature is set to mean.
        """
        base_score = self.model.decision_function(features)[0]
        contributions = {}

        # Get feature means from training (using scaler)
        feature_means = self.feature_engineer.scaler.mean_

        for i, name in enumerate(feature_names):
            if i >= features.shape[1]:
                break

            # Create perturbed version with this feature set to mean
            perturbed = features.copy()
            perturbed[0, i] = feature_means[i]

            perturbed_score = self.model.decision_function(perturbed)[0]
            contribution = base_score - perturbed_score
            contributions[name] = float(contribution)

        # Sort by absolute contribution and return top N
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:n_top]
        )
        return sorted_contributions

    def save(self, path: Optional[Path] = None) -> None:
        """Save the trained model and feature engineer to disk."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "training_stats": self.training_stats,
            "model_version": self.MODEL_VERSION,
        }
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        load_path = path or self.model_path
        if not load_path.exists():
            logger.warning(f"Model file not found: {load_path}")
            return False

        try:
            model_data = joblib.load(load_path)
            self.model = model_data["model"]
            self.feature_engineer = model_data["feature_engineer"]
            self.training_stats = model_data["training_stats"]
            self._is_trained = True
            logger.info(f"Model loaded from {load_path}")
            logger.info(f"Model version: {model_data.get('model_version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class PrometheusDataFetcher:
    """
    Fetches historical telemetry data from Prometheus.
    
    Used for training the ML model on historical data.
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        timeout_seconds: int = 30,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.timeout = timeout_seconds
        self.auth = (username, password) if username and password else None

    def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "1m",
    ) -> pd.DataFrame:
        """
        Query Prometheus for time-series data.
        
        Args:
            query: PromQL query
            start_time: Start of time range
            end_time: End of time range
            step: Query resolution
            
        Returns:
            DataFrame with timestamp and value columns
        """
        import requests

        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step,
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout, auth=self.auth)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return pd.DataFrame()

        if data.get("status") != "success":
            logger.error(f"Prometheus query error: {data.get('error', 'unknown')}")
            return pd.DataFrame()

        results = data.get("data", {}).get("result", [])
        if not results:
            return pd.DataFrame()

        # Parse results into DataFrame
        rows = []
        for result in results:
            metric = result.get("metric", {})
            values = result.get("values", [])
            for ts, val in values:
                rows.append({
                    "timestamp": datetime.fromtimestamp(float(ts)),
                    "value": float(val) if val != "NaN" else None,
                    **metric,
                })

        return pd.DataFrame(rows)

    def discover_telemetry_keys(
        self,
        device_id: str,
        days: int = 30,
    ) -> List[str]:
        """
        Discover available telemetry keys for a device.
        
        Args:
            device_id: NightOwl device ID
            days: Number of days to look back
            
        Returns:
            List of available telemetry keys
        """
        import requests
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": f'count by (key) (nightowl_telemetry_value{{device_id="{device_id}"}})',
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": "1d",
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout, auth=self.auth)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to discover telemetry keys: {e}")
            return []
        
        if data.get("status") != "success":
            return []
        
        results = data.get("data", {}).get("result", [])
        keys = [r.get("metric", {}).get("key") for r in results]
        return [k for k in keys if k]

    def fetch_training_data(
        self,
        device_id: str,
        days: int = 30,
        telemetry_keys: Optional[List[str]] = None,
        step: str = "5m",
    ) -> pd.DataFrame:
        """
        Fetch training data for a specific device.
        
        Args:
            device_id: NightOwl device ID
            days: Number of days of historical data to fetch
            telemetry_keys: List of telemetry keys to fetch. If None, discovers available keys.
            step: Query resolution step
            
        Returns:
            DataFrame ready for training
        """
        if telemetry_keys is None:
            # Auto-discover available keys
            telemetry_keys = self.discover_telemetry_keys(device_id, days)
            if not telemetry_keys:
                logger.warning(f"No telemetry keys found for device {device_id}")
                return pd.DataFrame()
            logger.info(f"Discovered telemetry keys: {telemetry_keys}")

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        logger.info(f"Fetching {days} days of training data for device {device_id}")

        # Fetch each telemetry key
        dfs = []
        for key in telemetry_keys:
            query = f'nightowl_telemetry_value{{device_id="{device_id}",key="{key}"}}'
            df = self.query_range(query, start_time, end_time, step=step)
            if not df.empty:
                df = df.rename(columns={"value": key})
                df = df[["timestamp", key]]
                dfs.append(df)
                logger.info(f"  Fetched {len(df)} rows for key '{key}'")

        if not dfs:
            logger.warning("No data fetched from Prometheus")
            return pd.DataFrame()

        # Merge all dataframes on timestamp
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on="timestamp", how="outer")

        # Sort by timestamp and forward-fill missing values
        result = result.sort_values("timestamp").reset_index(drop=True)
        result = result.fillna(method="ffill")

        logger.info(f"Fetched {len(result)} data points")
        return result
