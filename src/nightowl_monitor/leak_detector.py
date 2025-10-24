"""
Leak detection analytics for NightOwl water monitoring system.

Analyzes historical data to detect abnormal water usage patterns that may indicate leaks.
Uses statistical analysis comparing 14-day baseline against recent behavior.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LeakDetectionResult:
    """Result of leak detection analysis."""
    
    leak_probability: float  # 0.0 to 1.0
    anomaly_score: float  # Standard deviations from baseline
    triggers: List[str]  # List of triggered detection rules
    metrics: Dict[str, float]  # Key metrics that contributed to detection
    timestamp: datetime
    confidence: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "leak_probability": round(self.leak_probability * 100, 2),
            "anomaly_score": round(self.anomaly_score, 2),
            "triggers": self.triggers,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "is_leak_detected": self.leak_probability >= 0.7,
        }


@dataclass
class BaselineStats:
    """Statistical baseline calculated from historical data."""
    
    mean: float
    std_dev: float
    median: float
    p25: float  # 25th percentile
    p75: float  # 75th percentile
    min_val: float
    max_val: float
    sample_count: int


class LeakDetector:
    """
    Analyzes water usage patterns to detect potential leaks.
    
    Detection strategies:
    1. Water level drop rate anomaly
    2. Total gallons usage deviation
    3. Pump duty cycle increase
    4. Pressure drop rate anomaly
    5. Off-hours consumption anomaly
    """
    
    def __init__(
        self,
        baseline_days: int = 14,
        comparison_hours: int = 24,
        z_score_threshold: float = 2.5,
        leak_probability_threshold: float = 0.7,
    ):
        """
        Initialize leak detector.
        
        Args:
            baseline_days: Days of historical data to use for baseline
            comparison_hours: Recent hours to compare against baseline
            z_score_threshold: Z-score threshold for anomaly detection
            leak_probability_threshold: Probability threshold for leak alert
        """
        self.baseline_days = baseline_days
        self.comparison_hours = comparison_hours
        self.z_score_threshold = z_score_threshold
        self.leak_probability_threshold = leak_probability_threshold
        
    def calculate_baseline_stats(self, values: List[float]) -> Optional[BaselineStats]:
        """Calculate statistical baseline from historical values."""
        if not values or len(values) < 10:
            logger.warning("Insufficient data for baseline calculation")
            return None
            
        # Remove None/null values
        clean_values = [v for v in values if v is not None]
        if len(clean_values) < 10:
            return None
            
        # Calculate statistics
        sorted_values = sorted(clean_values)
        n = len(sorted_values)
        
        mean = sum(sorted_values) / n
        variance = sum((x - mean) ** 2 for x in sorted_values) / n
        std_dev = variance ** 0.5
        
        median = sorted_values[n // 2] if n % 2 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        p25 = sorted_values[n // 4]
        p75 = sorted_values[(3 * n) // 4]
        
        return BaselineStats(
            mean=mean,
            std_dev=std_dev,
            median=median,
            p25=p25,
            p75=p75,
            min_val=sorted_values[0],
            max_val=sorted_values[-1],
            sample_count=n,
        )
    
    def calculate_z_score(self, value: float, baseline: BaselineStats) -> float:
        """Calculate z-score (standard deviations from mean)."""
        if baseline.std_dev == 0:
            return 0.0
        return abs(value - baseline.mean) / baseline.std_dev
    
    def detect_level_drop_anomaly(
        self,
        recent_level_drops: List[float],
        baseline_level_drops: List[float],
    ) -> Tuple[bool, float, Dict]:
        """
        Detect abnormal water level drop rate.
        
        A leak would show faster-than-normal level drops during periods
        when the pump is not running.
        """
        baseline = self.calculate_baseline_stats(baseline_level_drops)
        if not baseline or not recent_level_drops:
            return False, 0.0, {}
            
        # Calculate average drop rate in recent period
        recent_avg_drop = sum(recent_level_drops) / len(recent_level_drops)
        
        z_score = self.calculate_z_score(recent_avg_drop, baseline)
        is_anomaly = z_score > self.z_score_threshold and recent_avg_drop > baseline.mean
        
        return is_anomaly, z_score, {
            "recent_avg_drop_rate": round(recent_avg_drop, 4),
            "baseline_avg_drop_rate": round(baseline.mean, 4),
            "z_score": round(z_score, 2),
        }
    
    def detect_gallons_usage_anomaly(
        self,
        recent_daily_usage: List[float],
        baseline_daily_usage: List[float],
    ) -> Tuple[bool, float, Dict]:
        """
        Detect abnormal total gallons usage.
        
        A leak would show consistently higher daily water consumption.
        """
        baseline = self.calculate_baseline_stats(baseline_daily_usage)
        if not baseline or not recent_daily_usage:
            return False, 0.0, {}
            
        recent_avg_usage = sum(recent_daily_usage) / len(recent_daily_usage)
        
        z_score = self.calculate_z_score(recent_avg_usage, baseline)
        is_anomaly = z_score > self.z_score_threshold and recent_avg_usage > baseline.mean
        
        # Calculate percentage increase
        pct_increase = ((recent_avg_usage - baseline.mean) / baseline.mean * 100) if baseline.mean > 0 else 0
        
        return is_anomaly, z_score, {
            "recent_avg_daily_gallons": round(recent_avg_usage, 2),
            "baseline_avg_daily_gallons": round(baseline.mean, 2),
            "percent_increase": round(pct_increase, 1),
            "z_score": round(z_score, 2),
        }
    
    def detect_pump_duty_cycle_anomaly(
        self,
        recent_pump_runtime_pct: List[float],
        baseline_pump_runtime_pct: List[float],
    ) -> Tuple[bool, float, Dict]:
        """
        Detect abnormal pump duty cycle.
        
        A leak would cause the pump to run more frequently to maintain pressure.
        """
        baseline = self.calculate_baseline_stats(baseline_pump_runtime_pct)
        if not baseline or not recent_pump_runtime_pct:
            return False, 0.0, {}
            
        recent_avg_runtime = sum(recent_pump_runtime_pct) / len(recent_pump_runtime_pct)
        
        z_score = self.calculate_z_score(recent_avg_runtime, baseline)
        is_anomaly = z_score > self.z_score_threshold and recent_avg_runtime > baseline.mean
        
        return is_anomaly, z_score, {
            "recent_avg_pump_duty_cycle_pct": round(recent_avg_runtime, 2),
            "baseline_avg_pump_duty_cycle_pct": round(baseline.mean, 2),
            "z_score": round(z_score, 2),
        }
    
    def detect_pressure_drop_anomaly(
        self,
        recent_pressure_drops: List[float],
        baseline_pressure_drops: List[float],
    ) -> Tuple[bool, float, Dict]:
        """
        Detect abnormal pressure drop rate.
        
        A leak would show faster pressure drops when pump is off.
        """
        baseline = self.calculate_baseline_stats(baseline_pressure_drops)
        if not baseline or not recent_pressure_drops:
            return False, 0.0, {}
            
        recent_avg_drop = sum(recent_pressure_drops) / len(recent_pressure_drops)
        
        z_score = self.calculate_z_score(recent_avg_drop, baseline)
        is_anomaly = z_score > self.z_score_threshold and recent_avg_drop > baseline.mean
        
        return is_anomaly, z_score, {
            "recent_avg_pressure_drop_psi_per_min": round(recent_avg_drop, 3),
            "baseline_avg_pressure_drop_psi_per_min": round(baseline.mean, 3),
            "z_score": round(z_score, 2),
        }
    
    def detect_off_hours_usage_anomaly(
        self,
        recent_night_usage: List[float],
        baseline_night_usage: List[float],
    ) -> Tuple[bool, float, Dict]:
        """
        Detect abnormal night-time water usage.
        
        Unexpected usage during 11pm-5am when consumption should be minimal
        is a strong indicator of a leak.
        """
        baseline = self.calculate_baseline_stats(baseline_night_usage)
        if not baseline or not recent_night_usage:
            return False, 0.0, {}
            
        recent_avg_night = sum(recent_night_usage) / len(recent_night_usage)
        
        # Off-hours anomaly is more sensitive (lower threshold)
        z_score = self.calculate_z_score(recent_avg_night, baseline)
        is_anomaly = z_score > (self.z_score_threshold * 0.7) and recent_avg_night > baseline.mean
        
        return is_anomaly, z_score, {
            "recent_avg_night_usage_gallons": round(recent_avg_night, 2),
            "baseline_avg_night_usage_gallons": round(baseline.mean, 2),
            "z_score": round(z_score, 2),
        }
    
    def analyze(
        self,
        level_drops_recent: List[float],
        level_drops_baseline: List[float],
        daily_usage_recent: List[float],
        daily_usage_baseline: List[float],
        pump_runtime_recent: List[float],
        pump_runtime_baseline: List[float],
        pressure_drops_recent: List[float],
        pressure_drops_baseline: List[float],
        night_usage_recent: List[float],
        night_usage_baseline: List[float],
    ) -> LeakDetectionResult:
        """
        Perform comprehensive leak detection analysis.
        
        Returns a LeakDetectionResult with leak probability, triggers, and metrics.
        """
        triggers = []
        all_metrics = {}
        z_scores = []
        
        # Run all detection strategies
        checks = [
            ("level_drop", self.detect_level_drop_anomaly(level_drops_recent, level_drops_baseline)),
            ("gallons_usage", self.detect_gallons_usage_anomaly(daily_usage_recent, daily_usage_baseline)),
            ("pump_duty_cycle", self.detect_pump_duty_cycle_anomaly(pump_runtime_recent, pump_runtime_baseline)),
            ("pressure_drop", self.detect_pressure_drop_anomaly(pressure_drops_recent, pressure_drops_baseline)),
            ("off_hours_usage", self.detect_off_hours_usage_anomaly(night_usage_recent, night_usage_baseline)),
        ]
        
        for check_name, (is_anomaly, z_score, metrics) in checks:
            if metrics:  # Only process if we got valid metrics
                all_metrics.update({f"{check_name}_{k}": v for k, v in metrics.items()})
                z_scores.append(z_score)
                
                if is_anomaly:
                    triggers.append(check_name)
        
        # Calculate overall leak probability
        # Weight factors: more triggers = higher probability
        trigger_weight = len(triggers) / len(checks) if checks else 0
        
        # Average z-score indicates severity
        avg_z_score = sum(z_scores) / len(z_scores) if z_scores else 0
        z_score_weight = min(avg_z_score / (self.z_score_threshold * 2), 1.0)
        
        # Combined probability (weighted average)
        leak_probability = (trigger_weight * 0.6) + (z_score_weight * 0.4)
        
        # Determine confidence level
        if len(triggers) >= 3 and avg_z_score > self.z_score_threshold * 1.5:
            confidence = "high"
        elif len(triggers) >= 2 or avg_z_score > self.z_score_threshold:
            confidence = "medium"
        else:
            confidence = "low"
        
        return LeakDetectionResult(
            leak_probability=leak_probability,
            anomaly_score=avg_z_score,
            triggers=triggers,
            metrics=all_metrics,
            timestamp=datetime.utcnow(),
            confidence=confidence,
        )


class PrometheusLeakDetector:
    """
    Leak detector that queries Prometheus for historical data.
    
    This is designed to work with your existing Prometheus setup,
    querying the nightowl_* metrics.
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize with Prometheus endpoint.
        
        Args:
            prometheus_url: Base URL for Prometheus API
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.detector = LeakDetector()
        
    async def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "5m",
    ) -> List[Tuple[datetime, float]]:
        """
        Query Prometheus for time-series data.
        
        Args:
            query: PromQL query
            start_time: Start of time range
            end_time: End of time range
            step: Query resolution
            
        Returns:
            List of (timestamp, value) tuples
        """
        # This would use aiohttp or requests to query Prometheus
        # Placeholder for now - implement with actual HTTP client
        logger.info(f"Querying Prometheus: {query}")
        return []
    
    async def run_leak_detection(self, device_id: str) -> LeakDetectionResult:
        """
        Run leak detection for a specific device.
        
        Queries Prometheus for the past 14 days + 1 day of data,
        then performs statistical analysis.
        """
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=15)
        baseline_end = now - timedelta(days=1)
        recent_start = now - timedelta(hours=24)
        
        logger.info(f"Running leak detection for device {device_id}")
        logger.info(f"Baseline period: {baseline_start} to {baseline_end}")
        logger.info(f"Recent period: {recent_start} to {now}")
        
        # Query metrics from Prometheus
        # These are placeholder queries - adjust based on your actual metric names
        
        # 1. Water level drops (rate of Level1_precent decrease)
        level_query = f'rate(nightowl_telemetry_value{{device_id="{device_id}",key="Level1_precent"}}[5m])'
        
        # 2. Daily water usage (Pulse_TotalGallons delta)
        usage_query = f'increase(nightowl_telemetry_value{{device_id="{device_id}",key="Pulse_TotalGallons"}}[1d])'
        
        # 3. Pump duty cycle (P1C1 current > 0 indicates pump running)
        pump_query = f'avg_over_time((nightowl_telemetry_value{{device_id="{device_id}",key="P1C1"}} > 0)[1h:])'
        
        # 4. Pressure drop rate
        pressure_query = f'rate(nightowl_telemetry_value{{device_id="{device_id}",key="Pressure"}}[5m])'
        
        # In a real implementation, you'd query these and extract the data
        # For now, return a placeholder result
        
        return LeakDetectionResult(
            leak_probability=0.0,
            anomaly_score=0.0,
            triggers=[],
            metrics={"status": "not_implemented"},
            timestamp=now,
            confidence="low",
        )
