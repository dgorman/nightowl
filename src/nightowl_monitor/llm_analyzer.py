"""LLM-based leak analysis for NightOwl water monitoring.

This module provides natural language analysis of water usage patterns
and anomaly explanations using a local Ollama LLM.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM analyzer."""
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    
    # Grafana Cloud settings (for historical data)
    grafana_cloud_url: str = "https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom"
    grafana_cloud_user: str = ""
    grafana_cloud_token: str = ""
    
    # Local Prometheus (for recent data)
    local_prometheus_url: str = "http://prometheus.solardashboard.svc.cluster.local:9090"
    
    # Query settings
    max_context_hours: int = 168  # 7 days
    cache_ttl_seconds: int = 300  # 5 minutes
    
    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "LLMConfig":
        """Load configuration from environment variables."""
        source = env or os.environ
        return cls(
            ollama_url=source.get("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=source.get("OLLAMA_MODEL", "qwen2.5:14b"),
            grafana_cloud_url=source.get(
                "GRAFANA_CLOUD_PROMETHEUS_URL",
                "https://prometheus-prod-13-prod-us-east-0.grafana.net/api/prom"
            ),
            grafana_cloud_user=source.get("GRAFANA_CLOUD_USER", ""),
            grafana_cloud_token=source.get("GRAFANA_CLOUD_TOKEN", ""),
            local_prometheus_url=source.get(
                "NIGHTOWL_ML_PROMETHEUS_URL",
                "http://prometheus.solardashboard.svc.cluster.local:9090"
            ),
        )


class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, config: LLMConfig):
        self.url = config.ollama_url
        self.model = config.ollama_model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate a response from the LLM."""
        
        # Build the full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            }
        }
        
        try:
            req = Request(
                f"{self.url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": self.model,
                    "total_duration_ms": result.get("total_duration", 0) / 1_000_000,
                }
        except (URLError, HTTPError) as e:
            logger.error(f"Ollama request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
            }
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            req = Request(f"{self.url}/api/tags", method="GET")
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Check if our model is available (handle tag suffix)
                return any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False


class PrometheusClient:
    """Client for querying Prometheus (local and Grafana Cloud)."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def _get_grafana_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Grafana Cloud."""
        if not self.config.grafana_cloud_user or not self.config.grafana_cloud_token:
            return {}
        
        credentials = f"{self.config.grafana_cloud_user}:{self.config.grafana_cloud_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    
    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m",
        use_grafana_cloud: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a range query against Prometheus."""
        
        if use_grafana_cloud:
            base_url = self.config.grafana_cloud_url
            headers = self._get_grafana_auth_headers()
        else:
            base_url = self.config.local_prometheus_url
            headers = {}
        
        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }
        
        url = f"{base_url}/api/v1/query_range?{urlencode(params)}"
        
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "success":
                    return data.get("data", {}).get("result", [])
                else:
                    logger.warning(f"Prometheus query failed: {data}")
                    return []
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            return []
    
    def query_instant(
        self,
        query: str,
        use_grafana_cloud: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute an instant query against Prometheus."""
        
        if use_grafana_cloud:
            base_url = self.config.grafana_cloud_url
            headers = self._get_grafana_auth_headers()
        else:
            base_url = self.config.local_prometheus_url
            headers = {}
        
        params = {"query": query}
        url = f"{base_url}/api/v1/query?{urlencode(params)}"
        
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "success":
                    return data.get("data", {}).get("result", [])
                else:
                    logger.warning(f"Prometheus query failed: {data}")
                    return []
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            return []


class NightOwlContext:
    """Gathers context from NightOwl metrics for LLM analysis."""
    
    def __init__(self, prometheus: PrometheusClient, device_id: str = ".*"):
        self.prometheus = prometheus
        self.device_id = device_id
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current ML and sensor status."""
        queries = {
            "ml_probability": f'nightowl_ml_leak_probability{{device_id=~"{self.device_id}"}}',
            "ml_anomaly_score": f'nightowl_ml_anomaly_score{{device_id=~"{self.device_id}"}}',
            "ml_is_anomaly": f'nightowl_ml_is_anomaly{{device_id=~"{self.device_id}"}}',
            "pressure_s1": f'nightowl_telemetry_value{{device_id=~"{self.device_id}",key="S1"}}',
            "pump_current_p1c1": f'nightowl_telemetry_value{{device_id=~"{self.device_id}",key="P1C1"}}',
        }
        
        status = {}
        for name, query in queries.items():
            results = self.prometheus.query_instant(query, use_grafana_cloud=True)
            if results:
                status[name] = float(results[0].get("value", [0, 0])[1])
        
        return status
    
    def get_recent_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent sensor history for context."""
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        
        # Get pressure history
        pressure_data = self.prometheus.query_range(
            f'nightowl_telemetry_value{{device_id=~"{self.device_id}",key="S1"}}',
            start, end, step="5m",
            use_grafana_cloud=True
        )
        
        # Get pump activity
        pump_data = self.prometheus.query_range(
            f'nightowl_telemetry_value{{device_id=~"{self.device_id}",key="P1C1"}}',
            start, end, step="5m",
            use_grafana_cloud=True
        )
        
        # Get ML probability history
        ml_data = self.prometheus.query_range(
            f'nightowl_ml_leak_probability{{device_id=~"{self.device_id}"}}',
            start, end, step="5m",
            use_grafana_cloud=True
        )
        
        return {
            "pressure_history": self._summarize_timeseries(pressure_data),
            "pump_history": self._summarize_timeseries(pump_data),
            "ml_history": self._summarize_timeseries(ml_data),
            "hours": hours,
        }
    
    def _summarize_timeseries(self, data: List[Dict]) -> Dict[str, Any]:
        """Summarize a timeseries into statistics."""
        if not data or not data[0].get("values"):
            return {"available": False}
        
        values = [float(v[1]) for v in data[0].get("values", []) if v[1] != "NaN"]
        if not values:
            return {"available": False}
        
        return {
            "available": True,
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "current": values[-1] if values else None,
            "samples": len(values),
        }
    
    def get_anomaly_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent anomaly events."""
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        # Find times when ML flagged anomaly
        anomaly_data = self.prometheus.query_range(
            f'nightowl_ml_is_anomaly{{device_id=~"{self.device_id}"}} == 1',
            start, end, step="5m",
            use_grafana_cloud=True
        )
        
        events = []
        if anomaly_data and anomaly_data[0].get("values"):
            # Group consecutive anomaly points into events
            timestamps = [float(v[0]) for v in anomaly_data[0].get("values", [])]
            if timestamps:
                # Simple event detection: gaps > 30 min = new event
                current_event_start = timestamps[0]
                current_event_end = timestamps[0]
                
                for ts in timestamps[1:]:
                    if ts - current_event_end > 1800:  # 30 min gap
                        events.append({
                            "start": datetime.fromtimestamp(current_event_start).isoformat(),
                            "end": datetime.fromtimestamp(current_event_end).isoformat(),
                            "duration_minutes": (current_event_end - current_event_start) / 60,
                        })
                        current_event_start = ts
                    current_event_end = ts
                
                # Add last event
                events.append({
                    "start": datetime.fromtimestamp(current_event_start).isoformat(),
                    "end": datetime.fromtimestamp(current_event_end).isoformat(),
                    "duration_minutes": (current_event_end - current_event_start) / 60,
                })
        
        return events


class NightOwlLLMAnalyzer:
    """Main LLM analyzer for NightOwl leak detection."""
    
    SYSTEM_PROMPT = """You are an expert water system analyst for NightOwl, a well water monitoring system.
Your role is to help homeowners understand their water usage patterns and identify potential leaks.

Key metrics you analyze:
- S1 (Pressure): House water pressure in PSI. Normal range is 30-50 PSI. Drops indicate water usage.
- P1C1/P1C2/P1C3 (Pump Current): Well pump electrical current in Amps. >0 means pump is running.
- ML Leak Probability: 0-100%, based on Isolation Forest anomaly detection. >50% is flagged as anomaly.
- Anomaly Score: Negative = anomaly, Positive = normal. Raw output from ML model.

Common patterns:
- Normal usage: Pressure drops briefly, pump kicks in, pressure recovers
- Slow leak: Sustained low pressure, frequent pump cycles, elevated overnight activity
- Running toilet: Continuous small current draw, pressure doesn't fully recover
- Irrigation: Regular scheduled drops, longer pump runs

When analyzing:
1. Look at time patterns (day vs night, weekday vs weekend)
2. Consider seasonal factors
3. Check for gradual changes that might indicate developing issues
4. Be specific about what the data shows
5. Give actionable recommendations

Respond in a conversational but informative tone. Be direct about concerns but don't be alarmist."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.ollama = OllamaClient(self.config)
        self.prometheus = PrometheusClient(self.config)
        self.context = NightOwlContext(self.prometheus)
    
    def is_available(self) -> Dict[str, bool]:
        """Check if the analyzer components are available."""
        return {
            "ollama": self.ollama.is_available(),
            "grafana_cloud": bool(
                self.config.grafana_cloud_user and self.config.grafana_cloud_token
            ),
        }
    
    def analyze_current_state(self, question: Optional[str] = None) -> Dict[str, Any]:
        """Analyze current system state and optionally answer a question."""
        
        # Gather context
        current = self.context.get_current_status()
        history = self.context.get_recent_history(hours=24)
        events = self.context.get_anomaly_events(days=7)
        
        # Build context summary for the LLM
        context_text = self._build_context_text(current, history, events)
        
        # Default question if none provided
        if not question:
            question = "What is the current status of my water system? Are there any concerns?"
        
        # Generate response
        prompt = f"""Based on the following water system data:

{context_text}

User Question: {question}"""
        
        result = self.ollama.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=1000,
        )
        
        return {
            "question": question,
            "answer": result.get("response", "Unable to generate response"),
            "success": result.get("success", False),
            "context": {
                "current_status": current,
                "recent_history": history,
                "anomaly_events_7d": len(events),
            },
            "model": result.get("model"),
            "duration_ms": result.get("total_duration_ms"),
        }
    
    def explain_anomaly(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Explain why an anomaly was detected at a given time."""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Get context around the anomaly time
        start = timestamp - timedelta(hours=6)
        end = timestamp + timedelta(hours=1)
        
        # Query relevant metrics
        pressure_data = self.prometheus.query_range(
            f'nightowl_telemetry_value{{key="S1"}}',
            start, end, step="1m",
            use_grafana_cloud=True
        )
        
        pump_data = self.prometheus.query_range(
            f'nightowl_telemetry_value{{key="P1C1"}}',
            start, end, step="1m",
            use_grafana_cloud=True
        )
        
        ml_data = self.prometheus.query_range(
            f'nightowl_ml_leak_probability',
            start, end, step="1m",
            use_grafana_cloud=True
        )
        
        # Summarize the data
        context_text = f"""Anomaly Analysis for {timestamp.isoformat()}

Pressure (S1) around the event:
{self._format_timeseries_summary(pressure_data)}

Pump Current (P1C1) around the event:
{self._format_timeseries_summary(pump_data)}

ML Leak Probability around the event:
{self._format_timeseries_summary(ml_data)}
"""
        
        prompt = f"""{context_text}

Please explain:
1. What the ML model detected as unusual
2. Whether this appears to be a true anomaly or a false positive
3. What might have caused this pattern
4. Recommended actions if any"""
        
        result = self.ollama.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=1200,
        )
        
        return {
            "timestamp": timestamp.isoformat(),
            "explanation": result.get("response", "Unable to generate explanation"),
            "success": result.get("success", False),
            "model": result.get("model"),
        }
    
    def _build_context_text(
        self,
        current: Dict[str, Any],
        history: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> str:
        """Build context text for the LLM prompt."""
        lines = []
        
        # Current status
        lines.append("=== CURRENT STATUS ===")
        if "ml_probability" in current:
            prob = current["ml_probability"]
            status = "ANOMALY" if prob >= 50 else "NORMAL"
            lines.append(f"ML Leak Probability: {prob:.1f}% ({status})")
        if "ml_anomaly_score" in current:
            lines.append(f"Anomaly Score: {current['ml_anomaly_score']:.4f}")
        if "pressure_s1" in current:
            lines.append(f"Current Pressure (S1): {current['pressure_s1']:.1f} PSI")
        if "pump_current_p1c1" in current:
            pump = current["pump_current_p1c1"]
            pump_status = "RUNNING" if pump > 0.1 else "OFF"
            lines.append(f"Pump Current (P1C1): {pump:.3f} A ({pump_status})")
        
        # Recent history
        lines.append("\n=== LAST 24 HOURS ===")
        if history.get("pressure_history", {}).get("available"):
            ph = history["pressure_history"]
            lines.append(f"Pressure: min={ph['min']:.1f}, max={ph['max']:.1f}, avg={ph['mean']:.1f} PSI")
        if history.get("pump_history", {}).get("available"):
            pump_h = history["pump_history"]
            lines.append(f"Pump Activity: max current={pump_h['max']:.3f}A")
        if history.get("ml_history", {}).get("available"):
            ml_h = history["ml_history"]
            lines.append(f"ML Probability: min={ml_h['min']:.1f}%, max={ml_h['max']:.1f}%, avg={ml_h['mean']:.1f}%")
        
        # Recent events
        if events:
            lines.append(f"\n=== ANOMALY EVENTS (last 7 days): {len(events)} ===")
            for i, event in enumerate(events[-5:], 1):  # Show last 5
                lines.append(f"  {i}. {event['start']} ({event['duration_minutes']:.0f} min)")
        else:
            lines.append("\n=== ANOMALY EVENTS (last 7 days): None ===")
        
        return "\n".join(lines)
    
    def _format_timeseries_summary(self, data: List[Dict]) -> str:
        """Format timeseries data for LLM context."""
        if not data or not data[0].get("values"):
            return "  No data available"
        
        values = [(float(v[0]), float(v[1])) for v in data[0].get("values", []) if v[1] != "NaN"]
        if not values:
            return "  No valid data points"
        
        vals = [v[1] for v in values]
        return f"  Samples: {len(vals)}, Min: {min(vals):.2f}, Max: {max(vals):.2f}, Mean: {sum(vals)/len(vals):.2f}"


# CLI for testing
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load config from environment
    config = LLMConfig.from_env()
    analyzer = NightOwlLLMAnalyzer(config)
    
    # Check availability
    print("Checking component availability...")
    status = analyzer.is_available()
    print(f"  Ollama: {'✓' if status['ollama'] else '✗'}")
    print(f"  Grafana Cloud: {'✓' if status['grafana_cloud'] else '✗'}")
    
    if not status["ollama"]:
        print("\nOllama not available. Make sure it's running:")
        print("  ollama serve")
        print(f"  ollama pull {config.ollama_model}")
        sys.exit(1)
    
    if not status["grafana_cloud"]:
        print("\nGrafana Cloud credentials not configured. Set:")
        print("  GRAFANA_CLOUD_USER=1953220")
        print("  GRAFANA_CLOUD_TOKEN=<your-read-token>")
        sys.exit(1)
    
    # Get question from args or use default
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    
    print("\nAnalyzing current state...")
    result = analyzer.analyze_current_state(question)
    
    print(f"\n{'='*60}")
    print(f"Question: {result['question']}")
    print(f"{'='*60}")
    print(f"\n{result['answer']}")
    print(f"\n{'='*60}")
    print(f"Model: {result['model']}")
    print(f"Duration: {result.get('duration_ms', 0):.0f}ms")
    print(f"Success: {result['success']}")
