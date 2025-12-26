#!/usr/bin/env python3
"""Test script for NightOwl LLM Analyzer.

Usage:
    # First, get the Grafana Cloud token from the K8s secret
    export GRAFANA_CLOUD_USER="1953220"
    export GRAFANA_CLOUD_TOKEN="<token-from-k8s-secret>"
    
    # Make sure Ollama is running locally
    ollama serve &
    ollama pull qwen2.5:14b  # or llama3.2:3b for faster testing
    
    # Run the test
    python scripts/test_llm_analyzer.py
    
    # Ask a specific question
    python scripts/test_llm_analyzer.py "Is there a leak right now?"
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nightowl_monitor.llm_analyzer import LLMConfig, NightOwlLLMAnalyzer


def main():
    print("=" * 60)
    print("NightOwl LLM Analyzer Test")
    print("=" * 60)
    
    # Check environment
    user = os.environ.get("GRAFANA_CLOUD_USER", "")
    token = os.environ.get("GRAFANA_CLOUD_TOKEN", "")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
    
    print(f"\nConfiguration:")
    print(f"  Ollama URL: {ollama_url}")
    print(f"  Ollama Model: {model}")
    print(f"  Grafana Cloud User: {user or '(not set)'}")
    print(f"  Grafana Cloud Token: {'*' * 20 if token else '(not set)'}")
    
    # Create config and analyzer
    config = LLMConfig(
        ollama_url=ollama_url,
        ollama_model=model,
        grafana_cloud_user=user,
        grafana_cloud_token=token,
    )
    analyzer = NightOwlLLMAnalyzer(config)
    
    # Check availability
    print(f"\nComponent Status:")
    status = analyzer.is_available()
    print(f"  Ollama ({model}): {'✓ Available' if status['ollama'] else '✗ Not available'}")
    print(f"  Grafana Cloud: {'✓ Configured' if status['grafana_cloud'] else '✗ Not configured'}")
    
    if not status["ollama"]:
        print("\n⚠️  Ollama is not available!")
        print("   Make sure Ollama is running: ollama serve")
        print(f"   And the model is pulled: ollama pull {model}")
        print("\n   For faster testing, try a smaller model:")
        print("   export OLLAMA_MODEL=llama3.2:3b")
        return 1
    
    if not status["grafana_cloud"]:
        print("\n⚠️  Grafana Cloud credentials not set!")
        print("   Get the token from K8s:")
        print("   kubectl get secret grafana-cloud-nightowl -n solardashboard -o jsonpath='{.data.read-token}' | base64 -d")
        print("\n   Then set:")
        print("   export GRAFANA_CLOUD_USER='1953220'")
        print("   export GRAFANA_CLOUD_TOKEN='<token>'")
        return 1
    
    # Get question from args
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    
    if question:
        print(f"\nQuestion: {question}")
    else:
        print("\nNo question provided, using default status check.")
    
    print("\n" + "-" * 60)
    print("Fetching context from Grafana Cloud...")
    
    # Test context gathering first
    print("\nCurrent Status:")
    current = analyzer.context.get_current_status()
    if current:
        for key, value in current.items():
            print(f"  {key}: {value}")
    else:
        print("  (no data retrieved)")
    
    print("\nRecent History (24h):")
    history = analyzer.context.get_recent_history(hours=24)
    for metric, stats in history.items():
        if metric != "hours" and isinstance(stats, dict) and stats.get("available"):
            print(f"  {metric}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
    
    print("\nAnomaly Events (7 days):")
    events = analyzer.context.get_anomaly_events(days=7)
    if events:
        for event in events[-3:]:
            print(f"  {event['start']} ({event['duration_minutes']:.0f} min)")
    else:
        print("  None")
    
    # Now ask the LLM
    print("\n" + "-" * 60)
    print("Asking the LLM...")
    print("-" * 60)
    
    result = analyzer.analyze_current_state(question)
    
    if result["success"]:
        print(f"\n📊 Question: {result['question']}")
        print(f"\n🤖 Answer:\n{result['answer']}")
        print(f"\n⏱️  Generated in {result.get('duration_ms', 0):.0f}ms using {result['model']}")
    else:
        print(f"\n❌ Failed to generate response")
        print(f"   Error: {result.get('answer', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
