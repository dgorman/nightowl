#!/bin/bash
# Build and deploy nightowl to local K8s (Docker Desktop)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🏗️  Building NightOwl Docker image for local development..."

# Build NightOwl image with 'dev' tag
docker build -t nightowl-monitor:dev -f Dockerfile .

echo "✅ Docker image built successfully"

# Ensure we're using docker-desktop context
echo "🔧 Setting kubectl context to docker-desktop..."
kubectl config use-context docker-desktop

# Create namespace if it doesn't exist
echo "📦 Creating nightowl namespace..."
kubectl create namespace nightowl --dry-run=client -o yaml | kubectl apply -f -

# Apply dev overlay using Kustomize
echo "🚀 Deploying NightOwl to local Kubernetes..."
kubectl apply -k k8s/overlays/dev

# Wait for rollout
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/nightowl-monitor -n nightowl --timeout=60s

echo ""
echo "✅ NightOwl deployment complete!"
echo ""
echo "🌐 Access NightOwl metrics at: http://localhost:8010/metrics"
echo ""
echo "📊 View pods:    kubectl get pods -n nightowl"
echo "📝 View logs:    kubectl logs -n nightowl -l app=nightowl-monitor -f"
echo "🔍 Port forward: kubectl port-forward -n nightowl svc/nightowl-monitor 8010:8010"
