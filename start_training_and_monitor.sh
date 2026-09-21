#!/usr/bin/env bash
# ==============================================================================
# 🛰️ 8-Fold Cross-Validation Anomaly Detection Training & Remote Monitor
# ==============================================================================

PYTHON_BIN="/home/n3uron/miniconda3/envs/PINN_CFD/bin/python"
WORKSPACE="/home/n3uron/workspace/Anomaly_Detection"
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "100.107.39.95")

cd "$WORKSPACE" || exit 1

echo "======================================================================"
echo "🛰️  8-FOLD MULTI-CLASS VIDEO ANOMALY DETECTION LAUNCHER"
echo "======================================================================"
echo "• Tailscale VPN IP: $TAILSCALE_IP"
echo "• AMOLED Mobile Monitor: http://${TAILSCALE_IP}:8080"
echo "• TensorBoard Dashboard: http://${TAILSCALE_IP}:6006"
echo "======================================================================"

mkdir -p results

# Run 8-fold training with full arguments
"$PYTHON_BIN" -u scripts/train_multiclass_8fold.py \
  --features-dir data/unified_features \
  --folds 8 \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001 \
  --device cuda \
  "$@" 2>&1 | tee results/training.log
