#!/usr/bin/env python3
"""Unified Class-Aware Video Anomaly Detection Training Orchestrator.

Trains on the 6 consolidated macro classes (+ Normal) using 8-Fold cross validation,
automatic TensorBoard logging, and real-time AMOLED mobile telemetry over Tailscale VPN.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from data.taxonomy import DEFAULT_MACRO_CLASSES, load_dataset_taxonomy
from model.head import AnomalyHead
from train.trainer import MultiClassTrainer as ClassAwareMILTrainer


def get_tailscale_ip() -> str:
    """Retrieves active Tailscale IP, falling back to local IP."""
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], stderr=subprocess.DEVNULL, text=True).strip()
        if out:
            return out.splitlines()[0]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_background_servers(log_dir: str, tb_port: int = 6006, monitor_port: int = 8080) -> list:
    """Spawns TensorBoard and AMOLED monitor server if not already active."""
    procs = []
    py_exec = sys.executable

    # 1. TensorBoard
    if not is_port_in_use(tb_port):
        os.makedirs(log_dir, exist_ok=True)
        tb_cmd = [
            py_exec,
            "-m",
            "tensorboard.main",
            f"--logdir={log_dir}",
            f"--port={tb_port}",
            "--bind_all",
        ]
        tb_proc = subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
        procs.append(tb_proc)
        print(f"  ✓ TensorBoard started on port {tb_port} (PID: {tb_proc.pid})")
    else:
        print(f"  ✓ TensorBoard already active on port {tb_port}")

    # 2. AMOLED Mobile Monitor Server
    if not is_port_in_use(monitor_port):
        monitor_script = os.path.join(os.path.dirname(__file__), "monitor_server.py")
        mon_cmd = [py_exec, monitor_script, str(monitor_port)]
        mon_proc = subprocess.Popen(mon_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
        procs.append(mon_proc)
        print(f"  ✓ AMOLED Monitor Server started on port {monitor_port} (PID: {mon_proc.pid})")
    else:
        print(f"  ✓ AMOLED Monitor Server already active on port {monitor_port}")

    return procs


def main():
    parser = argparse.ArgumentParser(description="Train 8-Fold Class-Aware Video Anomaly Detector")
    parser.add_argument("--features-dir", type=str, default="data/unified_features", help="Path to unified features directory")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--smoothness-weight", type=float, default=0.00001, help="Smoothness loss weight")
    parser.add_argument("--sparsity-weight", type=float, default=0.00001, help="Sparsity loss weight")
    parser.add_argument("--k-top", type=int, default=3, help="Top-k anomalous segments for MIL pooling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/class_aware_8fold", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="runs/class_aware_8fold", help="TensorBoard log directory")
    parser.add_argument("--state-file", type=str, default="results/training_state.json", help="Live telemetry JSON file")
    parser.add_argument("--tb-port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--monitor-port", type=int, default=8080, help="AMOLED Monitor port")
    parser.add_argument("--skip-servers", action="store_true", help="Skip spawning background monitor servers")

    args = parser.parse_args()

    ip = get_tailscale_ip()

    taxonomy = load_dataset_taxonomy(args.features_dir)
    feat_dim = taxonomy["feature_dim"]
    class_list = taxonomy["anomaly_classes"]
    num_classes = len(class_list)

    print("\n" + "=" * 70)
    print(f"🛰️  ANOMALY DETECTION: {args.folds}-FOLD CLASS-AWARE TRAINING PIPELINE")
    print("=" * 70)
    print(f"• Dataset Features:     {args.features_dir}")
    print(f"• Macro Classes:        {num_classes} ({', '.join(class_list)})")
    print(f"• Feature Dimension:    {feat_dim}")
    print(f"• Cross-Validation:     {args.folds}-Fold Stratified Split")
    print(f"• Epochs per Fold:      {args.epochs} (Total: {args.folds * args.epochs} Epochs)")
    print(f"• Batch Size:           {args.batch_size}")
    print(f"• Device:               {args.device.upper()}")
    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"• GPU:                  {torch.cuda.get_device_name(0)}")

    if not args.skip_servers:
        print("\n🌐 REMOTE MONITORING (TAILSCALE VPN / LOCAL NETWORK):")
        start_background_servers(args.log_dir, tb_port=args.tb_port, monitor_port=args.monitor_port)
        print(f"  📱 AMOLED Mobile Monitor: http://{ip}:{args.monitor_port}")
        print(f"  📊 TensorBoard Metrics:   http://{ip}:{args.tb_port}")
    print("=" * 70 + "\n")

    # Dynamic AnomalyHead initialization
    model = AnomalyHead(in_features=feat_dim, num_classes=num_classes, hidden_dims=[512,344], dropout_rates=[0.5,0.4])

    trainer = ClassAwareMILTrainer(
        model=model,
        features_dir=args.features_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        smoothness_weight=args.smoothness_weight,
        sparsity_weight=args.sparsity_weight,
        k_top=args.k_top,
        k_fold=args.folds,
        early_stop_patience=8,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        state_file=args.state_file,
        class_list=class_list,
    )

    results = trainer.train()

    # Save human-readable markdown report
    os.makedirs("results", exist_ok=True)
    report_md = os.path.join("results", f"{args.folds}fold_training_report.md")
    model_name = taxonomy.get("model", "MViT_v2_S")
    fps_val = taxonomy.get("fps", 20)
    overlap_val = taxonomy.get("overlap_ratio", 0.5)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.folds}-Fold Sınıf-Farkında (Class-Aware) Eğitim Raporu\n\n")
        f.write(f"- **Tarih:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Backbone Modeli:** {model_name} (Öznitelik Boyutu: {feat_dim})\n")
        f.write(f"- **Örnekleme:** {fps_val} FPS, Örtüşme (Overlap): {overlap_val}\n")
        f.write(f"- **Makro Sınıflar:** {num_classes} ({', '.join(class_list)})\n")
        f.write(f"- **Veri Seti:** `{args.features_dir}` (Toplam: {taxonomy.get('total_files', 0)} video)\n")
        f.write(f"- **Fold Sayısı:** {args.folds}\n")
        f.write(f"- **Epoch / Fold:** {args.epochs}\n")
        f.write(f"- **Toplam Süre:** {results['total_duration_sec']} saniye\n\n")
        f.write("## 1. Genel Performans Özeti\n\n")
        f.write("| Metrik | Ortalama (Mean) | Standart Sapma (Std) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Doğrulama mAP** | **{results['mean_best_map']:.4f}** | ± {results['std_best_map']:.4f} |\n")
        f.write(f"| **Doğrulama ROC-AUC** | **{results['mean_best_auc']:.4f}** | ± {results['std_best_auc']:.4f} |\n")
        f.write(f"| **Doğrulama Kaybı (Loss)** | **{results['mean_best_loss']:.4f}** | ± {results['std_best_loss']:.4f} |\n\n")
        f.write("## 2. Fold Bazlı Detaylar\n\n")
        f.write("| Fold | En İyi mAP | En İyi ROC-AUC | En İyi Val Loss | En İyi Epoch |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for fr in results["fold_results"]:
            f.write(f"| Fold {fr['fold']} | {fr['best_val_map']:.4f} | {fr['best_val_auc']:.4f} | {fr['best_val_loss']:.4f} | Epoch {fr['best_epoch']} |\n")

    print(f"📄 Markdown raporu kaydedildi: {report_md}")


if __name__ == "__main__":
    main()
