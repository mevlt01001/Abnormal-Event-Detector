#!/usr/bin/env python3
"""High-Performance Real-Time Monitoring Server for 15-Hour XD-Violence Pipeline.

- True AMOLED Pure Black (#000000) for zero-power OLED pixel consumption
- Sub-second real-time streaming (100ms polling)
- Safe GPU telemetry parsing handling [N/A] power limits
- System RAM, CPU Load, and comprehensive GPU telemetry
- Training state detection (Active Model, Stage, Fold, Epoch, Loss, mAP)
- 15-Hour 6-Model Roadmap & Dynamic Leaderboard
- Mobile horizontal responsive split-screen layout
- Clean CLI endpoint for Termux curl
"""

import http.server
import json
import os
import re
import socketserver
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

PORT = 8080
LOG_FILE = "/home/n3uron/workspace/Anomaly_Detection/pipeline_run.log"
FEATURES_DIR = "/home/n3uron/workspace/Anomaly_Detection/data/features"
CHECKPOINTS_DIR = "/home/n3uron/workspace/Anomaly_Detection/checkpoints"
RESULTS_DIR = "/home/n3uron/workspace/Anomaly_Detection/results"

MODELS_ORDER: List[str] = [
    "s3d",
    "swin3d_t",
    "mvit_v1_b",
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
]

# Global CPU measurement tracker
prev_cpu_idle = 0.0
prev_cpu_total = 0.0

_gpu_cache: Dict[str, Any] = {}
_gpu_cache_time: float = 0.0


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def get_cpu_utilization() -> float:
    """Computes instant system CPU utilization % from /proc/stat."""
    global prev_cpu_idle, prev_cpu_total
    try:
        with open("/proc/stat", "r") as f:
            first_line = f.readline()
        fields = [float(x) for x in first_line.strip().split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
        total = sum(fields)

        diff_idle = idle - prev_cpu_idle
        diff_total = total - prev_cpu_total
        prev_cpu_idle = idle
        prev_cpu_total = total

        if diff_total > 0:
            util = (1.0 - (diff_idle / diff_total)) * 100.0
            return max(0.0, min(100.0, round(util, 1)))
    except Exception:
        pass
    return 0.0


def get_system_ram() -> Dict[str, Any]:
    """Reads system RAM statistics from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                mem[parts[0].strip()] = int(parts[1].split()[0])

        total_mb = mem.get("MemTotal", 0) // 1024
        avail_mb = mem.get("MemAvailable", 0) // 1024
        used_mb = max(0, total_mb - avail_mb)
        pct = (used_mb / max(1, total_mb)) * 100.0
        return {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "avail_mb": avail_mb,
            "pct": round(pct, 1),
            "text": f"{used_mb} / {total_mb} MB ({pct:.1f}%)",
        }
    except Exception as e:
        return {"total_mb": 0, "used_mb": 0, "avail_mb": 0, "pct": 0, "text": f"N/A: {e}"}


def get_gpu_telemetry() -> Dict[str, Any]:
    """Extracts GPU hardware telemetry using nvidia-smi with safe parsing and sub-second caching."""
    global _gpu_cache, _gpu_cache_time
    now = time.time()
    if _gpu_cache and (now - _gpu_cache_time) < 0.15:
        return _gpu_cache

    try:
        raw = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        raw = f"nvidia-smi error: {e}"

    info = {
        "raw": raw,
        "util": 0,
        "util_str": "0%",
        "mem_used": 0,
        "mem_total": 4096,
        "mem_pct": 0.0,
        "temp": 0,
        "power_draw": 0.0,
        "power_limit": 50.0,
    }

    try:
        csv_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        parts = [p.strip() for p in csv_out.split(",")]
        if len(parts) >= 6:
            u = safe_int(parts[0], 0)
            mu = safe_int(parts[1], 0)
            mt = safe_int(parts[2], 4096)
            t = safe_int(parts[3], 0)
            pd = safe_float(parts[4], 0.0)
            pl = safe_float(parts[5], 50.0)
            info["util"] = u
            info["util_str"] = f"{u}%"
            info["mem_used"] = mu
            info["mem_total"] = mt
            info["mem_pct"] = round((mu / max(1, mt)) * 100.0, 1)
            info["temp"] = t
            info["power_draw"] = pd
            info["power_limit"] = pl
    except Exception:
        pass

    _gpu_cache = info
    _gpu_cache_time = now
    return info


def get_active_stage_and_training_info() -> Dict[str, Any]:
    """Parses pipeline_run.log to extract real-time training status and current active stage."""
    state = {
        "active_model": "s3d",
        "active_stage": "INITIALIZING",
        "stage_description": "Pipeline initializing...",
        "is_training": False,
        "train_type": "",
        "fold": "",
        "epoch": "",
        "train_loss": "",
        "val_loss": "",
        "val_map": "",
        "recent_metric": "",
    }

    if not os.path.isfile(LOG_FILE):
        return state

    try:
        try:
            stage_lines = subprocess.check_output(["grep", "-E", "STAGE [0-9]:", LOG_FILE], text=True).splitlines()
            if stage_lines:

                last_stage = stage_lines[-1]
                m_stage = re.search(r"STAGE (\d): (.+?) FOR \[([a-zA-Z0-9_]+)\]", last_stage)
                if m_stage:
                    s_num, s_desc, m_name = m_stage.groups()
                    state["active_model"] = m_name.lower()
                    state["active_stage"] = f"STAGE {s_num}: {s_desc}"
                    state["stage_description"] = f"Model {m_name.upper()} | {s_desc}"
                    state["is_training"] = "TRAINING" in s_desc
                    state["train_type"] = "Binary MIL" if "BINARY" in s_desc else ("Multi-Class MIL" if "MULTI-CLASS" in s_desc else "")
        except Exception:
            pass

        lines = subprocess.check_output(["tail", "-n", "100", LOG_FILE], text=True).splitlines()
        for line in lines:


            # Binary Training Epoch line:
            m_bin = re.search(r"Fold (\d+) \| Epoch (\d+/\d+) - Train Loss: ([\d\.]+) \(Hinge: ([\d\.]+)\) \| Val Loss: ([\d\.]+)", line)
            if m_bin:
                f_num, ep, t_loss, h_loss, v_loss = m_bin.groups()
                state["is_training"] = True
                state["train_type"] = "Binary MIL (Sultani)"
                state["fold"] = f"Fold {f_num}/5"
                state["epoch"] = ep
                state["train_loss"] = t_loss
                state["val_loss"] = v_loss
                state["recent_metric"] = f"Train: {t_loss} | Val: {v_loss}"

            # Multi-Class Training Epoch line:
            m_mul = re.search(r"Fold (\d+) Epoch (\d+/\d+) \| Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+) \| Val mAP: ([\d\.]+)", line)
            if m_mul:
                f_num, ep, t_loss, v_loss, v_map = m_mul.groups()
                state["is_training"] = True
                state["train_type"] = "Multi-Class MIL (Top-K BCE)"
                state["fold"] = f"Fold {f_num}/5"
                state["epoch"] = ep
                state["train_loss"] = t_loss
                state["val_loss"] = v_loss
                state["val_map"] = v_map
                state["recent_metric"] = f"Val Loss: {v_loss} | mAP: {v_map}"

            if "Parallel Extraction:" in line:
                state["stage_description"] = line.strip().replace("\r", "")
    except Exception:
        pass

    return state


def get_models_roadmap() -> List[Dict[str, Any]]:
    """Inspects filesystem to build complete status of all 6 models."""
    roadmap = []

    for model in MODELS_ORDER:
        m_data = {
            "name": model,
            "extraction_count": 0,
            "extraction_pct": 0.0,
            "extraction_done": False,
            "binary_done": False,
            "binary_loss": "N/A",
            "multiclass_done": False,
            "multiclass_map": "N/A",
            "eval_done": False,
            "frame_auc": "N/A",
            "tiou_05": "N/A",
        }

        # 1. Extraction status
        m_feat_dir = os.path.join(FEATURES_DIR, model)
        if os.path.isdir(m_feat_dir):
            norm_dir = os.path.join(m_feat_dir, "normal")
            anom_dir = os.path.join(m_feat_dir, "anomal")
            nc = len([f for f in os.listdir(norm_dir) if f.endswith(".pt")]) if os.path.isdir(norm_dir) else 0
            ac = len([f for f in os.listdir(anom_dir) if f.endswith(".pt")]) if os.path.isdir(anom_dir) else 0
            tot = nc + ac
            m_data["extraction_count"] = tot
            m_data["extraction_pct"] = round((tot / 800.0) * 100.0, 1)
            if tot >= 800:
                m_data["extraction_done"] = True

        # 2. Binary Training status
        bin_ckpt_dir = os.path.join(CHECKPOINTS_DIR, "binary", model)
        if os.path.isdir(bin_ckpt_dir):
            ckpts = [f for f in os.listdir(bin_ckpt_dir) if f.startswith("best_loss")]
            if ckpts:
                m_data["binary_done"] = len(ckpts) >= 5
                m_data["binary_loss"] = f"{len(ckpts)}/5 Folds"

        # 3. Multi-Class Training status
        mul_ckpt_dir = os.path.join(CHECKPOINTS_DIR, "multiclass", model)
        if os.path.isdir(mul_ckpt_dir):
            ckpts = [f for f in os.listdir(mul_ckpt_dir) if f.startswith("best_loss")]
            if ckpts:
                m_data["multiclass_done"] = len(ckpts) >= 5
                m_data["multiclass_map"] = f"{len(ckpts)}/5 Folds"

        # 4. Evaluation results
        eval_json = os.path.join(RESULTS_DIR, f"{model}_eval.json")
        if os.path.isfile(eval_json):
            try:
                with open(eval_json, "r") as f:
                    ev = json.load(f)
                b_ev = ev.get("evaluation", {}).get("binary", {})
                m_ev = ev.get("evaluation", {}).get("multiclass", {})
                m_data["eval_done"] = True
                m_data["frame_auc"] = f"{b_ev.get('frame_auc', 0.0):.4f}"
                m_data["tiou_05"] = f"{b_ev.get('tiou_recalls', {}).get('tIoU_0.5', 0.0):.4f}"
                m_data["multiclass_map"] = f"{m_ev.get('mAP', 0.0):.4f}"
            except Exception:
                pass

        roadmap.append(m_data)

    return roadmap


def get_log_tail(n: int = 40) -> str:
    """Reads last n lines from pipeline log, cleaning terminal codes."""
    if not os.path.isfile(LOG_FILE):
        return "Log file not created yet..."
    try:
        out = subprocess.check_output(["tail", "-n", str(n), LOG_FILE], text=True)
        clean_lines = []
        for line in out.splitlines():
            if "\r" in line:
                line = line.split("\r")[-1]
            clean_lines.append(line)
        return "\n".join(clean_lines)
    except Exception as e:
        return f"Error reading log: {e}"


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>🛰️ Mission Control | AMOLED Pure Black</title>
  <style>
    :root {
      --bg: #000000;
      --card-bg: #000000;
      --card-border: #141824;
      --card-border-active: #00b4d8;
      --text: #718096;
      --text-bright: #ffffff;
      --neon-green: #00ff88;
      --neon-blue: #00b4d8;
      --neon-amber: #ffb703;
      --neon-purple: #c77dff;
      --neon-red: #ff3366;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "JetBrains Mono", monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      background: #000000 !important;
      color: var(--text);
      font-family: var(--font);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      padding: 6px;
    }

    /* TOP HEADER */
    .header {
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 5px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
      flex-shrink: 0;
    }
    .header-left { display: flex; align-items: center; gap: 8px; }
    .header-title { font-size: 0.9rem; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; }
    .pulse-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: #000000;
      border: 1px solid #00ff88;
      color: #00ff88;
      padding: 2px 7px;
      border-radius: 10px;
      font-size: 0.68rem;
      font-weight: 700;
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #00ff88;
      box-shadow: 0 0 6px #00ff88;
      animation: blink 1s infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

    .stage-pill {
      background: #000000;
      border: 1px solid var(--neon-blue);
      color: var(--neon-blue);
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 700;
      max-width: 480px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* MAIN HORIZONTAL 3-COLUMN LAYOUT */
    .main-container {
      display: grid;
      grid-template-columns: 230px 380px 1fr;
      gap: 6px;
      flex: 1;
      min-height: 0;
      background: #000000;
    }

    /* COLUMN 1: HARDWARE */
    .col-hardware {
      display: flex;
      flex-direction: column;
      gap: 6px;
      overflow-y: auto;
      background: #000000;
    }
    .hw-card {
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 10px;
    }
    .hw-title {
      font-size: 0.68rem;
      color: #718096;
      text-transform: uppercase;
      font-weight: 800;
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
    }
    .hw-val-large {
      font-size: 1.35rem;
      font-weight: 900;
      color: #ffffff;
    }
    .hw-subval {
      font-size: 0.7rem;
      color: #a0aec0;
      margin-top: 2px;
    }
    .progress-bar-bg {
      background: #10141e;
      border-radius: 3px;
      height: 5px;
      overflow: hidden;
      margin-top: 5px;
    }
    .progress-bar-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.2s ease;
    }
    .fill-green { background: #00ff88; box-shadow: 0 0 5px #00ff88; }
    .fill-blue { background: #00b4d8; box-shadow: 0 0 5px #00b4d8; }
    .fill-amber { background: #ffb703; box-shadow: 0 0 5px #ffb703; }
    .fill-purple { background: #c77dff; box-shadow: 0 0 5px #c77dff; }

    /* COLUMN 2: ROADMAP */
    .col-roadmap {
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      overflow-y: auto;
    }
    .training-status-box {
      background: #000000;
      border: 1px solid var(--neon-blue);
      border-radius: 6px;
      padding: 6px 8px;
    }
    .ts-title { font-size: 0.68rem; color: var(--neon-blue); font-weight: 800; text-transform: uppercase; margin-bottom: 4px; }
    .ts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.72rem; }
    .ts-item span:first-child { color: #718096; }
    .ts-item span:last-child { color: #ffffff; font-weight: 700; }

    .models-list { display: flex; flex-direction: column; gap: 5px; }
    .model-card {
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 5px;
      padding: 5px 8px;
      transition: border 0.15s ease;
    }
    .model-card.active {
      border: 1px solid var(--neon-blue);
      box-shadow: 0 0 8px rgba(0, 180, 216, 0.3);
    }
    .mc-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      font-weight: 800;
      color: #ffffff;
    }
    .badge-chip {
      font-size: 0.62rem;
      padding: 1px 5px;
      border-radius: 3px;
      font-weight: 700;
    }
    .chip-done { background: #000000; border: 1px solid #00ff88; color: #00ff88; }
    .chip-active { background: #000000; border: 1px solid var(--neon-blue); color: var(--neon-blue); animation: blink 1.5s infinite; }
    .chip-wait { background: #000000; border: 1px solid #2d3748; color: #4a5568; }

    .mc-metrics {
      display: flex;
      justify-content: space-between;
      font-size: 0.65rem;
      color: #a0aec0;
      margin-top: 3px;
    }

    /* COLUMN 3: TERMINAL CONSOLE */
    .col-console {
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .console-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 4px;
      flex-shrink: 0;
    }
    .console-title { font-size: 0.72rem; color: #718096; font-weight: 800; text-transform: uppercase; }
    .console-controls { display: flex; gap: 6px; }
    .btn-toggle {
      background: #000000;
      border: 1px solid #2d3748;
      color: #a0aec0;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 0.65rem;
      cursor: pointer;
      font-weight: 700;
    }
    .btn-toggle.active { border-color: var(--neon-blue); color: var(--neon-blue); }

    .console-pre {
      font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
      font-size: 0.7rem;
      color: #00ff88;
      line-height: 1.3;
      background: #000000 !important;
      white-space: pre-wrap;
      word-break: break-all;
      flex: 1;
      overflow-y: auto;
      padding-right: 4px;
    }

    @media (max-width: 800px) and (orientation: portrait) {
      html, body { height: auto; overflow: auto; }
      .main-container { grid-template-columns: 1fr; }
      .console-pre { max-height: 350px; }
    }
  </style>
</head>
<body>

  <!-- TOP HEADER -->
  <div class="header">
    <div class="header-left">
      <div class="header-title">🛰️ MISSION CONTROL</div>
      <div class="pulse-badge">
        <span class="pulse-dot"></span>
        <span id="rate-indicator">100ms CANLI</span>
      </div>
      <div class="stage-pill" id="stage-badge">BAŞLATILIYOR...</div>
    </div>
    <div style="font-size: 0.7rem; color: #718096;" id="clock-display">00:00:00</div>
  </div>

  <!-- 3-COLUMN MAIN GRID -->
  <div class="main-container">

    <!-- COL 1: HARDWARE & SYSTEM TELEMETRY -->
    <div class="col-hardware">
      <!-- GPU UTIL & TEMP -->
      <div class="hw-card">
        <div class="hw-title">
          <span>GPU YÜKÜ (GTX 1650 Ti)</span>
          <span id="gpu-temp-badge" style="color: #00ff88;">--°C</span>
        </div>
        <div class="hw-val-large" id="gpu-util">--%</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill fill-green" id="gpu-util-bar" style="width: 0%;"></div>
        </div>
        <div class="hw-subval" id="gpu-power-sub">Güç: --W / 50W</div>
      </div>

      <!-- VRAM -->
      <div class="hw-card">
        <div class="hw-title">
          <span>VRAM KULLANIMI</span>
          <span id="vram-pct" style="color: var(--neon-blue);">--%</span>
        </div>
        <div class="hw-val-large" id="vram-val">-- MB</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill fill-blue" id="vram-bar" style="width: 0%;"></div>
        </div>
        <div class="hw-subval" id="vram-total-sub">Toplam: 4096 MB</div>
      </div>

      <!-- SYSTEM RAM -->
      <div class="hw-card">
        <div class="hw-title">
          <span>SİSTEM RAM'İ</span>
          <span id="sys-ram-pct" style="color: var(--neon-purple);">--%</span>
        </div>
        <div class="hw-val-large" id="sys-ram-val">-- MB</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill fill-purple" id="sys-ram-bar" style="width: 0%;"></div>
        </div>
        <div class="hw-subval" id="sys-ram-sub">Boş: -- MB</div>
      </div>

      <!-- CPU LOAD -->
      <div class="hw-card">
        <div class="hw-title">
          <span>CPU KULLANIMI</span>
          <span id="cpu-load-val" style="color: var(--neon-amber);">--%</span>
        </div>
        <div class="progress-bar-bg" style="margin-top: 2px;">
          <div class="progress-bar-fill fill-amber" id="cpu-bar" style="width: 0%;"></div>
        </div>
      </div>
    </div>

    <!-- COL 2: 15-HOUR ROADMAP & ACTIVE TRAINING -->
    <div class="col-roadmap">
      <div class="training-status-box" id="training-box">
        <div class="ts-title" id="ts-type">AKTİF DURUM</div>
        <div class="ts-grid">
          <div class="ts-item"><span>Model:</span> <span id="ts-model" style="color:var(--neon-blue);">--</span></div>
          <div class="ts-item"><span>Aşama:</span> <span id="ts-stage">--</span></div>
          <div class="ts-item"><span>Fold:</span> <span id="ts-fold">--</span></div>
          <div class="ts-item"><span>Epoch:</span> <span id="ts-epoch">--</span></div>
          <div class="ts-item"><span>Train Loss:</span> <span id="ts-tloss">--</span></div>
          <div class="ts-item"><span>Val Loss:</span> <span id="ts-vloss">--</span></div>
          <div class="ts-item" style="grid-column: span 2;"><span>Val mAP / Metrik:</span> <span id="ts-map" style="color:#00ff88;">--</span></div>
        </div>
      </div>

      <div style="font-size: 0.68rem; color: #718096; font-weight: 800; text-transform: uppercase;">
        15-Saatlik 6-Model Yol Haritası
      </div>
      <div class="models-list" id="models-list-container"></div>
    </div>

    <!-- COL 3: REAL-TIME CONSOLE LOG -->
    <div class="col-console">
      <div class="console-header">
        <div class="console-title">CANLI LOG (100ms AKIŞ)</div>
        <div class="console-controls">
          <button class="btn-toggle active" id="btn-autoscroll" onclick="toggleAutoScroll()">Auto-Scroll: ON</button>
        </div>
      </div>
      <pre class="console-pre" id="console-output">Loglar akıyor...</pre>
    </div>

  </div>

  <script>
    let autoScroll = true;

    function toggleAutoScroll() {
      autoScroll = !autoScroll;
      const btn = document.getElementById('btn-autoscroll');
      btn.textContent = 'Auto-Scroll: ' + (autoScroll ? 'ON' : 'OFF');
      btn.classList.toggle('active', autoScroll);
    }

    function updateClock() {
      document.getElementById('clock-display').textContent = new Date().toLocaleTimeString();
    }
    setInterval(updateClock, 100);
    updateClock();

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // GPU
        const gpu = data.gpu;
        document.getElementById('gpu-util').textContent = gpu.util_str;
        document.getElementById('gpu-util-bar').style.width = gpu.util + '%';
        document.getElementById('gpu-temp-badge').textContent = gpu.temp + '°C';
        if (gpu.temp > 80) document.getElementById('gpu-temp-badge').style.color = 'var(--neon-red)';
        else if (gpu.temp > 72) document.getElementById('gpu-temp-badge').style.color = 'var(--neon-amber)';
        else document.getElementById('gpu-temp-badge').style.color = 'var(--neon-green)';
        document.getElementById('gpu-power-sub').textContent = 'Güç: ' + gpu.power_draw.toFixed(1) + 'W / ' + gpu.power_limit.toFixed(0) + 'W';

        // VRAM
        document.getElementById('vram-val').textContent = gpu.mem_used + ' MB';
        document.getElementById('vram-pct').textContent = gpu.mem_pct + '%';
        document.getElementById('vram-bar').style.width = gpu.mem_pct + '%';
        document.getElementById('vram-total-sub').textContent = 'Toplam: ' + gpu.mem_total + ' MB';

        // System RAM
        const sram = data.system_ram;
        document.getElementById('sys-ram-val').textContent = sram.used_mb + ' MB';
        document.getElementById('sys-ram-pct').textContent = sram.pct + '%';
        document.getElementById('sys-ram-bar').style.width = sram.pct + '%';
        document.getElementById('sys-ram-sub').textContent = 'Kullanılabilir: ' + sram.avail_mb + ' MB / ' + sram.total_mb + ' MB';

        // CPU
        const cpu = data.cpu_util;
        document.getElementById('cpu-load-val').textContent = cpu + '%';
        document.getElementById('cpu-bar').style.width = cpu + '%';

        // Active State & Training
        const st = data.stage_info;
        document.getElementById('stage-badge').textContent = st.stage_description || st.active_stage;
        document.getElementById('ts-type').textContent = st.is_training ? ('🔥 ' + st.train_type.toUpperCase()) : 'AKTİF İŞLEM';
        document.getElementById('ts-model').textContent = st.active_model.toUpperCase();
        document.getElementById('ts-stage').textContent = st.active_stage;
        document.getElementById('ts-fold').textContent = st.fold || '-';
        document.getElementById('ts-epoch').textContent = st.epoch || '-';
        document.getElementById('ts-tloss').textContent = st.train_loss || '-';
        document.getElementById('ts-vloss').textContent = st.val_loss || '-';
        document.getElementById('ts-map').textContent = st.val_map || st.recent_metric || '-';

        // Roadmap Models List
        const mCont = document.getElementById('models-list-container');
        let mHtml = '';
        for (const m of data.roadmap) {
          const isActive = m.name.toLowerCase() === st.active_model.toLowerCase();
          const extBadge = m.extraction_done 
            ? '<span class="badge-chip chip-done">800/800 ✓</span>'
            : (m.extraction_count > 0 ? `<span class="badge-chip chip-active">${m.extraction_count}/800 (${m.extraction_pct}%)</span>` : '<span class="badge-chip chip-wait">0/800</span>');

          const binBadge = m.binary_done 
            ? '<span class="badge-chip chip-done">Binary: 5/5 ✓</span>' 
            : (m.binary_loss !== 'N/A' ? `<span class="badge-chip chip-active">${m.binary_loss}</span>` : '<span class="badge-chip chip-wait">Binary: Bekliyor</span>');

          const mulBadge = m.multiclass_done 
            ? '<span class="badge-chip chip-done">Multi: 5/5 ✓</span>' 
            : (m.multiclass_map !== 'N/A' ? `<span class="badge-chip chip-active">${m.multiclass_map}</span>` : '<span class="badge-chip chip-wait">Multi: Bekliyor</span>');

          const evalBadge = m.eval_done 
            ? `<span class="badge-chip chip-done">AUC: ${m.frame_auc}</span>` 
            : '<span class="badge-chip chip-wait">Eval: -</span>';

          mHtml += `
            <div class="model-card ${isActive ? 'active' : ''}">
              <div class="mc-header">
                <span>${m.name.toUpperCase()}</span>
                ${extBadge}
              </div>
              <div class="progress-bar-bg" style="height: 3px; margin-top: 2px;">
                <div class="progress-bar-fill ${m.extraction_done ? 'fill-green' : 'fill-blue'}" style="width: ${m.extraction_pct}%;"></div>
              </div>
              <div class="mc-metrics">
                ${binBadge}
                ${mulBadge}
                ${evalBadge}
              </div>
            </div>`;
        }
        mCont.innerHTML = mHtml;

        // Console Log
        const cPre = document.getElementById('console-output');
        cPre.textContent = data.log;
        if (autoScroll) {
          cPre.scrollTop = cPre.scrollHeight;
        }

      } catch (e) {
        document.getElementById('stage-badge').textContent = 'YENİDEN BAĞLANILIYOR...';
      }
    }

    // 100 ms real-time polling
    fetchStatus();
    setInterval(fetchStatus, 100);
  </script>
</body>
</html>
"""


class MonitorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif self.path == "/api/status":
            data = {
                "gpu": get_gpu_telemetry(),
                "system_ram": get_system_ram(),
                "cpu_util": get_cpu_utilization(),
                "stage_info": get_active_stage_and_training_info(),
                "roadmap": get_models_roadmap(),
                "log": get_log_tail(45),
                "timestamp": time.time(),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif self.path == "/cli":
            gpu = get_gpu_telemetry()
            sram = get_system_ram()
            cpu = get_cpu_utilization()
            st = get_active_stage_and_training_info()
            roadmap = get_models_roadmap()
            log = get_log_tail(18)

            rm_lines = []
            for m in roadmap:
                rm_lines.append(
                    f"{m['name']:<12} | Ext: {m['extraction_count']:>3}/800 ({m['extraction_pct']:>5}%) | "
                    f"Bin: {m['binary_loss']} | Multi: {m['multiclass_map']} | AUC: {m['frame_auc']}"
                )
            roadmap_str = "\n".join(rm_lines)

            output = (
                f"{gpu['raw']}\n"
                f"{'='*65}\n"
                f"SYSTEM: CPU {cpu}% | RAM {sram['text']} | STAGE: {st['stage_description']}\n"
                f"{'='*65}\n"
                f"15-HOUR ROADMAP & METRICS:\n{roadmap_str}\n"
                f"{'='*65}\n"
                f"LATEST LOGS:\n{log}\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(output.encode("utf-8"))

        elif self.path == "/gpu":
            gpu = get_gpu_telemetry()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(gpu["raw"].encode("utf-8"))

        elif self.path == "/log":
            log = get_log_tail(50)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(log.encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), MonitorHandler) as httpd:
        print(f"AMOLED Pure Black Monitor running on http://0.0.0.0:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
