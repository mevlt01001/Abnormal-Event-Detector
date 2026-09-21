#!/usr/bin/env python3
"""High-Performance Real-Time Monitoring Server for 8-Fold Anomaly Detection Training.

- True AMOLED Pure Black (#000000) for zero-power OLED pixel consumption on smartphones
- Direct streaming from results/training_state.json with sub-second polling (250ms)
- 8-Fold Multi-Class Cross Validation live leaderboard & per-class AP breakdown
- Instant TensorBoard bridge button (:6006)
- Safe GPU telemetry parsing handling [N/A] power limits (NVIDIA GTX 1650 Ti)
- System RAM, CPU Load, and comprehensive hardware vitals
- Mobile responsive layout for smartphone screens
- Clean CLI endpoint for curl / Termux
"""

import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from data.taxonomy import load_dataset_taxonomy
    _tax = load_dataset_taxonomy()
    DEFAULT_CLASSES = _tax["anomaly_classes"]
except Exception:
    DEFAULT_CLASSES = ["Violence", "Shooting", "Explosion_Fire", "Accident", "Riot_Vandalism", "Theft_Robbery"]

PORT = 8080
STATE_FILE = "/home/n3uron/workspace/Anomaly_Detection/results/training_state.json"
LOG_FILE = "/home/n3uron/workspace/Anomaly_Detection/pipeline_run.log"
TAILSCALE_IP = "100.107.39.95"
TENSORBOARD_PORT = 6006

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
    if _gpu_cache and (now - _gpu_cache_time) < 0.2:
        return _gpu_cache

    try:
        raw = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        raw = f"nvidia-smi error: {e}"

    info = {
        "raw": raw,
        "name": "NVIDIA GTX 1650 Ti",
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
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        parts = [p.strip() for p in csv_out.split(",")]
        if len(parts) >= 7:
            info["name"] = parts[0]
            u = safe_int(parts[1], 0)
            mu = safe_int(parts[2], 0)
            mt = safe_int(parts[3], 4096)
            t = safe_int(parts[4], 0)
            pd = safe_float(parts[5], 0.0)
            pl = safe_float(parts[6], 50.0)
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


def get_training_state() -> Dict[str, Any]:
    """Reads live JSON training state or falls back to sensible defaults."""
    default_state = {
        "is_training": False,
        "status": "READY",
        "current_fold": 1,
        "total_folds": 8,
        "current_epoch": 0,
        "total_epochs": 20,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_map": 0.0,
        "val_auc": 0.0,
        "best_val_map": 0.0,
        "best_val_loss": 0.0,
        "best_epoch": 0,
        "per_class_ap": {c: 0.0 for c in DEFAULT_CLASSES},
        "completed_folds": [],
        "elapsed_sec": 0.0,
        "eta_sec": 0.0,
        "timestamp": time.time(),
    }

    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_state.update(data)
                return default_state
        except Exception:
            pass

    # Fallback to log file parsing if available
    if os.path.isfile(LOG_FILE):
        try:
            lines = subprocess.check_output(["tail", "-n", "30", LOG_FILE], text=True).splitlines()
            for line in lines:
                m = re.search(r"Fold (\d+)/(\d+) Epoch (\d+)/(\d+) \| Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+) \| Val mAP: ([\d\.]+) \| Val AUC: ([\d\.]+)", line)
                if m:
                    f_num, tot_f, ep, tot_ep, t_loss, v_loss, v_map, v_auc = m.groups()
                    default_state["is_training"] = True
                    default_state["status"] = "TRAINING"
                    default_state["current_fold"] = int(f_num)
                    default_state["total_folds"] = int(tot_f)
                    default_state["current_epoch"] = int(ep)
                    default_state["total_epochs"] = int(tot_ep)
                    default_state["train_loss"] = float(t_loss)
                    default_state["val_loss"] = float(v_loss)
                    default_state["val_map"] = float(v_map)
                    default_state["val_auc"] = float(v_auc)
        except Exception:
            pass

    return default_state


def get_log_tail(n: int = 30) -> str:
    """Reads last n lines from pipeline or training log."""
    train_log = "/home/n3uron/workspace/Anomaly_Detection/results/training.log"
    target_log = train_log if os.path.isfile(train_log) else LOG_FILE

    if not os.path.isfile(target_log):
        return "Eğitim başlatıldığında konsol logları burada canlı akacaktır..."
    try:
        out = subprocess.check_output(["tail", "-n", str(n), target_log], text=True)
        clean_lines = []
        for line in out.splitlines():
            if "\r" in line:
                line = line.split("\r")[-1]
            clean_lines.append(line)
        return "\n".join(clean_lines)
    except Exception as e:
        return f"Log okuma hatası: {e}"


DASHBOARD_HTML = f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>🛰️ 8-Fold Anomaly Mission Control</title>
  <style>
    :root {{
      --bg: #000000;
      --card-bg: #05070c;
      --card-border: #141b2d;
      --neon-green: #00ff88;
      --neon-blue: #00b4d8;
      --neon-cyan: #00f0ff;
      --neon-amber: #ffb703;
      --neon-purple: #c77dff;
      --neon-red: #ff3366;
      --text: #8892b0;
      --text-bright: #ffffff;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "JetBrains Mono", monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      background: #000000 !important;
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      padding: 8px;
    }}

    /* HEADER */
    .top-header {{
      background: #000000;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .brand-title {{
      font-size: 1rem;
      font-weight: 900;
      color: #ffffff;
      letter-spacing: 0.5px;
    }}
    .vpn-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: rgba(0, 255, 136, 0.1);
      border: 1px solid var(--neon-green);
      color: var(--neon-green);
      padding: 3px 8px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 700;
    }}
    .live-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--neon-green);
      box-shadow: 0 0 8px var(--neon-green);
      animation: pulse 1s infinite alternate;
    }}
    @keyframes pulse {{ from {{ opacity: 0.3; }} to {{ opacity: 1; }} }}

    .header-actions {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .tb-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #ff5722;
      color: #ffffff;
      border: none;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
      box-shadow: 0 0 10px rgba(255, 87, 34, 0.3);
      transition: transform 0.15s ease;
    }}
    .tb-btn:hover {{ transform: scale(1.03); }}

    /* GRID SYSTEM */
    .dashboard-grid {{
      display: grid;
      grid-template-columns: 280px 1fr 340px;
      gap: 8px;
      align-items: start;
    }}
    @media (max-width: 1024px) {{
      .dashboard-grid {{ grid-template-columns: 1fr; }}
    }}

    /* CARD STYLING */
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 8px;
    }}
    .card-title {{
      font-size: 0.72rem;
      font-weight: 800;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    /* METRIC HERO */
    .hero-stat {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 4px;
    }}
    .stat-large {{
      font-size: 1.8rem;
      font-weight: 900;
      color: #ffffff;
    }}
    .stat-sub {{
      font-size: 0.75rem;
      color: var(--neon-blue);
      font-weight: 700;
    }}

    /* PROGRESS BARS */
    .pbar-track {{
      width: 100%;
      height: 6px;
      background: #0f172a;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }}
    .pbar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease;
    }}
    .fill-green {{ background: linear-gradient(90deg, #00b4d8, #00ff88); }}
    .fill-cyan {{ background: linear-gradient(90deg, #0077b6, #00f0ff); }}
    .fill-amber {{ background: linear-gradient(90deg, #ff9e00, #ffb703); }}
    .fill-purple {{ background: linear-gradient(90deg, #7209b7, #c77dff); }}

    /* 8-FOLD LEADERBOARD TABLE */
    .fold-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.75rem;
      margin-top: 6px;
    }}
    .fold-table th {{
      text-align: left;
      padding: 5px 6px;
      color: #64748b;
      border-bottom: 1px solid var(--card-border);
      font-weight: 700;
    }}
    .fold-table td {{
      padding: 6px;
      border-bottom: 1px solid rgba(20, 27, 45, 0.5);
      color: #e2e8f0;
      font-family: monospace;
    }}
    .row-active {{
      background: rgba(0, 180, 216, 0.1) !important;
      font-weight: 800;
      border-left: 3px solid var(--neon-blue);
    }}
    .status-tag {{
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.65rem;
      font-weight: 800;
    }}
    .tag-done {{ background: rgba(0, 255, 136, 0.15); color: var(--neon-green); border: 1px solid var(--neon-green); }}
    .tag-run {{ background: rgba(0, 180, 216, 0.2); color: var(--neon-cyan); border: 1px solid var(--neon-cyan); }}
    .tag-wait {{ background: #0f172a; color: #475569; }}

    /* CLASS AP CARDS */
    .class-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .class-item {{
      background: #090d16;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 6px 8px;
    }}
    .class-name {{
      font-size: 0.7rem;
      font-weight: 800;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
    }}
    .class-val {{
      font-size: 0.85rem;
      font-weight: 900;
      color: #ffffff;
      margin-top: 2px;
    }}

    /* CONSOLE LOG BOX */
    .console-pre {{
      background: #000000;
      border: 1px solid #141b2d;
      border-radius: 6px;
      padding: 8px;
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 0.68rem;
      color: #38bdf8;
      height: 180px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }}
  </style>
</head>
<body>

  <!-- HEADER -->
  <div class="top-header">
    <div class="header-left">
      <span class="brand-title">ANOMALY DETECTOR 8-FOLD</span>
      <span class="vpn-badge">
        <span class="live-dot"></span>
        <span id="vpn-label">TAILSCALE VPN: {TAILSCALE_IP}</span>
      </span>
    </div>
    <div class="header-actions">
      <a id="tb-link" class="tb-btn" href="http://{TAILSCALE_IP}:{TENSORBOARD_PORT}" target="_blank">
        📊 TENSORBOARD (:6006)
      </a>
    </div>
  </div>

  <!-- MAIN 3-COL DASHBOARD -->
  <div class="dashboard-grid">

    <!-- COLUMN 1: HARDWARE VITALS -->
    <div class="col-hardware">
      <!-- GPU CARD -->
      <div class="card">
        <div class="card-title">
          <span>GPU TELEMETRY</span>
          <span id="gpu-name" style="color: var(--neon-cyan);">GTX 1650 Ti</span>
        </div>
        <div class="hero-stat">
          <span class="stat-large" id="gpu-util">0%</span>
          <span class="stat-sub" id="gpu-temp">0°C</span>
        </div>
        <div class="pbar-track">
          <div id="gpu-util-bar" class="pbar-fill fill-green" style="width: 0%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.72rem; color: #94a3b8;">
          <span>VRAM: <b id="gpu-vram-text" style="color: #fff;">0/4096 MB</b></span>
          <span>Güç: <b id="gpu-power" style="color: #fff;">0W</b></span>
        </div>
        <div class="pbar-track">
          <div id="gpu-vram-bar" class="pbar-fill fill-cyan" style="width: 0%;"></div>
        </div>
      </div>

      <!-- CPU & RAM CARD -->
      <div class="card">
        <div class="card-title">
          <span>CPU & SYSTEM RAM</span>
          <span id="cpu-pct" style="color: var(--neon-amber);">0%</span>
        </div>
        <div class="pbar-track">
          <div id="cpu-bar" class="pbar-fill fill-amber" style="width: 0%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.72rem; color: #94a3b8;">
          <span>RAM: <b id="ram-text" style="color: #fff;">0 MB</b></span>
          <span id="ram-pct" style="color: var(--neon-purple);">0%</span>
        </div>
        <div class="pbar-track">
          <div id="ram-bar" class="pbar-fill fill-purple" style="width: 0%;"></div>
        </div>
      </div>

      <!-- LIVE ETA & TIME -->
      <div class="card">
        <div class="card-title">ZAMAN & TAHMİNİ BİTİŞ</div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
          <span>Geçen Süre:</span>
          <b id="time-elapsed" style="color: #fff; font-family: monospace;">00:00:00</b>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span>Kalan Tahmini (ETA):</span>
          <b id="time-eta" style="color: var(--neon-cyan); font-family: monospace;">Hesaplanıyor...</b>
        </div>
      </div>
    </div>

    <!-- COLUMN 2: ACTIVE TRAINING & CLASS METRICS -->
    <div class="col-center">
      <!-- ACTIVE FOLD & EPOCH HERO -->
      <div class="card">
        <div class="card-title">
          <span id="active-fold-title">AKTİF FOLD 1 / 8</span>
          <span id="training-status-tag" class="status-tag tag-run">HAZIRLANIYOR</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; text-align: center; margin: 8px 0;">
          <div style="background: #090d16; border-radius: 6px; padding: 6px;">
            <div style="font-size: 0.65rem; color: #64748b;">EPOCH</div>
            <div id="metric-epoch" style="font-size: 1.1rem; font-weight: 900; color: #fff;">0/20</div>
          </div>
          <div style="background: #090d16; border-radius: 6px; padding: 6px;">
            <div style="font-size: 0.65rem; color: #64748b;">TRAIN LOSS</div>
            <div id="metric-train-loss" style="font-size: 1.1rem; font-weight: 900; color: var(--neon-amber);">0.0000</div>
          </div>
          <div style="background: #090d16; border-radius: 6px; padding: 6px;">
            <div style="font-size: 0.65rem; color: #64748b;">VAL LOSS</div>
            <div id="metric-val-loss" style="font-size: 1.1rem; font-weight: 900; color: var(--neon-blue);">0.0000</div>
          </div>
          <div style="background: #090d16; border-radius: 6px; padding: 6px;">
            <div style="font-size: 0.65rem; color: #64748b;">VAL mAP</div>
            <div id="metric-val-map" style="font-size: 1.1rem; font-weight: 900; color: var(--neon-green);">0.0000</div>
          </div>
        </div>
        <div class="pbar-track" style="height: 8px;">
          <div id="epoch-progress-bar" class="pbar-fill fill-green" style="width: 0%;"></div>
        </div>
      </div>

      <!-- MACRO CLASSES AP BREAKDOWN -->
      <div class="card">
        <div class="card-title">
          <span id="classes-card-title">MAKRO ANOMALİ SINIFLARI AP (%) SKORLARI</span>
          <span style="color: var(--neon-cyan); font-size: 0.65rem;">Top-K SALIENT SEGMENTS</span>
        </div>
        <div class="class-grid" id="classes-container">
          <!-- Dynamic Class Cards Injected Here -->
        </div>
      </div>

      <!-- LIVE TERMINAL LOGS -->
      <div class="card">
        <div class="card-title">
          <span>CANLI TERMİNAL ÇIKTISI</span>
          <span style="font-size: 0.65rem; color: #64748b;">OTOMATİK KAYDIRMA AKTİF</span>
        </div>
        <pre class="console-pre" id="console-output"></pre>
      </div>
    </div>

    <!-- COLUMN 3: 8-FOLD LEADERBOARD -->
    <div class="col-leaderboard">
      <div class="card">
        <div class="card-title">
          <span>8-FOLD ÇAPRAZ DOĞRULAMA TABLOSU</span>
          <span id="folds-completed-count" style="color: var(--neon-green);">0/8 Tamamlandı</span>
        </div>
        <table class="fold-table">
          <thead>
            <tr>
              <th>Fold</th>
              <th>En İyi mAP</th>
              <th>En İyi Loss</th>
              <th>Durum</th>
            </tr>
          </thead>
          <tbody id="fold-table-body">
            <!-- 8 Folds Injected Here -->
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    // Adapt link if user opens via local IP or Tailscale IP
    const currentHost = window.location.hostname;
    document.getElementById('tb-link').href = `http://${{currentHost}}:{TENSORBOARD_PORT}`;
    document.getElementById('vpn-label').textContent = `HOST IP: ${{currentHost}}`;

    function fmtTime(sec) {{
      if (!sec || isNaN(sec) || sec <= 0) return "--:--:--";
      const s = Math.floor(sec % 60);
      const m = Math.floor((sec / 60) % 60);
      const h = Math.floor(sec / 3600);
      return `${{h.toString().padStart(2, '0')}}:${{m.toString().padStart(2, '0')}}:${{s.toString().padStart(2, '0')}}`;
    }}

    async function fetchVitals() {{
      try {{
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();

        // GPU
        const gpu = data.gpu;
        document.getElementById('gpu-name').textContent = gpu.name || 'NVIDIA GPU';
        document.getElementById('gpu-util').textContent = gpu.util_str || '0%';
        document.getElementById('gpu-temp').textContent = (gpu.temp || 0) + '°C';
        document.getElementById('gpu-power').textContent = (gpu.power_draw || 0) + 'W';
        document.getElementById('gpu-vram-text').textContent = `${{gpu.mem_used}}/${{gpu.mem_total}} MB`;
        document.getElementById('gpu-util-bar').style.width = (gpu.util || 0) + '%';
        document.getElementById('gpu-vram-bar').style.width = (gpu.mem_pct || 0) + '%';

        // CPU & RAM
        document.getElementById('cpu-pct').textContent = (data.cpu_util || 0) + '%';
        document.getElementById('cpu-bar').style.width = (data.cpu_util || 0) + '%';
        document.getElementById('ram-text').textContent = data.system_ram.text || '0 MB';
        document.getElementById('ram-pct').textContent = (data.system_ram.pct || 0) + '%';
        document.getElementById('ram-bar').style.width = (data.system_ram.pct || 0) + '%';

        // Training State
        const st = data.state;
        const totalFolds = st.total_folds || 8;
        const curFold = st.current_fold || 1;
        const curEpoch = st.current_epoch || 0;
        const totalEpochs = st.total_epochs || 20;

        document.getElementById('active-fold-title').textContent = `AKTİF FOLD ${{curFold}} / ${{totalFolds}}`;
        const stTag = document.getElementById('training-status-tag');
        if (st.status === 'COMPLETED') {{
          stTag.className = 'status-tag tag-done';
          stTag.textContent = 'TAMAMLANDI ✓';
        }} else if (st.is_training) {{
          stTag.className = 'status-tag tag-run';
          stTag.textContent = 'EĞİTİLİYOR ⚡';
        }} else {{
          stTag.className = 'status-tag tag-wait';
          stTag.textContent = 'HAZIR';
        }}

        document.getElementById('metric-epoch').textContent = `${{curEpoch}}/${{totalEpochs}}`;
        document.getElementById('metric-train-loss').textContent = (st.train_loss || 0).toFixed(4);
        document.getElementById('metric-val-loss').textContent = (st.val_loss || 0).toFixed(4);
        document.getElementById('metric-val-map').textContent = (st.val_map || 0).toFixed(4);

        const epochPct = totalEpochs > 0 ? (curEpoch / totalEpochs) * 100 : 0;
        document.getElementById('epoch-progress-bar').style.width = epochPct + '%';

        // Elapsed & ETA
        document.getElementById('time-elapsed').textContent = fmtTime(st.elapsed_sec);
        document.getElementById('time-eta').textContent = st.eta_sec > 0 ? fmtTime(st.eta_sec) : (st.status === 'COMPLETED' ? 'Tamamlandı' : 'Hesaplanıyor...');

        // Per-Class AP (Dynamically populated from live state)
        const classCont = document.getElementById('classes-container');
        const perClass = st.per_class_ap || {{}};
        const activeClasses = Object.keys(perClass).length > 0 
          ? Object.keys(perClass) 
          : """ + json.dumps(DEFAULT_CLASSES) + """;

        const titleEl = document.getElementById('classes-card-title');
        if (titleEl) {{
          titleEl.textContent = `${{activeClasses.length}} MAKRO ANOMALİ SINIFI AP (%) SKORLARI`;
        }}

        let cHtml = '';
        activeClasses.forEach(c => {{
          const val = perClass[c] !== undefined ? perClass[c] : 0.0;
          cHtml += `
            <div class="class-item">
              <div class="class-name">
                <span>${{c}}</span>
                <span style="color: var(--neon-cyan); font-weight: bold;">${{Number(val).toFixed(1)}}%</span>
              </div>
              <div class="pbar-track" style="height: 4px;">
                <div class="pbar-fill fill-cyan" style="width: ${{Math.min(100, Math.max(0, val))}}%;"></div>
              </div>
            </div>
          `;
        }});
        classCont.innerHTML = cHtml;

        // Fold Leaderboard Table
        const completedMap = {{}};
        (st.completed_folds || []).forEach(f => {{ completedMap[f.fold] = f; }});
        document.getElementById('folds-completed-count').textContent = `${{Object.keys(completedMap).length}}/${{totalFolds}} Tamamlandı`;

        const tBody = document.getElementById('fold-table-body');
        let tHtml = '';
        for (let i = 1; i <= totalFolds; i++) {{
          const isCurrent = i === curFold && st.is_training;
          const doneInfo = completedMap[i];

          let mAPStr = '-';
          let lossStr = '-';
          let tagHtml = '<span class="status-tag tag-wait">Bekliyor</span>';

          if (doneInfo) {{
            mAPStr = (doneInfo.best_val_map || 0).toFixed(4);
            lossStr = (doneInfo.best_val_loss || 0).toFixed(4);
            tagHtml = '<span class="status-tag tag-done">Tamamlandı</span>';
          }} else if (isCurrent) {{
            mAPStr = (st.best_val_map || st.val_map || 0).toFixed(4);
            lossStr = (st.best_val_loss || st.val_loss || 0).toFixed(4);
            tagHtml = '<span class="status-tag tag-run">Eğitiliyor</span>';
          }}

          tHtml += `
            <tr class="${{isCurrent ? 'row-active' : ''}}">
              <td><b>Fold ${{i}}</b></td>
              <td style="color: var(--neon-green);">${{mAPStr}}</td>
              <td style="color: var(--neon-amber);">${{lossStr}}</td>
              <td>${{tagHtml}}</td>
            </tr>
          `;
        }}
        tBody.innerHTML = tHtml;

        // Terminal Log
        const logBox = document.getElementById('console-output');
        if (data.log) {{
          logBox.textContent = data.log;
          logBox.scrollTop = logBox.scrollHeight;
        }}

      }} catch (e) {{
        console.error("Vitals fetch error:", e);
      }}
    }}

    fetchVitals();
    setInterval(fetchVitals, 250);
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
                "state": get_training_state(),
                "log": get_log_tail(35),
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
            st = get_training_state()
            output = (
                f"GPU: {gpu['util_str']} | Temp: {gpu['temp']}C | VRAM: {gpu['mem_used']}/{gpu['mem_total']} MB\n"
                f"CPU: {cpu}% | RAM: {sram['text']}\n"
                f"STATUS: Fold {st['current_fold']}/{st['total_folds']} | Epoch {st['current_epoch']}/{st['total_epochs']} | Val mAP: {st['val_map']:.4f}\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(output.encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def run_server(port: int = PORT):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), MonitorHandler) as httpd:
        print(f"🛰️ AMOLED Live Monitor running on http://0.0.0.0:{port}")
        print(f"📱 Tailscale URL for Mobile: http://{TAILSCALE_IP}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(p)
