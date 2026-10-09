"""
Triton Inference Server Üzerinden Video Anomali Tespiti Test Betiği.
Eğitilmiş AnomalyDetector (S3D + AnomalyHead) modelini Triton GPU sunucusunda test eder.
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2
import tritonclient.http as httpclient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_clips_from_video(video_path: str, clip_len: int = 16, imgsz: int = 224):
    """Videodan 16 karelik klipleri okur ve [N, 3, 16, 224, 224] tensörüne dönüştürür."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB ve 224x224 resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (imgsz, imgsz))
        frames.append(frame_resized)
    cap.release()

    if len(frames) < clip_len:
        # Yeterli kare yoksa döngüyle uzat
        frames = (frames * ((clip_len // len(frames)) + 1))[:clip_len]

    # [T, H, W, C] -> [T, C, H, W]
    frames_arr = np.array(frames, dtype=np.float32) / 255.0

    # 16'lık segmentlere böl
    clips = []
    stride = clip_len // 2  # %50 örtüşmeli klipler
    for i in range(0, len(frames_arr) - clip_len + 1, stride):
        clip = frames_arr[i : i + clip_len]  # [16, H, W, C]
        clip = clip.transpose(3, 0, 1, 2)     # [3, 16, H, W]
        clips.append(clip)

    if not clips:
        clip = frames_arr[:clip_len].transpose(3, 0, 1, 2)
        clips.append(clip)

    return np.array(clips, dtype=np.float32)  # [N, 3, 16, 224, 224]


def run_triton_anomaly_test(
    video_path: str = "Anomaly_Detection/Test_Videos/test_video20.mp4",
    triton_url: str = "localhost:8000",
    model_name: str = "anomaly_detector",
):
    print("=" * 65)
    print("  NVIDIA Triton Inference Server: Anomaly Detector Testi")
    print("=" * 65)

    client = httpclient.InferenceServerClient(url=triton_url)
    if not client.is_server_live():
        print(f"🔴 Hata: Triton sunucusu ({triton_url}) kapalı.")
        return

    if not client.is_model_ready(model_name):
        print(f"🔴 Hata: '{model_name}' modeli Triton'da hazır değil.")
        return

    print(f"🟢 Triton Sunucusu Aktif & Model '{model_name}' Hazır!")

    # 1. Videoyu oku ve klipleri hazırla
    print(f"\n[1/3] Video işleniyor: {video_path}")
    clips = extract_clips_from_video(video_path, clip_len=16, imgsz=224)
    print(f"      Toplam Segment (Klip) Sayısı: {len(clips)} (Her klip 16 kare)")

    # 2. Triton üzerinden çıkarım yap
    print(f"\n[2/3] Triton üzerinden GPU çıkarımı yapılıyor...")
    scores = []
    latencies = []

    for i, clip in enumerate(clips):
        clip_batch = np.expand_dims(clip, axis=0)  # [1, 3, 16, 224, 224]

        inp = httpclient.InferInput("video_segment", clip_batch.shape, "FP32")
        inp.set_data_from_numpy(clip_batch)
        out = httpclient.InferRequestedOutput("logits")

        t0 = time.perf_counter()
        res = client.infer(model_name, inputs=[inp], outputs=[out])
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000
        latencies.append(lat_ms)

        logits = res.as_numpy("logits")
        # Sigmoid ile anomali olasılığı [0.0 - 1.0]
        prob = 1.0 / (1.0 + np.exp(-logits[0][0]))
        scores.append(prob)

        status_icon = "🚨 ANOMALİ" if prob > 0.50 else "🟢 Normal"
        print(f"      Segment {i+1:2d}/{len(clips)} -> Skor: %{prob * 100:5.1f} ({status_icon}) | Gecikme: {lat_ms:.1f} ms")

    # 3. Özet Sonuçlar
    max_score = max(scores)
    avg_lat = np.mean(latencies)

    print("\n" + "=" * 65)
    print("                  ANOMALİ TESPİT SONUÇLARI")
    print("=" * 65)
    print(f"Video Dosyası        : {video_path}")
    print(f"İşlenen Segmentler   : {len(clips)} klip ({len(clips) * 16} kare)")
    print(f"Maksimum Anomali     : %{max_score * 100:.2f} ({'🚨 ANOMALİ TESPİT EDİLDİ' if max_score > 0.5 else '🟢 TEMİZ'})")
    print(f"Ortalama Çıkarım Hızı: {avg_lat:.2f} ms/segment ({1000 / avg_lat:.1f} Clip/sn)")
    print("=" * 65)


if __name__ == "__main__":
    test_video = "Anomaly_Detection/Test_Videos/test_video20.mp4"
    run_triton_anomaly_test(video_path=test_video)
