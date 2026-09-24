# 🎬 Test_Videos Çıkarım Raporu: `XDV_binary`

- **Model Checkpoint**: `checkpoints/XDV_binary/best_model.pt`
- **Model Tipi**: Binary (1 Skor)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **7** video
- **Üretilen Anomali Klibi**: **11** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %50.5 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | 🚨 **Anomali** | 98.0s | %73.5 | 4 | `Anomaly` | 4 klip | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %0.1 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | ✅ Normal | 186.4s | %5.6 | 0 | `Normal` | — | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | ✅ Normal | 10.0s | %7.3 | 0 | `Normal` | — | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | ✅ Normal | 109.2s | %14.9 | 0 | `Normal` | — | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %99.9 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_3.mp4` | 🚨 **Anomali** | 35.0s | %38.0 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | ✅ Normal | 69.8s | %2.2 | 0 | `Normal` | — | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | ✅ Normal | 93.7s | %1.6 | 0 | `Normal` | — | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %74.8 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %56.0 | 1 | `Anomaly` | 1 klip | [Grafik](plots/video_timeline.png) |
| `test_video_21.mp4` | 🚨 **Anomali** | 27.2s | %89.5 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_21_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %50.5
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[2.0s - 4.9s]` (2.9s) — **Anomaly** (Skor: %50.5)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/test_video20/test_video20_01_Anomaly_2.0s-4.9s.mp4`

---
### 📹 `test_video_0.mp4`
- **Pik Anomali Güveni**: %73.5
- **Zaman Grafiği**: ![test_video_0](plots/test_video_0_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[4.7s - 8.0s]` (3.3s) — **Anomaly** (Skor: %39.7)
  2. `[25.7s - 31.2s]` (5.5s) — **Anomaly** (Skor: %70.8)
  3. `[67.0s - 75.4s]` (8.4s) — **Anomaly** (Skor: %73.5)
  4. `[85.9s - 91.0s]` (5.1s) — **Anomaly** (Skor: %62.1)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/test_video_0/test_video_0_01_Anomaly_4.7s-8.0s.mp4`
  - `XDV_binary/clips/test_video_0/test_video_0_02_Anomaly_25.7s-31.2s.mp4`
  - `XDV_binary/clips/test_video_0/test_video_0_03_Anomaly_67.0s-75.4s.mp4`
  - `XDV_binary/clips/test_video_0/test_video_0_04_Anomaly_85.9s-91.0s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %99.9
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[5.2s - 31.8s]` (26.6s) — **Anomaly** (Skor: %99.9)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/test_video_2/test_video_2_01_Anomaly_5.2s-31.8s.mp4`

---
### 📹 `test_video_3.mp4`
- **Pik Anomali Güveni**: %38.0
- **Zaman Grafiği**: ![test_video_3](plots/test_video_3_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[11.2s - 13.8s]` (2.6s) — **Anomaly** (Skor: %38.0)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/test_video_3/test_video_3_01_Anomaly_11.2s-13.8s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %74.8
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[30.4s - 34.2s]` (3.8s) — **Anomaly** (Skor: %74.8)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/test_video_8/test_video_8_01_Anomaly_30.4s-34.2s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %56.0
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.2s - 6.3s]` (3.1s) — **Anomaly** (Skor: %56.0)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/video/video_01_Anomaly_3.2s-6.3s.mp4`

---
### 📹 `test_video_21.mp4`
- **Pik Anomali Güveni**: %89.5
- **Zaman Grafiği**: ![test_video_21](plots/test_video_21_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[16.6s - 21.5s]` (4.9s) — **Anomaly** (Skor: %89.5)
  2. `[21.7s - 25.1s]` (3.4s) — **Anomaly** (Skor: %54.2)
- **Kaydedilen Klipler**:
  - `XDV_binary/clips/test_video_21/test_video_21_01_Anomaly_16.6s-21.5s.mp4`
  - `XDV_binary/clips/test_video_21/test_video_21_02_Anomaly_21.7s-25.1s.mp4`

---
