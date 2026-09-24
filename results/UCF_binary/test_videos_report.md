# 🎬 Test_Videos Çıkarım Raporu: `UCF_binary`

- **Model Checkpoint**: `checkpoints/UCF_binary/best_model.pt`
- **Model Tipi**: Binary (1 Skor)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **12** video
- **Üretilen Anomali Klibi**: **21** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %88.3 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | 🚨 **Anomali** | 98.0s | %65.6 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %4.3 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | 🚨 **Anomali** | 186.4s | %88.0 | 3 | `Anomaly` | 3 klip | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | 🚨 **Anomali** | 10.0s | %73.6 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | 🚨 **Anomali** | 109.2s | %41.0 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %35.3 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_3.mp4` | 🚨 **Anomali** | 35.0s | %92.3 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | 🚨 **Anomali** | 69.8s | %76.9 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | 🚨 **Anomali** | 93.7s | %43.0 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %81.9 | 4 | `Anomaly` | 4 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %93.3 | 1 | `Anomaly` | 1 klip | [Grafik](plots/video_timeline.png) |
| `test_video_21.mp4` | 🚨 **Anomali** | 27.2s | %76.9 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_21_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %88.3
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[1.8s - 10.7s]` (8.8s) — **Anomaly** (Skor: %88.3)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video20/test_video20_01_Anomaly_1.8s-10.7s.mp4`

---
### 📹 `test_video_0.mp4`
- **Pik Anomali Güveni**: %65.6
- **Zaman Grafiği**: ![test_video_0](plots/test_video_0_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[4.7s - 10.5s]` (5.8s) — **Anomaly** (Skor: %40.3)
  2. `[61.2s - 73.0s]` (11.8s) — **Anomaly** (Skor: %65.6)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_0/test_video_0_01_Anomaly_4.7s-10.5s.mp4`
  - `UCF_binary/clips/test_video_0/test_video_0_02_Anomaly_61.2s-73.0s.mp4`

---
### 📹 `test_video_11.mp4`
- **Pik Anomali Güveni**: %88.0
- **Zaman Grafiği**: ![test_video_11](plots/test_video_11_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 4.0s]` (4.0s) — **Anomaly** (Skor: %42.0)
  2. `[7.3s - 72.5s]` (65.2s) — **Anomaly** (Skor: %88.0)
  3. `[93.9s - 184.6s]` (90.6s) — **Anomaly** (Skor: %87.1)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_11/test_video_11_01_Anomaly_0.0s-4.0s.mp4`
  - `UCF_binary/clips/test_video_11/test_video_11_02_Anomaly_7.3s-72.5s.mp4`
  - `UCF_binary/clips/test_video_11/test_video_11_03_Anomaly_93.9s-184.6s.mp4`

---
### 📹 `test_video_15.mp4`
- **Pik Anomali Güveni**: %73.6
- **Zaman Grafiği**: ![test_video_15](plots/test_video_15_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 2.1s]` (2.1s) — **Anomaly** (Skor: %73.6)
  2. `[4.3s - 9.2s]` (4.9s) — **Anomaly** (Skor: %46.2)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_15/test_video_15_01_Anomaly_0.0s-2.1s.mp4`
  - `UCF_binary/clips/test_video_15/test_video_15_02_Anomaly_4.3s-9.2s.mp4`

---
### 📹 `test_video_16.mp4`
- **Pik Anomali Güveni**: %41.0
- **Zaman Grafiği**: ![test_video_16](plots/test_video_16_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[5.4s - 10.0s]` (4.6s) — **Anomaly** (Skor: %41.0)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_16/test_video_16_01_Anomaly_5.4s-10.0s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %35.3
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[29.6s - 31.8s]` (2.2s) — **Anomaly** (Skor: %35.3)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_2/test_video_2_01_Anomaly_29.6s-31.8s.mp4`

---
### 📹 `test_video_3.mp4`
- **Pik Anomali Güveni**: %92.3
- **Zaman Grafiği**: ![test_video_3](plots/test_video_3_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[9.9s - 18.7s]` (8.8s) — **Anomaly** (Skor: %92.3)
  2. `[33.3s - 35.0s]` (1.7s) — **Anomaly** (Skor: %61.3)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_3/test_video_3_01_Anomaly_9.9s-18.7s.mp4`
  - `UCF_binary/clips/test_video_3/test_video_3_02_Anomaly_33.3s-35.0s.mp4`

---
### 📹 `test_video_4.mp4`
- **Pik Anomali Güveni**: %76.9
- **Zaman Grafiği**: ![test_video_4](plots/test_video_4_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[4.4s - 9.3s]` (4.9s) — **Anomaly** (Skor: %65.5)
  2. `[37.5s - 45.2s]` (7.7s) — **Anomaly** (Skor: %76.9)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_4/test_video_4_01_Anomaly_4.4s-9.3s.mp4`
  - `UCF_binary/clips/test_video_4/test_video_4_02_Anomaly_37.5s-45.2s.mp4`

---
### 📹 `test_video_6.mp4`
- **Pik Anomali Güveni**: %43.0
- **Zaman Grafiği**: ![test_video_6](plots/test_video_6_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[15.8s - 23.5s]` (7.7s) — **Anomaly** (Skor: %43.0)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_6/test_video_6_01_Anomaly_15.8s-23.5s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %81.9
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[20.6s - 24.1s]` (3.5s) — **Anomaly** (Skor: %45.3)
  2. `[26.0s - 30.9s]` (4.9s) — **Anomaly** (Skor: %81.9)
  3. `[31.2s - 34.5s]` (3.3s) — **Anomaly** (Skor: %39.6)
  4. `[41.3s - 45.5s]` (4.2s) — **Anomaly** (Skor: %46.2)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_8/test_video_8_01_Anomaly_20.6s-24.1s.mp4`
  - `UCF_binary/clips/test_video_8/test_video_8_02_Anomaly_26.0s-30.9s.mp4`
  - `UCF_binary/clips/test_video_8/test_video_8_03_Anomaly_31.2s-34.5s.mp4`
  - `UCF_binary/clips/test_video_8/test_video_8_04_Anomaly_41.3s-45.5s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %93.3
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.0s - 19.9s]` (16.9s) — **Anomaly** (Skor: %93.3)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/video/video_01_Anomaly_3.0s-19.9s.mp4`

---
### 📹 `test_video_21.mp4`
- **Pik Anomali Güveni**: %76.9
- **Zaman Grafiği**: ![test_video_21](plots/test_video_21_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[16.7s - 22.5s]` (5.8s) — **Anomaly** (Skor: %76.9)
- **Kaydedilen Klipler**:
  - `UCF_binary/clips/test_video_21/test_video_21_01_Anomaly_16.7s-22.5s.mp4`

---
