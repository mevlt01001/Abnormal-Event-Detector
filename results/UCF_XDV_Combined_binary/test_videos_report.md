# 🎬 Test_Videos Çıkarım Raporu: `UCF_XDV_Combined_binary`

- **Model Checkpoint**: `checkpoints/UCF_XDV_Combined_binary/best_model.pt`
- **Model Tipi**: Binary (1 Skor)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **11** video
- **Üretilen Anomali Klibi**: **20** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %85.7 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | 🚨 **Anomali** | 98.0s | %94.6 | 3 | `Anomaly` | 3 klip | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %0.4 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | 🚨 **Anomali** | 186.4s | %55.2 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | 🚨 **Anomali** | 10.0s | %85.0 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | 🚨 **Anomali** | 109.2s | %78.0 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %97.4 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_3.mp4` | 🚨 **Anomali** | 35.0s | %88.2 | 1 | `Anomaly` | 1 klip | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | 🚨 **Anomali** | 69.8s | %80.4 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | ✅ Normal | 93.7s | %15.2 | 0 | `Normal` | — | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %79.2 | 3 | `Anomaly` | 3 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %86.8 | 1 | `Anomaly` | 1 klip | [Grafik](plots/video_timeline.png) |
| `test_video_21.mp4` | 🚨 **Anomali** | 27.2s | %77.5 | 2 | `Anomaly` | 2 klip | [Grafik](plots/test_video_21_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %85.7
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[1.4s - 10.7s]` (9.2s) — **Anomaly** (Skor: %85.7)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video20/test_video20_01_Anomaly_1.4s-10.7s.mp4`

---
### 📹 `test_video_0.mp4`
- **Pik Anomali Güveni**: %94.6
- **Zaman Grafiği**: ![test_video_0](plots/test_video_0_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.4s - 10.7s]` (7.3s) — **Anomaly** (Skor: %81.2)
  2. `[25.5s - 31.5s]` (6.0s) — **Anomaly** (Skor: %87.2)
  3. `[66.6s - 75.7s]` (9.1s) — **Anomaly** (Skor: %94.6)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_0/test_video_0_01_Anomaly_3.4s-10.7s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_0/test_video_0_02_Anomaly_25.5s-31.5s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_0/test_video_0_03_Anomaly_66.6s-75.7s.mp4`

---
### 📹 `test_video_11.mp4`
- **Pik Anomali Güveni**: %55.2
- **Zaman Grafiği**: ![test_video_11](plots/test_video_11_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[26.2s - 50.4s]` (24.2s) — **Anomaly** (Skor: %46.0)
  2. `[126.4s - 181.9s]` (55.4s) — **Anomaly** (Skor: %55.2)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_11/test_video_11_01_Anomaly_26.2s-50.4s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_11/test_video_11_02_Anomaly_126.4s-181.9s.mp4`

---
### 📹 `test_video_15.mp4`
- **Pik Anomali Güveni**: %85.0
- **Zaman Grafiği**: ![test_video_15](plots/test_video_15_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 2.3s]` (2.3s) — **Anomaly** (Skor: %85.0)
  2. `[4.1s - 7.4s]` (3.3s) — **Anomaly** (Skor: %68.5)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_15/test_video_15_01_Anomaly_0.0s-2.3s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_15/test_video_15_02_Anomaly_4.1s-7.4s.mp4`

---
### 📹 `test_video_16.mp4`
- **Pik Anomali Güveni**: %78.0
- **Zaman Grafiği**: ![test_video_16](plots/test_video_16_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[56.8s - 75.2s]` (18.3s) — **Anomaly** (Skor: %78.0)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_16/test_video_16_01_Anomaly_56.8s-75.2s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %97.4
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[8.4s - 19.8s]` (11.4s) — **Anomaly** (Skor: %70.7)
  2. `[20.2s - 31.8s]` (11.6s) — **Anomaly** (Skor: %97.4)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_2/test_video_2_01_Anomaly_8.4s-19.8s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_2/test_video_2_02_Anomaly_20.2s-31.8s.mp4`

---
### 📹 `test_video_3.mp4`
- **Pik Anomali Güveni**: %88.2
- **Zaman Grafiği**: ![test_video_3](plots/test_video_3_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[11.0s - 17.3s]` (6.3s) — **Anomaly** (Skor: %88.2)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_3/test_video_3_01_Anomaly_11.0s-17.3s.mp4`

---
### 📹 `test_video_4.mp4`
- **Pik Anomali Güveni**: %80.4
- **Zaman Grafiği**: ![test_video_4](plots/test_video_4_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[4.5s - 9.1s]` (4.6s) — **Anomaly** (Skor: %70.5)
  2. `[38.1s - 44.6s]` (6.5s) — **Anomaly** (Skor: %80.4)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_4/test_video_4_01_Anomaly_4.5s-9.1s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_4/test_video_4_02_Anomaly_38.1s-44.6s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %79.2
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[10.2s - 13.1s]` (2.9s) — **Anomaly** (Skor: %39.4)
  2. `[20.0s - 24.1s]` (4.1s) — **Anomaly** (Skor: %79.2)
  3. `[30.7s - 34.0s]` (3.3s) — **Anomaly** (Skor: %51.6)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_8/test_video_8_01_Anomaly_10.2s-13.1s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_8/test_video_8_02_Anomaly_20.0s-24.1s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_8/test_video_8_03_Anomaly_30.7s-34.0s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %86.8
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.1s - 6.7s]` (3.6s) — **Anomaly** (Skor: %86.8)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/video/video_01_Anomaly_3.1s-6.7s.mp4`

---
### 📹 `test_video_21.mp4`
- **Pik Anomali Güveni**: %77.5
- **Zaman Grafiği**: ![test_video_21](plots/test_video_21_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[16.8s - 22.0s]` (5.2s) — **Anomaly** (Skor: %77.5)
  2. `[22.4s - 24.9s]` (2.5s) — **Anomaly** (Skor: %37.8)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_binary/clips/test_video_21/test_video_21_01_Anomaly_16.8s-22.0s.mp4`
  - `UCF_XDV_Combined_binary/clips/test_video_21/test_video_21_02_Anomaly_22.4s-24.9s.mp4`

---
