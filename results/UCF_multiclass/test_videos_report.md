# 🎬 Test_Videos Çıkarım Raporu: `UCF_multiclass`

- **Model Checkpoint**: `checkpoints/UCF_multiclass/best_model.pt`
- **Model Tipi**: Çok Sınıflı (13 Sınıf)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **12** video
- **Üretilen Anomali Klibi**: **22** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %63.7 | 2 | `RoadAccidents` | 2 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | 🚨 **Anomali** | 98.0s | %57.0 | 3 | `Robbery` | 3 klip | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %1.9 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | 🚨 **Anomali** | 186.4s | %95.8 | 3 | `Stealing` | 3 klip | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | 🚨 **Anomali** | 10.0s | %43.2 | 2 | `Abuse` | 2 klip | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | 🚨 **Anomali** | 109.2s | %35.6 | 1 | `Stealing` | 1 klip | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %57.8 | 2 | `RoadAccidents` | 2 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_3.mp4` | 🚨 **Anomali** | 35.0s | %72.1 | 3 | `Robbery` | 3 klip | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | 🚨 **Anomali** | 69.8s | %38.7 | 1 | `Burglary` | 1 klip | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | 🚨 **Anomali** | 93.7s | %45.5 | 1 | `Stealing` | 1 klip | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %55.6 | 1 | `RoadAccidents` | 1 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %37.2 | 1 | `Arson` | 1 klip | [Grafik](plots/video_timeline.png) |
| `test_video_21.mp4` | 🚨 **Anomali** | 27.2s | %55.7 | 2 | `Stealing` | 2 klip | [Grafik](plots/test_video_21_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %63.7
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[2.8s - 6.1s]` (3.3s) — **RoadAccidents** (Skor: %56.8)
  2. `[6.5s - 10.7s]` (4.2s) — **RoadAccidents** (Skor: %63.7)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video20/test_video20_01_RoadAccidents_2.8s-6.1s.mp4`
  - `UCF_multiclass/clips/test_video20/test_video20_02_RoadAccidents_6.5s-10.7s.mp4`

---
### 📹 `test_video_0.mp4`
- **Pik Anomali Güveni**: %57.0
- **Zaman Grafiği**: ![test_video_0](plots/test_video_0_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.8s - 5.9s]` (5.1s) — **Robbery** (Skor: %57.0)
  2. `[45.1s - 49.7s]` (4.6s) — **Burglary** (Skor: %52.6)
  3. `[89.1s - 94.2s]` (5.1s) — **Burglary** (Skor: %51.6)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_0/test_video_0_01_Robbery_0.8s-5.9s.mp4`
  - `UCF_multiclass/clips/test_video_0/test_video_0_02_Burglary_45.1s-49.7s.mp4`
  - `UCF_multiclass/clips/test_video_0/test_video_0_03_Burglary_89.1s-94.2s.mp4`

---
### 📹 `test_video_11.mp4`
- **Pik Anomali Güveni**: %95.8
- **Zaman Grafiği**: ![test_video_11](plots/test_video_11_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 35.8s]` (35.8s) — **Stealing** (Skor: %81.4)
  2. `[36.4s - 100.3s]` (63.9s) — **Stealing** (Skor: %67.6)
  3. `[109.7s - 185.2s]` (75.4s) — **Stealing** (Skor: %95.8)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_11/test_video_11_01_Stealing_0.0s-35.8s.mp4`
  - `UCF_multiclass/clips/test_video_11/test_video_11_02_Stealing_36.4s-100.3s.mp4`
  - `UCF_multiclass/clips/test_video_11/test_video_11_03_Stealing_109.7s-185.2s.mp4`

---
### 📹 `test_video_15.mp4`
- **Pik Anomali Güveni**: %43.2
- **Zaman Grafiği**: ![test_video_15](plots/test_video_15_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 1.4s]` (1.4s) — **Abuse** (Skor: %43.2)
  2. `[7.0s - 9.6s]` (2.6s) — **Stealing** (Skor: %37.0)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_15/test_video_15_01_Abuse_0.0s-1.4s.mp4`
  - `UCF_multiclass/clips/test_video_15/test_video_15_02_Stealing_7.0s-9.6s.mp4`

---
### 📹 `test_video_16.mp4`
- **Pik Anomali Güveni**: %35.6
- **Zaman Grafiği**: ![test_video_16](plots/test_video_16_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[62.3s - 65.3s]` (3.0s) — **Stealing** (Skor: %35.6)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_16/test_video_16_01_Stealing_62.3s-65.3s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %57.8
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[21.0s - 24.2s]` (3.2s) — **RoadAccidents** (Skor: %50.9)
  2. `[29.1s - 31.8s]` (2.7s) — **RoadAccidents** (Skor: %57.8)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_2/test_video_2_01_RoadAccidents_21.0s-24.2s.mp4`
  - `UCF_multiclass/clips/test_video_2/test_video_2_02_RoadAccidents_29.1s-31.8s.mp4`

---
### 📹 `test_video_3.mp4`
- **Pik Anomali Güveni**: %72.1
- **Zaman Grafiği**: ![test_video_3](plots/test_video_3_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 1.3s]` (1.3s) — **Robbery** (Skor: %37.5)
  2. `[7.7s - 10.5s]` (2.8s) — **Robbery** (Skor: %41.9)
  3. `[10.5s - 15.6s]` (5.1s) — **Robbery** (Skor: %72.1)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_3/test_video_3_01_Robbery_0.0s-1.3s.mp4`
  - `UCF_multiclass/clips/test_video_3/test_video_3_02_Robbery_7.7s-10.5s.mp4`
  - `UCF_multiclass/clips/test_video_3/test_video_3_03_Robbery_10.5s-15.6s.mp4`

---
### 📹 `test_video_4.mp4`
- **Pik Anomali Güveni**: %38.7
- **Zaman Grafiği**: ![test_video_4](plots/test_video_4_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[67.4s - 69.8s]` (2.4s) — **Burglary** (Skor: %38.7)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_4/test_video_4_01_Burglary_67.4s-69.8s.mp4`

---
### 📹 `test_video_6.mp4`
- **Pik Anomali Güveni**: %45.5
- **Zaman Grafiği**: ![test_video_6](plots/test_video_6_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[79.7s - 84.8s]` (5.1s) — **Stealing** (Skor: %45.5)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_6/test_video_6_01_Stealing_79.7s-84.8s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %55.6
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[21.7s - 29.5s]` (7.8s) — **RoadAccidents** (Skor: %55.6)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_8/test_video_8_01_RoadAccidents_21.7s-29.5s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %37.2
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[4.0s - 6.5s]` (2.5s) — **Arson** (Skor: %37.2)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/video/video_01_Arson_4.0s-6.5s.mp4`

---
### 📹 `test_video_21.mp4`
- **Pik Anomali Güveni**: %55.7
- **Zaman Grafiği**: ![test_video_21](plots/test_video_21_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.9s - 7.4s]` (3.5s) — **Stealing** (Skor: %55.7)
  2. `[18.9s - 21.8s]` (2.9s) — **Robbery** (Skor: %40.8)
- **Kaydedilen Klipler**:
  - `UCF_multiclass/clips/test_video_21/test_video_21_01_Stealing_3.9s-7.4s.mp4`
  - `UCF_multiclass/clips/test_video_21/test_video_21_02_Robbery_18.9s-21.8s.mp4`

---
