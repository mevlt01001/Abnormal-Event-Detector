# 🎬 Test_Videos Çıkarım Raporu: `UCF_XDV_Combined_exact_match`

- **Model Checkpoint**: `checkpoints/UCF_XDV_Combined_exact_match/best_model.pt`
- **Model Tipi**: Çok Sınıflı (5 Sınıf)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **7** video
- **Üretilen Anomali Klibi**: **10** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %40.4 | 1 | `Explosion` | 1 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | ✅ Normal | 98.0s | %26.8 | 0 | `Normal` | — | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %0.0 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | ✅ Normal | 186.4s | %7.5 | 0 | `Normal` | — | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | 🚨 **Anomali** | 10.0s | %64.5 | 2 | `Abuse` | 2 klip | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | ✅ Normal | 109.2s | %0.1 | 0 | `Normal` | — | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %84.4 | 2 | `RoadAccidents` | 2 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_3.mp4` | 🚨 **Anomali** | 35.0s | %58.2 | 1 | `Shooting` | 1 klip | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | ✅ Normal | 69.8s | %7.2 | 0 | `Normal` | — | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | ✅ Normal | 93.7s | %1.2 | 0 | `Normal` | — | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %48.7 | 2 | `RoadAccidents` | 2 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %55.7 | 1 | `Shooting` | 1 klip | [Grafik](plots/video_timeline.png) |
| `test_video_21.mp4` | 🚨 **Anomali** | 27.2s | %49.5 | 1 | `Fighting` | 1 klip | [Grafik](plots/test_video_21_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %40.4
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[5.3s - 7.9s]` (2.6s) — **Explosion** (Skor: %40.4)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/test_video20/test_video20_01_Explosion_5.3s-7.9s.mp4`

---
### 📹 `test_video_15.mp4`
- **Pik Anomali Güveni**: %64.5
- **Zaman Grafiği**: ![test_video_15](plots/test_video_15_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 2.2s]` (2.2s) — **Abuse** (Skor: %64.5)
  2. `[4.1s - 7.2s]` (3.1s) — **Abuse** (Skor: %62.3)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/test_video_15/test_video_15_01_Abuse_0.0s-2.2s.mp4`
  - `UCF_XDV_Combined_exact_match/clips/test_video_15/test_video_15_02_Abuse_4.1s-7.2s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %84.4
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[8.5s - 19.1s]` (10.6s) — **RoadAccidents** (Skor: %72.2)
  2. `[20.4s - 31.8s]` (11.4s) — **RoadAccidents** (Skor: %84.4)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/test_video_2/test_video_2_01_RoadAccidents_8.5s-19.1s.mp4`
  - `UCF_XDV_Combined_exact_match/clips/test_video_2/test_video_2_02_RoadAccidents_20.4s-31.8s.mp4`

---
### 📹 `test_video_3.mp4`
- **Pik Anomali Güveni**: %58.2
- **Zaman Grafiği**: ![test_video_3](plots/test_video_3_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[12.0s - 16.2s]` (4.2s) — **Shooting** (Skor: %58.2)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/test_video_3/test_video_3_01_Shooting_12.0s-16.2s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %48.7
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[21.7s - 25.1s]` (3.4s) — **RoadAccidents** (Skor: %48.7)
  2. `[30.8s - 33.9s]` (3.1s) — **Shooting** (Skor: %45.2)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/test_video_8/test_video_8_01_RoadAccidents_21.7s-25.1s.mp4`
  - `UCF_XDV_Combined_exact_match/clips/test_video_8/test_video_8_02_Shooting_30.8s-33.9s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %55.7
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.1s - 6.1s]` (3.0s) — **Shooting** (Skor: %55.7)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/video/video_01_Shooting_3.1s-6.1s.mp4`

---
### 📹 `test_video_21.mp4`
- **Pik Anomali Güveni**: %49.5
- **Zaman Grafiği**: ![test_video_21](plots/test_video_21_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[17.8s - 21.4s]` (3.6s) — **Fighting** (Skor: %49.5)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_exact_match/clips/test_video_21/test_video_21_01_Fighting_17.8s-21.4s.mp4`

---
