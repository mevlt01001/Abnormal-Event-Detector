# 🎬 Test_Videos Çıkarım Raporu: `XDV_multiclass`

- **Model Checkpoint**: `checkpoints/XDV_multiclass/best_model.pt`
- **Model Tipi**: Çok Sınıflı (6 Sınıf)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **8** video
- **Üretilen Anomali Klibi**: **10** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %51.7 | 1 | `Explosion` | 1 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | 🚨 **Anomali** | 98.0s | %61.9 | 3 | `Explosion` | 3 klip | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %0.5 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | ✅ Normal | 186.4s | %2.7 | 0 | `Normal` | — | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | 🚨 **Anomali** | 10.0s | %35.0 | 1 | `Fighting` | 1 klip | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | 🚨 **Anomali** | 109.2s | %47.7 | 1 | `Fighting` | 1 klip | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %96.9 | 1 | `RoadAccidents` | 1 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_3.mp4` | ✅ Normal | 35.0s | %18.5 | 0 | `Normal` | — | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | ✅ Normal | 69.8s | %15.0 | 0 | `Normal` | — | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | ✅ Normal | 93.7s | %27.2 | 0 | `Normal` | — | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %39.0 | 1 | `Fighting` | 1 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %45.4 | 1 | `Shooting` | 1 klip | [Grafik](plots/video_timeline.png) |
| `test_video_21.mp4` | 🚨 **Anomali** | 27.2s | %68.4 | 1 | `Fighting` | 1 klip | [Grafik](plots/test_video_21_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %51.7
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[1.9s - 5.0s]` (3.1s) — **Explosion** (Skor: %51.7)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video20/test_video20_01_Explosion_1.9s-5.0s.mp4`

---
### 📹 `test_video_0.mp4`
- **Pik Anomali Güveni**: %61.9
- **Zaman Grafiği**: ![test_video_0](plots/test_video_0_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[20.0s - 24.6s]` (4.6s) — **Explosion** (Skor: %50.9)
  2. `[25.6s - 31.0s]` (5.4s) — **Explosion** (Skor: %61.9)
  3. `[71.3s - 73.5s]` (2.2s) — **Explosion** (Skor: %35.0)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video_0/test_video_0_01_Explosion_20.0s-24.6s.mp4`
  - `XDV_multiclass/clips/test_video_0/test_video_0_02_Explosion_25.6s-31.0s.mp4`
  - `XDV_multiclass/clips/test_video_0/test_video_0_03_Explosion_71.3s-73.5s.mp4`

---
### 📹 `test_video_15.mp4`
- **Pik Anomali Güveni**: %35.0
- **Zaman Grafiği**: ![test_video_15](plots/test_video_15_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[5.1s - 7.2s]` (2.1s) — **Fighting** (Skor: %35.0)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video_15/test_video_15_01_Fighting_5.1s-7.2s.mp4`

---
### 📹 `test_video_16.mp4`
- **Pik Anomali Güveni**: %47.7
- **Zaman Grafiği**: ![test_video_16](plots/test_video_16_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[68.1s - 72.6s]` (4.5s) — **Fighting** (Skor: %47.7)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video_16/test_video_16_01_Fighting_68.1s-72.6s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %96.9
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 31.8s]` (31.8s) — **RoadAccidents** (Skor: %96.9)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video_2/test_video_2_01_RoadAccidents_0.0s-31.8s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %39.0
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[41.3s - 44.0s]` (2.7s) — **Fighting** (Skor: %39.0)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video_8/test_video_8_01_Fighting_41.3s-44.0s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %45.4
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.3s - 6.1s]` (2.8s) — **Shooting** (Skor: %45.4)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/video/video_01_Shooting_3.3s-6.1s.mp4`

---
### 📹 `test_video_21.mp4`
- **Pik Anomali Güveni**: %68.4
- **Zaman Grafiği**: ![test_video_21](plots/test_video_21_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[17.4s - 21.8s]` (4.4s) — **Fighting** (Skor: %68.4)
- **Kaydedilen Klipler**:
  - `XDV_multiclass/clips/test_video_21/test_video_21_01_Fighting_17.4s-21.8s.mp4`

---
