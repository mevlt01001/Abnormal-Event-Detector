# 🎬 Test_Videos Çıkarım Raporu: `UCF_XDV_Combined_multiclass`

- **Model Checkpoint**: `checkpoints/UCF_XDV_Combined_multiclass/best_model.pt`
- **Model Tipi**: Çok Sınıflı (4 Sınıf)
- **Uygulanan Eşik (tau)**: **`0.35`**
- **Zaman Parametreleri**: `tolerance_sec=1.5s`, `padding_sec=1.0s`
- **Toplam Test Videosu**: 13
- **Anomali Tespit Edilen**: **9** video
- **Üretilen Anomali Klibi**: **14** adet MP4

## 📊 Video Çıkarım Özet Tablosu

| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_video20.mp4` | 🚨 **Anomali** | 10.7s | %63.5 | 2 | `Accidents_Disasters` | 2 klip | [Grafik](plots/test_video20_timeline.png) |
| `test_video_0.mp4` | 🚨 **Anomali** | 98.0s | %88.1 | 3 | `Property_Crimes` | 3 klip | [Grafik](plots/test_video_0_timeline.png) |
| `test_video_1.mp4` | ✅ Normal | 17.0s | %0.0 | 0 | `Normal` | — | [Grafik](plots/test_video_1_timeline.png) |
| `test_video_11.mp4` | 🚨 **Anomali** | 186.4s | %67.9 | 1 | `Property_Crimes` | 1 klip | [Grafik](plots/test_video_11_timeline.png) |
| `test_video_15.mp4` | 🚨 **Anomali** | 10.0s | %44.1 | 2 | `Violence_Affray` | 2 klip | [Grafik](plots/test_video_15_timeline.png) |
| `test_video_16.mp4` | 🚨 **Anomali** | 109.2s | %47.1 | 1 | `Property_Crimes` | 1 klip | [Grafik](plots/test_video_16_timeline.png) |
| `test_video_2.mp4` | 🚨 **Anomali** | 31.8s | %89.0 | 2 | `Accidents_Disasters` | 2 klip | [Grafik](plots/test_video_2_timeline.png) |
| `test_video_21.mp4` | ✅ Normal | 27.2s | %31.9 | 0 | `Normal` | — | [Grafik](plots/test_video_21_timeline.png) |
| `test_video_3.mp4` | 🚨 **Anomali** | 35.0s | %37.4 | 1 | `Property_Crimes` | 1 klip | [Grafik](plots/test_video_3_timeline.png) |
| `test_video_4.mp4` | ✅ Normal | 69.8s | %27.8 | 0 | `Normal` | — | [Grafik](plots/test_video_4_timeline.png) |
| `test_video_6.mp4` | ✅ Normal | 93.7s | %2.9 | 0 | `Normal` | — | [Grafik](plots/test_video_6_timeline.png) |
| `test_video_8.mp4` | 🚨 **Anomali** | 45.6s | %43.2 | 1 | `Gun_Violence` | 1 klip | [Grafik](plots/test_video_8_timeline.png) |
| `video.mp4` | 🚨 **Anomali** | 19.9s | %44.6 | 1 | `Gun_Violence` | 1 klip | [Grafik](plots/video_timeline.png) |

## 🔍 Tespit Detayları ve Anomali Aralıkları

### 📹 `test_video20.mp4`
- **Pik Anomali Güveni**: %63.5
- **Zaman Grafiği**: ![test_video20](plots/test_video20_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[1.9s - 5.5s]` (3.6s) — **Accidents_Disasters** (Skor: %61.6)
  2. `[5.8s - 10.6s]` (4.7s) — **Accidents_Disasters** (Skor: %63.5)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video20/test_video20_01_Accidents_Disasters_1.9s-5.5s.mp4`
  - `UCF_XDV_Combined_multiclass/clips/test_video20/test_video20_02_Accidents_Disasters_5.8s-10.6s.mp4`

---
### 📹 `test_video_0.mp4`
- **Pik Anomali Güveni**: %88.1
- **Zaman Grafiği**: ![test_video_0](plots/test_video_0_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[1.1s - 6.1s]` (5.0s) — **Property_Crimes** (Skor: %49.2)
  2. `[68.7s - 75.7s]` (7.0s) — **Accidents_Disasters** (Skor: %88.1)
  3. `[86.6s - 90.4s]` (3.8s) — **Accidents_Disasters** (Skor: %44.6)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_0/test_video_0_01_Property_Crimes_1.1s-6.1s.mp4`
  - `UCF_XDV_Combined_multiclass/clips/test_video_0/test_video_0_02_Accidents_Disasters_68.7s-75.7s.mp4`
  - `UCF_XDV_Combined_multiclass/clips/test_video_0/test_video_0_03_Accidents_Disasters_86.6s-90.4s.mp4`

---
### 📹 `test_video_11.mp4`
- **Pik Anomali Güveni**: %67.9
- **Zaman Grafiği**: ![test_video_11](plots/test_video_11_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[115.6s - 177.4s]` (61.7s) — **Property_Crimes** (Skor: %67.9)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_11/test_video_11_01_Property_Crimes_115.6s-177.4s.mp4`

---
### 📹 `test_video_15.mp4`
- **Pik Anomali Güveni**: %44.1
- **Zaman Grafiği**: ![test_video_15](plots/test_video_15_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[0.0s - 1.3s]` (1.3s) — **Violence_Affray** (Skor: %37.2)
  2. `[4.7s - 7.3s]` (2.6s) — **Violence_Affray** (Skor: %44.1)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_15/test_video_15_01_Violence_Affray_0.0s-1.3s.mp4`
  - `UCF_XDV_Combined_multiclass/clips/test_video_15/test_video_15_02_Violence_Affray_4.7s-7.3s.mp4`

---
### 📹 `test_video_16.mp4`
- **Pik Anomali Güveni**: %47.1
- **Zaman Grafiği**: ![test_video_16](plots/test_video_16_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[58.2s - 72.6s]` (14.4s) — **Property_Crimes** (Skor: %47.1)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_16/test_video_16_01_Property_Crimes_58.2s-72.6s.mp4`

---
### 📹 `test_video_2.mp4`
- **Pik Anomali Güveni**: %89.0
- **Zaman Grafiği**: ![test_video_2](plots/test_video_2_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[20.8s - 24.7s]` (3.9s) — **Accidents_Disasters** (Skor: %79.0)
  2. `[29.0s - 31.8s]` (2.8s) — **Accidents_Disasters** (Skor: %89.0)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_2/test_video_2_01_Accidents_Disasters_20.8s-24.7s.mp4`
  - `UCF_XDV_Combined_multiclass/clips/test_video_2/test_video_2_02_Accidents_Disasters_29.0s-31.8s.mp4`

---
### 📹 `test_video_3.mp4`
- **Pik Anomali Güveni**: %37.4
- **Zaman Grafiği**: ![test_video_3](plots/test_video_3_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[33.7s - 35.0s]` (1.3s) — **Property_Crimes** (Skor: %37.4)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_3/test_video_3_01_Property_Crimes_33.7s-35.0s.mp4`

---
### 📹 `test_video_8.mp4`
- **Pik Anomali Güveni**: %43.2
- **Zaman Grafiği**: ![test_video_8](plots/test_video_8_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[30.8s - 33.8s]` (3.0s) — **Gun_Violence** (Skor: %43.2)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/test_video_8/test_video_8_01_Gun_Violence_30.8s-33.8s.mp4`

---
### 📹 `video.mp4`
- **Pik Anomali Güveni**: %44.6
- **Zaman Grafiği**: ![video](plots/video_timeline.png)
- **Tespit Edilen Olay Segmentleri**:
  1. `[3.2s - 5.9s]` (2.7s) — **Gun_Violence** (Skor: %44.6)
- **Kaydedilen Klipler**:
  - `UCF_XDV_Combined_multiclass/clips/video/video_01_Gun_Violence_3.2s-5.9s.mp4`

---
