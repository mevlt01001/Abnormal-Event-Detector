# 🧪 Toplu Test_Videos Sınıf-Farkında Anomali Tespit Raporu

- **Tarih:** 2026-09-20 13:17:51
- **Test Edilen Model:** `checkpoints/class_aware_8fold/best_map_fold_3.pt`
- **Fold / Epoch:** Fold 3 / Epoch 15
- **Model Val mAP:** `0.7353`
- **Model Val ROC-AUC:** `0.9483`
- **Eşik Değeri (Threshold):** `0.35`
- **Toplam Video:** 10 adet (6 Anomali, 4 Normal)
- **Toplam Test Süresi:** 4.35 saniye

## 1. Toplu Test Sonuçları Tablosu

| Video Adı | Süre (s) | Durum | Baskın Sınıf | Tepe Skor (%) | Anomali Aralıkları | Grafik |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| `test_video_0.mp4` | 98.0s | 🚨 **ANOMALİ** | Explosion_Fire | %79.8 | `11.01s-21.12s` (Explosion_Fire %80) | [Grafik](test_video_0_class_aware_timeline.png) |
| `test_video_1.mp4` | 17.0s | ✅ **NORMAL** | - | %1.0 | - | [Grafik](test_video_1_class_aware_timeline.png) |
| `test_video_11.mp4` | 186.4s | 🚨 **ANOMALİ** | Theft_Robbery | %58.4 | `0.0s-13.51s` (Theft_Robbery %55)<br>`69.96s-92.98s` (Theft_Robbery %58) | [Grafik](test_video_11_class_aware_timeline.png) |
| `test_video_15.mp4` | 10.0s | ✅ **NORMAL** | - | %25.8 | - | [Grafik](test_video_15_class_aware_timeline.png) |
| `test_video_16.mp4` | 109.2s | ✅ **NORMAL** | - | %0.1 | - | [Grafik](test_video_16_class_aware_timeline.png) |
| `test_video_2.mp4` | 31.8s | 🚨 **ANOMALİ** | Accident | %82.4 | `0.0s-31.8s` (Accident %82) | [Grafik](test_video_2_class_aware_timeline.png) |
| `test_video_3.mp4` | 35.0s | 🚨 **ANOMALİ** | Explosion_Fire | %40.7 | `16.16s-17.17s` (Explosion_Fire %41) | [Grafik](test_video_3_class_aware_timeline.png) |
| `test_video_4.mp4` | 69.8s | ✅ **NORMAL** | - | %35.1 | - | [Grafik](test_video_4_class_aware_timeline.png) |
| `test_video_6.mp4` | 93.7s | 🚨 **ANOMALİ** | Accident | %39.9 | `1.7s-3.7s` (Accident %40) | [Grafik](test_video_6_class_aware_timeline.png) |
| `test_video_8.mp4` | 45.6s | 🚨 **ANOMALİ** | Violence | %53.5 | `1.4s-4.41s` (Violence %54) | [Grafik](test_video_8_class_aware_timeline.png) |

## 2. Tespit Edilen Olay Detayları

### 🎬 test_video_0.mp4
- **Toplam Süre:** 98.0s
- **Maksimum Skor:** %79.8 (Explosion_Fire)
- **Olay Aralıkları:**
  - `[11.01s - 21.12s]` (10.11s): **Explosion_Fire** (Pik: %79.8)
- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/test_video_0_class_aware_timeline.png`

### 🎬 test_video_11.mp4
- **Toplam Süre:** 186.36s
- **Maksimum Skor:** %58.4 (Theft_Robbery)
- **Olay Aralıkları:**
  - `[0.0s - 13.51s]` (13.51s): **Theft_Robbery** (Pik: %54.5)
  - `[69.96s - 92.98s]` (23.02s): **Theft_Robbery** (Pik: %58.4)
- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/test_video_11_class_aware_timeline.png`

### 🎬 test_video_2.mp4
- **Toplam Süre:** 31.8s
- **Maksimum Skor:** %82.4 (Accident)
- **Olay Aralıkları:**
  - `[0.0s - 31.8s]` (31.8s): **Accident** (Pik: %82.4)
- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/test_video_2_class_aware_timeline.png`

### 🎬 test_video_3.mp4
- **Toplam Süre:** 35.04s
- **Maksimum Skor:** %40.7 (Explosion_Fire)
- **Olay Aralıkları:**
  - `[16.16s - 17.17s]` (1.0s): **Explosion_Fire** (Pik: %40.7)
- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/test_video_3_class_aware_timeline.png`

### 🎬 test_video_6.mp4
- **Toplam Süre:** 93.72s
- **Maksimum Skor:** %39.9 (Accident)
- **Olay Aralıkları:**
  - `[1.7s - 3.7s]` (2.0s): **Accident** (Pik: %39.9)
- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/test_video_6_class_aware_timeline.png`

### 🎬 test_video_8.mp4
- **Toplam Süre:** 45.65s
- **Maksimum Skor:** %53.5 (Violence)
- **Olay Aralıkları:**
  - `[1.4s - 4.41s]` (3.01s): **Violence** (Pik: %53.5)
- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/test_video_8_class_aware_timeline.png`

