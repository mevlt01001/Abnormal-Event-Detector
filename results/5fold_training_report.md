# 5-Fold Sınıf-Farkında (Class-Aware) Eğitim Raporu

- **Tarih:** 2026-09-22 12:19:44
- **Backbone Modeli:** mvit_v2_s (Öznitelik Boyutu: 768)
- **Örnekleme:** 20 FPS, Örtüşme (Overlap): 0.5
- **Makro Sınıflar:** 6 (Violence, Shooting, Explosion_Fire, Accident, Riot_Vandalism, Theft_Robbery)
- **Veri Seti:** `data/unified_features` (Toplam: 5409 video)
- **Fold Sayısı:** 5
- **Epoch / Fold:** 20
- **Toplam Süre:** 221.4 saniye

## 1. Genel Performans Özeti

| Metrik | Ortalama (Mean) | Standart Sapma (Std) |
| :--- | :---: | :---: |
| **Doğrulama mAP** | **0.7463** | ± 0.0114 |
| **Doğrulama ROC-AUC** | **0.9531** | ± 0.0070 |
| **Doğrulama Kaybı (Loss)** | **0.2655** | ± 0.0095 |

## 2. Fold Bazlı Detaylar

| Fold | En İyi mAP | En İyi ROC-AUC | En İyi Val Loss | En İyi Epoch |
| :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 0.7326 | 0.9466 | 0.2758 | Epoch 15 |
| Fold 2 | 0.7387 | 0.9565 | 0.2650 | Epoch 15 |
| Fold 3 | 0.7408 | 0.9626 | 0.2571 | Epoch 15 |
| Fold 4 | 0.7565 | 0.9563 | 0.2531 | Epoch 11 |
| Fold 5 | 0.7627 | 0.9437 | 0.2763 | Epoch 14 |
