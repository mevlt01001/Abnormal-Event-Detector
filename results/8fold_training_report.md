# 8-Fold Sınıf-Farkında (Class-Aware) Eğitim Raporu

- **Tarih:** 2026-09-20 13:12:07
- **Makro Sınıflar:** 6 (Violence, Shooting, Explosion_Fire, Accident, Riot_Vandalism, Theft_Robbery)
- **Öznitelik Boyutu:** 768
- **Veri Seti:** `data/unified_features`
- **Fold Sayısı:** 5
- **Epoch / Fold:** 20
- **Toplam Süre:** 228.4 saniye

## 1. Genel Performans Özeti

| Metrik | Ortalama (Mean) | Standart Sapma (Std) |
| :--- | :---: | :---: |
| **Doğrulama mAP** | **0.7276** | ± 0.0156 |
| **Doğrulama ROC-AUC** | **0.9444** | ± 0.0036 |
| **Doğrulama Kaybı (Loss)** | **0.2879** | ± 0.0071 |

## 2. Fold Bazlı Detaylar

| Fold | En İyi mAP | En İyi ROC-AUC | En İyi Val Loss | En İyi Epoch |
| :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 0.7086 | 0.9383 | 0.2952 | Epoch 13 |
| Fold 2 | 0.7122 | 0.9426 | 0.2973 | Epoch 11 |
| Fold 3 | 0.7353 | 0.9483 | 0.2831 | Epoch 15 |
| Fold 4 | 0.7310 | 0.9476 | 0.2789 | Epoch 17 |
| Fold 5 | 0.7511 | 0.9452 | 0.2850 | Epoch 14 |
