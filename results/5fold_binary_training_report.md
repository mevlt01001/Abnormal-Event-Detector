# 5-Fold Binary Video Anomali Tespiti Eğitim Raporu

- **Tarih:** 2026-09-22 12:45:35
- **Model Mimarisi:** Deep Pyramidal Binary Anomaly Head (768 -> 512 -> 256 -> 128 -> 1, Sigmoid)
- **Veri Seti:** `data/unified_features` (Toplam: 5409 video)
- **Fold Sayısı:** 5
- **Epoch / Fold:** 20
- **Batch Size:** 64
- **Toplam Süre:** 108.2 saniye

## 1. Genel Performans Özeti (5-Fold Stratified Cross-Validation)

| Metrik | Ortalama (Mean) | Standart Sapma (Std) |
| :--- | :---: | :---: |
| **Doğrulama ROC-AUC** | **0.9684** | ± 0.0030 |
| **Doğrulama PR-AUC (Average Precision)** | **0.9408** | ± 0.0035 |
| **Doğrulama Kaybı (Val Loss)** | **0.3506** | ± 0.0385 |

## 2. Fold Bazlı Detaylar

| Fold | En İyi ROC-AUC | En İyi PR-AUC | En İyi Val Loss | En İyi Epoch |
| :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 0.9679 | 0.9368 | 0.3750 | Epoch 14 |
| Fold 2 | 0.9673 | 0.9409 | 0.3228 | Epoch 9 |
| Fold 3 | 0.9731 | 0.9473 | 0.3405 | Epoch 19 |
| Fold 4 | 0.9695 | 0.9396 | 0.3032 | Epoch 10 |
| Fold 5 | 0.9640 | 0.9397 | 0.4113 | Epoch 18 |
