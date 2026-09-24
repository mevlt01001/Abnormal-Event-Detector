# Operasyonel Eşik Tarama Analiz Raporu (tIoU >= 0.30)

- **Değerlendirilen Eşikler**: 19 nokta (0.05 -> 0.95)
- **Zaman Parametreleri**: `tolerance_sec=2.0s`, `padding_sec=1.0s`
- **★ Önerilen Optimal Eşik**: **`tau = 0.10`** (En Yüksek F1: **%23.14**)
- **Optimal Doğruluk**: Alarm Güvenilirliği (Precision)=**%15.59** | Olay Yakalama (Recall)=**%44.87**

| Eşik (tau) | TP (Doğru Alarm) | FP (Yanlış Alarm) | FN (Kaçırılan) | Precision | Recall | F1-Score | Durum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.05` | 70 | 382 | 86 | %15.49 | %44.87 | %23.03 | Yüksek Yanlış Alarm |
| `0.10` | 70 | 379 | 86 | %15.59 | %44.87 | %23.14 | ★ OPTIMAL (EN İYİ F1) |
| `0.15` | 66 | 385 | 90 | %14.63 | %42.31 | %21.75 |  |
| `0.20` | 59 | 367 | 97 | %13.85 | %37.82 | %20.27 |  |
| `0.25` | 55 | 339 | 101 | %13.96 | %35.26 | %20.00 |  |
| `0.30` | 52 | 332 | 104 | %13.54 | %33.33 | %19.26 | Dengeli Bölge |
| `0.35` | 49 | 306 | 107 | %13.80 | %31.41 | %19.18 | Dengeli Bölge |
| `0.40` | 46 | 286 | 110 | %13.86 | %29.49 | %18.85 | Dengeli Bölge |
| `0.45` | 40 | 272 | 116 | %12.82 | %25.64 | %17.09 | Dengeli Bölge |
| `0.50` | 36 | 250 | 120 | %12.59 | %23.08 | %16.29 |  |
| `0.55` | 28 | 229 | 128 | %10.89 | %17.95 | %13.56 |  |
| `0.60` | 26 | 188 | 130 | %12.15 | %16.67 | %14.05 |  |
| `0.65` | 25 | 164 | 131 | %13.23 | %16.03 | %14.49 |  |
| `0.70` | 21 | 137 | 135 | %13.29 | %13.46 | %13.38 |  |
| `0.75` | 17 | 120 | 139 | %12.41 | %10.90 | %11.60 |  |
| `0.80` | 14 | 90 | 142 | %13.46 | %8.97 | %10.77 |  |
| `0.85` | 9 | 62 | 147 | %12.68 | %5.77 | %7.93 | Aşırı Kaçırma |
| `0.90` | 6 | 38 | 150 | %13.64 | %3.85 | %6.00 | Aşırı Kaçırma |
| `0.95` | 2 | 18 | 154 | %10.00 | %1.28 | %2.27 | Aşırı Kaçırma |

### Kademeli Zamansal IoU (Multi-tIoU) Başarı Özeti (Optimal Eşik tau = 0.10)

| tIoU Eşiği | Değerlendirme Standardı | TP | FP | FN | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tIoU >= 0.10` | Alarm Tetikleme (Point/Hit Level) | 108 | 341 | 48 | %24.05 | %69.23 | %35.70 |
| `tIoU >= 0.20` | Endüstriyel CCTV / S-THUMOS | 88 | 361 | 68 | %19.60 | %56.41 | %29.09 |
| `tIoU >= 0.30` | Operasyonel Referans Standart | 70 | 379 | 86 | %15.59 | %44.87 | %23.14 |
| `tIoU >= 0.50` | Katı TAL / Üst Sınır | 37 | 412 | 119 | %8.24 | %23.72 | %12.23 |