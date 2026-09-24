# Operasyonel Eşik Tarama Analiz Raporu (tIoU >= 0.30)

- **Değerlendirilen Eşikler**: 19 nokta (0.05 -> 0.95)
- **Zaman Parametreleri**: `tolerance_sec=2.0s`, `padding_sec=1.0s`
- **★ Önerilen Optimal Eşik**: **`tau = 0.05`** (En Yüksek F1: **%25.75**)
- **Optimal Doğruluk**: Alarm Güvenilirliği (Precision)=**%17.76** | Olay Yakalama (Recall)=**%46.79**

| Eşik (tau) | TP (Doğru Alarm) | FP (Yanlış Alarm) | FN (Kaçırılan) | Precision | Recall | F1-Score | Durum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.05` | 73 | 338 | 83 | %17.76 | %46.79 | %25.75 | ★ OPTIMAL (EN İYİ F1) |
| `0.10` | 65 | 340 | 91 | %16.05 | %41.67 | %23.17 | Yüksek Yanlış Alarm |
| `0.15` | 62 | 320 | 94 | %16.23 | %39.74 | %23.05 |  |
| `0.20` | 55 | 316 | 101 | %14.82 | %35.26 | %20.87 |  |
| `0.25` | 59 | 293 | 97 | %16.76 | %37.82 | %23.23 |  |
| `0.30` | 57 | 275 | 99 | %17.17 | %36.54 | %23.36 | Dengeli Bölge |
| `0.35` | 55 | 248 | 101 | %18.15 | %35.26 | %23.97 | Dengeli Bölge |
| `0.40` | 54 | 234 | 102 | %18.75 | %34.62 | %24.32 | Dengeli Bölge |
| `0.45` | 51 | 227 | 105 | %18.35 | %32.69 | %23.50 | Dengeli Bölge |
| `0.50` | 46 | 220 | 110 | %17.29 | %29.49 | %21.80 |  |
| `0.55` | 42 | 207 | 114 | %16.87 | %26.92 | %20.74 |  |
| `0.60` | 40 | 181 | 116 | %18.10 | %25.64 | %21.22 |  |
| `0.65` | 36 | 173 | 120 | %17.22 | %23.08 | %19.73 |  |
| `0.70` | 32 | 147 | 124 | %17.88 | %20.51 | %19.10 |  |
| `0.75` | 27 | 136 | 129 | %16.56 | %17.31 | %16.93 |  |
| `0.80` | 25 | 122 | 131 | %17.01 | %16.03 | %16.50 |  |
| `0.85` | 21 | 96 | 135 | %17.95 | %13.46 | %15.38 | Aşırı Kaçırma |
| `0.90` | 14 | 73 | 142 | %16.09 | %8.97 | %11.52 | Aşırı Kaçırma |
| `0.95` | 8 | 48 | 148 | %14.29 | %5.13 | %7.55 | Aşırı Kaçırma |

### Kademeli Zamansal IoU (Multi-tIoU) Başarı Özeti (Optimal Eşik tau = 0.05)

| tIoU Eşiği | Değerlendirme Standardı | TP | FP | FN | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tIoU >= 0.10` | Alarm Tetikleme (Point/Hit Level) | 112 | 299 | 44 | %27.25 | %71.79 | %39.51 |
| `tIoU >= 0.20` | Endüstriyel CCTV / S-THUMOS | 86 | 325 | 70 | %20.92 | %55.13 | %30.34 |
| `tIoU >= 0.30` | Operasyonel Referans Standart | 73 | 338 | 83 | %17.76 | %46.79 | %25.75 |
| `tIoU >= 0.50` | Katı TAL / Üst Sınır | 31 | 380 | 125 | %7.54 | %19.87 | %10.93 |