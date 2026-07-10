# Eksperimen SML — Heart Disease Prediction

**Nama:** Trinata Suryawan

Repository ini merupakan **Kriteria 1** proyek akhir kelas *Membangun Sistem
Machine Learning* (Dicoding): eksperimen dan otomatisasi preprocessing dataset.

## Dataset
**Heart Disease** — klasifikasi biner untuk memprediksi keberadaan penyakit
jantung berdasarkan 11 fitur klinis (`HeartDisease`: 1 = sakit, 0 = sehat).

## Struktur
```
Eksperimen_SML_Trinata-Suryawan
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions (Advance)
└── preprocessing/
    ├── Eksperimen_Trinata-Suryawan.ipynb   # Notebook eksperimen (loading, EDA, preprocessing)
    ├── automate_Trinata-Suryawan.py        # Otomatisasi preprocessing (Skilled)
    ├── heart_raw.csv                        # Dataset mentah
    ├── requirements.txt
    └── namadataset_preprocessing/
        ├── train.csv                        # Data siap latih
        └── test.csv
```

## Menjalankan Preprocessing Otomatis
```bash
cd preprocessing
pip install -r requirements.txt
python automate_Trinata-Suryawan.py --input heart_raw.csv --output-dir namadataset_preprocessing
```

## Otomatisasi (GitHub Actions)
Workflow `preprocessing.yml` menjalankan `automate_Trinata-Suryawan.py` secara
otomatis setiap kali ada push ke `main`, dijadwalkan harian, atau dipicu manual,
lalu menyimpan kembali dataset hasil preprocessing ke repository.
