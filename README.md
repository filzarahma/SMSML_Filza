# Proyek Prediksi Penyakit Jantung

Proyek ini mengimplementasikan sistem pembelajaran mesin untuk prediksi penyakit jantung dengan pemantauan komprehensif, pencatatan log, dan integrasi CI/CD.

## Komponen Proyek

### 1. Pembangunan Model

Model prediksi penyakit jantung dibangun menggunakan:
- RandomForestClassifier dari scikit-learn
- Penyesuaian hyperparameter untuk menemukan konfigurasi optimal
- MLflow untuk pelacakan eksperimen dan manajemen model

File utama:
- `modelling.py`: Implementasi model dasar dengan pelacakan MLflow
- `modelling_tuning.py`: Model lanjutan dengan penyesuaian hyperparameter dan metrik komprehensif

Signature model mencakup parameter kesehatan penting:
- Age (Umur)
- Sex (Jenis Kelamin)
- ChestPainType (Tipe Nyeri Dada)
- RestingBP (Tekanan Darah Istirahat)
- Cholesterol (Kolesterol)
- FastingBS (Gula Darah Puasa)
- RestingECG (EKG Istirahat)
- MaxHR (Detak Jantung Maksimal)
- ExerciseAngina (Angina saat Beraktivitas)
- Oldpeak (Depresi ST)
- ST_Slope (Kemiringan ST)

### 2. Pemantauan dan Pencatatan Log

Sistem ini mencakup solusi pemantauan komprehensif:

- **Exporter Prometheus** (`prometheus_exporter.py`): 
  - Melacak permintaan API, metrik sistem, dan performa
  - Mencakup metrik khusus untuk penggunaan CPU, RAM, jaringan, dan waktu respons
  - Mengekspos endpoint metrik untuk integrasi Prometheus

- **API Inferensi** (`inference.py`):
  - Antarmuka yang ramah pengguna untuk prediksi penyakit jantung
  - Menangani prapemrosesan data dan interaksi model
  - Memberikan hasil prediksi yang jelas dengan tingkat kepercayaan

- **Pengujian Pemantauan** (`test_monitoring.py`):
  - Memvalidasi fungsionalitas API
  - Merekam waktu respons dan akurasi prediksi
  - Mencatat interaksi untuk pemecahan masalah

### 3. Integrasi Eksternal

Proyek terhubung dengan:

- **Server Pelacakan MLflow**:
  - Endpoint lokal: http://127.0.0.1:5000/
  - Endpoint DagsHub: https://dagshub.com/filzarahma/heart-disease-prediction.mlflow

- **Penyediaan Model**:
  - Endpoint: http://127.0.0.1:5005/invocations (Server model MLflow)
  - API Prediksi: http://127.0.0.1:8000/predict (API Aplikasi)

- **Repositori GitHub**:
  - [Repositori Eksperimen](https://github.com/filzarahma/Eksperimen_SML_Filza.git)
  - [Repositori Workflow CI](https://github.com/filzarahma/Workflow-CI.git)

## Memulai

1. Instal dependensi yang diperlukan
2. Mulai server pelacakan MLflow
3. Latih dan daftarkan model
4. Deploy model dengan MLflow serve
5. Mulai exporter Prometheus
6. Jalankan inferensi untuk prediksi

## Contoh Penggunaan

```python
# Gunakan script inference.py untuk membuat prediksi
python "Monitoring dan logging/7. inference.py"

# Pantau metrik dengan Prometheus
curl http://127.0.0.1:8000/metrics
```

## Struktur Proyek

```
SMSML_Filza-Rahma-Muflihah/
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   └── mlartifacts/ (artefak model)
├── Monitoring dan logging/
│   ├── 3. prometheus_exporter.py
│   ├── 7. inference.py  
│   └── 8. Lainnya/
│       └── test_monitoring.py
└── README.md
```