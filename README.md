# Prediksi Harga Laptop

Proyek Akhir Praktikum Data Science 2026/2027 — regresi supervised learning untuk memperkirakan harga laptop berdasarkan spesifikasinya.

## Ringkasan Proyek

Proyek ini mengeksplorasi dan membangun model regresi untuk memprediksi `Final Price` laptop. Dataset berisi 2.160 baris dan 12 kolom, termasuk merek, model, prosesor, RAM, penyimpanan, GPU, ukuran layar, status produk, dan harga.

Tujuan proyek:

- Menilai seberapa baik fitur laptop dapat memperkirakan harga.
- Membandingkan beberapa model regresi.
- Mengukur selisih prediksi dengan R², RMSE, dan MAE.

## Dataset dan Sumber

- **Sumber:** [Laptops Price Dataset — Kaggle](https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset)
- **Pemilik dataset di Kaggle:** Juan Merino Bermejo (`juanmerinobermejo`)
- **File yang digunakan:** `laptops.csv`
- **Target:** `Final Price` (harga dalam euro sebagaimana dilaporkan notebook)
- **Lisensi:** _Periksa dan tuliskan lisensi yang tercantum pada halaman Kaggle sebelum repository dikumpulkan._

Kolom utama dataset:

| Kolom          | Keterangan                                    |
| -------------- | --------------------------------------------- |
| `Laptop`       | Nama/deskripsi produk laptop                  |
| `Status`       | Status produk, misalnya baru atau refurbished |
| `Brand`        | Merek laptop                                  |
| `Model`        | Model atau keluarga produk                    |
| `CPU`          | Informasi prosesor                            |
| `RAM`          | Kapasitas RAM                                 |
| `Storage`      | Kapasitas penyimpanan                         |
| `Storage type` | Jenis penyimpanan                             |
| `GPU`          | Informasi kartu grafis                        |
| `Screen`       | Ukuran layar                                  |
| `Touch`        | Informasi layar sentuh                        |
| `Final Price`  | Harga laptop, menjadi target prediksi         |

## Isi Repository

```text
.
├── README.md
├── Project_Supervised.ipynb
└── laptops.csv
```

## Metode

Notebook menjalankan tahapan berikut:

1. Memeriksa struktur, tipe data, nilai kosong, statistik deskriptif, dan distribusi data.
2. Membersihkan dan memproses fitur, termasuk membuat kategori CPU dan GPU serta fitur rasio RAM terhadap storage.
3. Membagi data menjadi data latih dan data uji dengan `test_size=0.2` dan `random_state=42`.
4. Mentransformasikan target harga menggunakan `log1p`.
5. Memilih 15 fitur dengan `SelectKBest` dan `f_regression`.
6. Membandingkan Linear Regression, Ridge, Decision Tree, XGBoost, LightGBM, dan CatBoost; beberapa model juga dituning menggunakan `GridSearchCV`.
7. Mengevaluasi model menggunakan R² dan RMSE pada target log, serta MAE setelah prediksi dikembalikan ke skala harga.

Hasil notebook terbaru mencatat CatBoost hasil tuning dengan R² test **0,8628**, RMSE test **0,2445** pada skala log, dan MAE sekitar **249 euro**. Angka ini merupakan hasil eksperimen notebook saat ini; gunakan dengan hati-hati karena data uji dipakai berulang kali selama perbandingan model.

## Informasi Anggota dan Kontribusi

Lengkapi tabel berikut dengan data kelompok yang sebenarnya.

| No. | Nama                           | NIM       | Kontribusi                              |
| --: | ------------------------------ | --------- | --------------------------------------- |
|   1 | Tanti Eka Lestari              | 123240059 | EDA                                     |
|   2 | Vincentius Gavrila Subagyo     | 123240146 | Modeling                                |
|   3 | Nicholaus Herjuna Dianaji      | 123240152 | Data Cleaning                           |
|   4 | Muhammad Jiddan Najwan Junaidi | 123240178 | Preprocessing                           |

## Cara Menjalankan

Pastikan `Project_Supervised.ipynb` dan `laptops.csv` berada di folder yang sama. Gunakan Python 3.10 atau lebih baru.

1. Buka terminal pada folder repository.
2. Buat dan aktifkan virtual environment (opsional, tetapi disarankan):

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang library yang digunakan:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install numpy pandas matplotlib seaborn "scikit-learn>=1.2" xgboost lightgbm catboost jupyter
   ```

4. Jalankan Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Buka `Project_Supervised.ipynb` dan jalankan sel secara berurutan dari atas. Notebook membaca dataset menggunakan nama file `laptops.csv`.

Karena `laptops.csv` disertakan di repository, proses menjalankan notebook tidak memerlukan Kaggle API. Jika dataset diperbarui, pastikan sumber, versi/tanggal pengambilan, dan lisensinya tetap dicatat.
