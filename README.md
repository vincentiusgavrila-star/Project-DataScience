# Prediksi Harga Laptop

Proyek Akhir Praktikum Data Science 2026/2027 — regresi supervised learning untuk memperkirakan harga laptop berdasarkan spesifikasinya.

## Ringkasan Proyek

Proyek ini mengeksplorasi dan membangun model regresi untuk memprediksi `Final Price` laptop. Dataset berisi 2.160 baris dan 12 kolom, termasuk merek, model, prosesor, RAM, penyimpanan, GPU, ukuran layar, status produk, dan harga.

Tujuan proyek:

- Menilai seberapa baik fitur laptop dapat memperkirakan harga.
- Membandingkan beberapa model regresi.
- Mengukur selisih prediksi dengan R², RMSE, MAE, dan MAPE.

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
2. Membersihkan data: merapikan nama brand, memperbaiki label CPU berdasarkan judul produk, merapikan GPU, menangani `Storage` bernilai 0 sebagai nilai kosong, dan menghapus baris duplikat (spesifikasi dan harga identik dengan SKU berbeda) agar tidak bocor antara data latih dan data uji.
3. Feature engineering: `CPU_Class`, `GPU_Class`, `RAM_Storage_Ratio`, `cpu_year`, `cpu_segment`, `gpu_score`, `is_oled`, dan `high_refresh`.
4. Membagi data menjadi data latih (1.678 baris) dan data uji (420 baris) dengan `test_size=0.2` dan `random_state=42`.
5. Mentransformasikan target harga menggunakan `log1p`. Outlier tidak di-clip karena merupakan laptop kelas atas yang nyata.
6. Preprocessing dalam satu `ColumnTransformer` yang hanya di-fit pada data latih (tanpa data leakage):
   - Imputer median untuk fitur numerik
   - Label/ordinal encoding untuk `Status` dan `CPU_Class`
   - One-hot encoding untuk `Storage type`, `GPU_Class`, `cpu_segment`
   - One-hot encoding dengan penggabungan kategori langka (kurang dari 5 kemunculan) untuk `Brand`, `Model`, `CPU`, `GPU`. Opsi ini dipilih lewat 5-fold CV dibanding target encoding (CV RMSE 0,2047 vs 0,2232).
7. Feature selection dengan `SelectKBest` dan `f_regression`. Nilai k dipilih lewat CV (15, 30, 50, semua); hasilnya semua 160 fitur dipakai.
8. Membandingkan Linear Regression, Ridge, Decision Tree, XGBoost, LightGBM, dan CatBoost dengan 5-fold CV; Ridge, XGBoost, LightGBM, dan CatBoost juga dituning menggunakan `GridSearchCV`.
9. Mengevaluasi model terpilih satu kali pada data uji menggunakan R² dan RMSE pada target log, serta MAE, RMSE, dan MAPE setelah prediksi dikembalikan ke skala harga.

## Hasil

Model terbaik adalah **CatBoost hasil tuning** (`depth=6`, `iterations=800`, `learning_rate=0.05`, `l2_leaf_reg=3`). XGBoost hasil tuning hampir sama (CV RMSE 0,2046 vs 0,2045).

| Metrik                        | Nilai      |
| ----------------------------- | ---------- |
| CV R² (skala log)             | 0,9016     |
| CV RMSE (skala log)           | 0,2045     |
| R² data uji (skala log)       | **0,9120** |
| RMSE data uji (skala log)     | 0,1885     |
| MAE data uji (skala harga)    | ±172 euro  |
| RMSE data uji (skala harga)   | ±269 euro  |
| MAPE data uji (skala harga)   | 14,51%     |
| R² data uji (skala harga)     | 0,9021     |

Seluruh pemilihan (encoding, jumlah fitur, model, dan hyperparameter) dilakukan dengan cross validation pada data latih; data uji hanya dipakai satu kali pada evaluasi akhir.

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
