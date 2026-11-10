# SCM Analytics & Forecasting Platform (Flask)

Platform **Supply Chain Management (SCM)** berbasis Flask yang menyediakan dashboard interaktif untuk memantau data historis, melihat hasil peramalan permintaan, serta menganalisis kebutuhan persediaan dan koordinasi supply chain.

Dashboard dilengkapi visualisasi grafik, ringkasan KPI, filter interaktif, tabel data, fitur pencarian, pengurutan, dan unduhan CSV.

---

## 📊 Tampilan dan Fitur Dashboard

Dashboard memiliki sidebar navigasi yang dapat dibuka dan ditutup menggunakan tombol menu (ikon hamburger ☰). Terdapat tiga menu utama:

### 1. Visualisasi Data Historis

Menu ini menampilkan informasi dan visualisasi data penjualan historis.

Fitur yang tersedia:
- **Ringkasan Data:** Menampilkan gambaran umum data penjualan.
- **Analisis Penjualan:** Visualisasi pola penjualan berdasarkan toko dan produk.
- **Analisis Cuaca dan Musim:** Menampilkan hubungan kondisi cuaca dan musim terhadap penjualan.
- **Analisis Promosi dan Harga:** Menampilkan informasi terkait diskon, promosi, dan harga kompetitor.
- **Analisis Statistik:** Menampilkan hasil pengujian stasioneritas, autokorelasi, dan dekomposisi data.
- **Visualisasi Grafik:** Menyajikan informasi dalam bentuk grafik untuk memudahkan eksplorasi data.

### 2. Forecasting

Menu ini menampilkan hasil peramalan permintaan produk berdasarkan kategori dan toko.

Fitur yang tersedia:

- **Filter Periode:** Memilih tanggal mulai dan tanggal selesai peramalan.
- **Preset Periode:** Pilihan periode cepat seperti 3 bulan, 6 bulan, sepanjang tahun, dan 12 bulan.
- **Filter Store:** Memilih seluruh toko atau toko tertentu, yaitu S001–S005.
- **Filter Kategori:** Memilih seluruh kategori atau kategori tertentu, yaitu Electronics, Clothing, Groceries, Toys, dan Furniture.
- **Ringkasan KPI:** Menampilkan total proyeksi demand, rata-rata demand bulanan, kategori dengan demand tertinggi, dan toko dengan demand tertinggi.
- **Matriks Forecast Demand:** Menampilkan ringkasan proyeksi permintaan berdasarkan kombinasi toko dan kategori.
- **Grafik Tren Forecast:** Menampilkan perkembangan hasil peramalan dalam bentuk grafik garis.
- **Grafik Perbandingan Kategori:** Membandingkan total proyeksi demand antar kategori.
- **Tabel Hasil Forecast:** Menampilkan rincian hasil peramalan berdasarkan periode, toko, kategori, dan produk.
- **Unduh CSV:** Mengunduh hasil peramalan dalam format CSV.

### 3. Inventory Management

Menu ini menampilkan informasi pengelolaan dan optimasi persediaan berdasarkan data toko dan produk.

Fitur yang tersedia:

- **Monitoring Level Persediaan & Titik Reorder:** Grafik untuk memantau stok saat ini, Reorder Point (ROP), dan batas maksimum persediaan.
- **Average EOQ, Average Safety Stock, dan Average ROP:** Ringkasan parameter persediaan dalam bentuk KPI.
- **Daftar Produk Perlu Restock:** Menampilkan produk dengan stok saat ini yang berada di bawah atau sama dengan ROP, diurutkan berdasarkan stok terendah.
- **Detail Perhitungan EOQ, Safety Stock & ROP:** Tabel perhitungan persediaan untuk setiap kombinasi toko dan produk, termasuk stok saat ini, EOQ, Safety Stock, ROP, status persediaan, dan saran pemesanan.
- **Koordinasi Supply Chain:** Menampilkan perbandingan kondisi sebelum dan sesudah koordinasi, termasuk total cost, ukuran lot, serta informasi penghematan buyer, supplier, dan total.
- **Tabel Data Koordinasi:** Menampilkan rincian data koordinasi yang dapat dicari, diurutkan, dan diunduh dalam format CSV.

---

## 💻 Instalasi dan Menjalankan Aplikasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi secara lokal.

### 1. Persiapkan Project

Pastikan Python sudah terinstal di komputer.

Buka terminal atau Command Prompt, kemudian masuk ke direktori project:

```bash
cd "d:\kuliah\semester 7\scm\project"
```

Sesuaikan lokasi direktori dengan tempat project disimpan di komputer masing-masing.

### 2. Instalasi Dependencies

Pastikan file `requirements.txt` tersedia di dalam folder project.

Jalankan perintah berikut:

```bash
pip install -r requirements.txt
```

Tunggu hingga proses instalasi selesai.

### 3. Jalankan Aplikasi

Setelah seluruh dependencies berhasil diinstal, jalankan aplikasi Flask:

```bash
python app.py
```

Pastikan aplikasi berjalan tanpa pesan error pada terminal.

### 4. Buka Dashboard

Buka browser, kemudian akses alamat berikut:

```text
http://127.0.0.1:5000
```

Dashboard SCM Analytics & Forecasting Platform akan ditampilkan dan siap digunakan.

---

## 📁 Struktur File Project

Berikut struktur direktori project berdasarkan susunan folder dan file aplikasi:

```text
project/
│
├── __pycache__/
│
├── models/
│   ├── model_metadata.json
│   └── ridge_forecast_artifacts.joblib
│
├── templates/
│   └── index.html
│
├── app.py
├── forecast_engine.py
├── historical_engine.py
├── inventory_engine.py
├── README.md
├── requirements.txt
├── sales_data.csv
└── train_model.py
```

**Keterangan file utama:**

- `app.py`: File utama untuk menjalankan aplikasi Flask.
- `templates/index.html`: File tampilan dashboard, visualisasi, tabel, dan navigasi menu.
- `forecast_engine.py`: File pendukung proses forecasting.
- `historical_engine.py`: File pendukung pengolahan dan visualisasi data historis.
- `inventory_engine.py`: File pendukung pengolahan dan perhitungan persediaan.
- `train_model.py`: Script pendukung proses pelatihan model.
- `requirements.txt`: Daftar library dan dependencies aplikasi.
- `sales_data.csv`: Dataset penjualan yang digunakan aplikasi.
- `models/`: Folder penyimpanan model dan metadata.
- `__pycache__/`: Folder cache Python.
