# PROPOSAL MACHINE LEARNING - DATA SCIENCE

## INFORMASI PROYEK

| Komponen | Keterangan |
| :--- | :--- |
| **Judul Proyek** | Perbandingan Kinerja Linear Regression, XGBoost, dan Long Short-Term Memory (LSTM) dalam Prediksi Jumlah Kasus Demam Berdarah Dengue (DBD) Berdasarkan Faktor Meteorologi di Jakarta Pusat |
| **Kelompok / Kelas** | Kelompok 8 / TRPL 5B |
| **Nama Mahasiswa** | 1. Habib Hajid Taqiuddin (NIM: 244311043)<br>2. Achmad Alvin Al Falah (NIM: 244311031) |
| **Program Studi** | Teknologi Rekayasa Perangkat Lunak (TRPL) |
| **Mata Kuliah** | Data Science |
| **Dosen Pengampu** | Gus Nanang Syaifuddiin, S.Kom., M.Kom. |
| **Tahun Akademik** | 2026/2027, Semester Gasal (Semester 5) |
| **Link Repository GitHub** | https://github.com/AchmadAlvin/dataScience_kelompok_8 |
| **SDG yang Dipilih** | SDG 3: Kehidupan Sehat dan Sejahtera (Good Health and Well-being) |
| **Target SDG Relevan** | Target 3.3: Mengakhiri epidemi AIDS, tuberkulosis, malaria, dan penyakit tropis terabaikan serta memerangi hepatitis, penyakit yang ditularkan melalui air, dan penyakit menular lainnya. |

---

## BAB I. PENDAHULUAN

### 1.1 Latar Belakang

Demam Berdarah Dengue (DBD) merupakan salah satu penyakit menular yang masih menjadi masalah kesehatan masyarakat di Indonesia. Penyakit ini disebabkan oleh virus dengue yang ditularkan melalui gigitan nyamuk Aedes aegypti. Perkembangbiakan nyamuk dan penularan virus sangat dipengaruhi oleh kondisi lingkungan serta cuaca harian. Curah hujan yang tinggi menyebabkan timbulnya genangan air sebagai tempat perindukan jentik, sementara suhu udara dan kelembapan memengaruhi laju metabolisme nyamuk serta masa inkubasi virus di dalam tubuh vektor. Oleh karena itu, data cuaca harian dapat dimanfaatkan untuk menganalisis pola dan memprediksi fluktuasi jumlah kasus DBD.

Penelitian sebelumnya oleh Syahbani dan Sukendra (2020) mengenai peramalan kasus DBD di Kota Magelang menunjukkan bahwa curah hujan memiliki hubungan yang signifikan dengan angka kejadian DBD (p-value = 0,001). Namun, penelitian tersebut hanya menggunakan variabel tunggal curah hujan, sehingga belum mencakup pengaruh faktor cuaca lain seperti suhu dan kelembapan secara bersamaan.

Kajian lain oleh Tarmana (2014) di DKI Jakarta menunjukkan bahwa kenaikan suhu dan perubahan pola hujan berpotensi meningkatkan risiko kejadian DBD. Selain itu, Ichwani dan Wibawa (2019) memodelkan angka kasus DBD di Kecamatan Tembalang menggunakan metode Extreme Learning Machine (ELM) dengan input suhu, kelembapan, dan curah hujan. Hasil kedua studi tersebut mengindikasikan bahwa penggunaan beberapa variabel cuaca sekaligus memberikan informasi prediktif yang lebih baik dibandingkan variabel tunggal.

Meskipun demikian, terdapat beberapa hal yang perlu dikembangkan lebih lanjut. Pertama, sebagian besar studi terdahulu berfokus pada satu metode saja tanpa membandingkan beberapa model secara langsung pada data yang sama. Kedua, belum banyak kajian yang membandingkan performa model statistik dasar (Linear Regression), algoritma ensemble pohon (XGBoost), dan jaringan saraf tiruan untuk deret waktu (LSTM) pada wilayah perkotaan padat seperti Jakarta Pusat. Ketiga, pengaruh cuaca terhadap jumlah kasus DBD memiliki jeda waktu (time lag) antara 1 hingga 4 minggu akibat siklus metamorfosis nyamuk dan masa inkubasi virus. Oleh karena itu, rekayasa fitur jeda waktu perlu diterapkan untuk menguji pengaruhnya terhadap akurasi model.

Berdasarkan permasalahan tersebut, penelitian ini bertujuan untuk membandingkan kinerja tiga model, yaitu Linear Regression, XGBoost, dan Long Short-Term Memory (LSTM), dalam memprediksi jumlah kasus DBD harian di Jakarta Pusat menggunakan data BMKG periode 2021 sampai 2024. Linear Regression digunakan sebagai model dasar (baseline), XGBoost diterapkan untuk menangkap interaksi non-linear antarfaktor cuaca, dan LSTM digunakan untuk memodelkan pola ketergantungan waktu jangka panjang.

Evaluasi performa dilakukan menggunakan metrik Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan koefisien determinasi (R²). Pembagian data latih dan data uji dilakukan secara kronologis untuk mencegah kebocoran data (data leakage). Hasil prediksi ini diharapkan dapat menjadi rujukan sistem kewaspadaan dini bagi dinas kesehatan, serta mendukung pencapaian SDG 3 Target 3.3 dalam menanggulangi penyakit menular.

### 1.2 Rumusan Masalah

1. Bagaimana merancang, melatih, dan mengevaluasi tiga model prediksi (Linear Regression, XGBoost, dan LSTM) pada data deret waktu kasus DBD?
2. Model manakah di antara Linear Regression, XGBoost, dan LSTM yang memberikan performa terbaik berdasarkan metrik MAE, RMSE, dan R² pada data uji kronologis?

### 1.3 Tujuan Penelitian

1. Mengolah dan menyelaraskan dataset harian kasus DBD dengan parameter cuaca BMKG di Jakarta Pusat periode 2021 sampai 2024 serta membentuk fitur jeda waktu (time lag).
2. Membangun dan melatih model Linear Regression, XGBoost, dan LSTM dengan skema pembagian data secara kronologis.
3. Menganalisis pengaruh fitur jeda waktu terhadap peningkatan akurasi estimasi pada ketiga model yang diuji.
4. Membandingkan performa ketiga model menggunakan metrik MAE, RMSE, dan R² untuk menentukan model dengan kesalahan prediksi terendah.
5. Menyusun simpulan hasil pemodelan sebagai bahan pendukung kewaspadaan dini penyebaran DBD yang selaras dengan SDG 3 Target 3.3.

### 1.4 Keterkaitan dengan SDGs dan Pertimbangan Etika

Rincian keterkaitan penelitian ini dengan Sustainable Development Goals (SDG 3 Target 3.3) serta aspek etika pengolahan data disajikan pada Tabel 1.1:

#### Tabel 1.1: Keterkaitan dengan SDGs dan Pertimbangan Etika

| Komponen Evaluasi | Deskripsi Keterkaitan dan Implementasi |
| :--- | :--- |
| **SDG yang Dipilih** | SDG 3: Kehidupan Sehat dan Sejahtera (Good Health and Well-being). |
| **Target SDG Relevan** | Target 3.3: Mengakhiri epidemi AIDS, tuberkulosis, malaria, dan penyakit tropis terabaikan (NTDs) serta penyakit menular lainnya. |
| **Masalah Nyata** | Tingginya potensi penyebaran DBD di permukiman padat Jakarta Pusat serta perlunya perkiraan awal lonjakan kasus berbasis cuaca untuk mendukung kesiapsiagaan dinas kesehatan. |
| **Peran Machine Learning** | Machine learning dan deep learning digunakan untuk menemukan pola non-linear dan ketergantungan waktu antara perubahan cuaca dan siklus hidup nyamuk, guna memproyeksikan kasus pada periode berikutnya. |
| **Indikator Kontribusi** | Tersedianya perbandingan performa model yang terukur (MAE, RMSE, R²) sebagai acuan teknis pengembangan sistem deteksi dini (early warning system). Kontribusi proyek ini bersifat tidak langsung terhadap mitigasi penyakit. |
| **Pertimbangan Etika** | Dataset hanya menggunakan data agregat statistik per wilayah tanpa identitas pasien (bebas PII). Model ini dirancang sebagai alat bantu perencanaan dinas kesehatan, bukan sebagai penentu diagnosis medis individu. |
| **Sumber Ide** | Kekhawatiran terhadap siklus tahunan kenaikan kasus DBD di DKI Jakarta dan ketersediaan data terbuka cuaca BMKG serta data surveilans Dinkes DKI. |

---

## BAB II. LANDASAN KERANGKA KERJA DAN FORMULASI DATASET

### 2.1 Alur Kerangka Kerja ML Lifecycle (L-M-O-P)

Pelaksanaan proyek ini mengikuti empat tahap Machine Learning Lifecycle: tahap Learning Problem (L) berfokus pada perumusan tugas regresi deret waktu multivariat dan pemeriksaan dataset; tahap Model Design (M) menentukan arsitektur tiga model (Linear Regression, XGBoost, dan LSTM); tahap Optimization (O) mencakup pembersihan data hilang, rekayasa fitur jeda waktu (time lag), penyetelan hyperparameter, dan fungsi loss MSE; serta tahap Predict and Evaluate (P) menguji prediksi pada data uji kronologis berbasis MAE, RMSE, dan R².

### 2.2 Formulasi Learning Problem dan Spesifikasi Dataset

Rumusan teknis masalah pembelajaran mesin dan spesifikasi dataset dirangkum pada Tabel 2.1:

#### Tabel 2.1: Formulasi Learning Problem dan Spesifikasi Dataset

| Komponen Masalah | Formulasi Formal dan Spesifikasi Teknis |
| :--- | :--- |
| **Target Prediksi** | Jumlah kasus DBD harian pada periode ke depan (t+k) di Jakarta Pusat. Jenis tugas adalah Supervised Time Series Regression. |
| **Objective Matematis** | Meminimalkan fungsi loss Mean Squared Error (MSE) selama proses pelatihan model. Evaluasi akhir dihitung menggunakan metrik MAE, RMSE, dan R² pada data uji kronologis. |
| **Karakteristik Dataset** | Data sekunder harian gabungan dari surveilans Dinkes DKI Jakarta dan stasiun cuaca BMKG Kemayoran melalui Kaggle (periode 1 Januari 2021 sampai 2024, total 2.923 baris data x 14 variabel). URL: https://www.kaggle.com/datasets/yutakatarokusumah/jakarta-dengue-weather-2021-2024/data. Variabel mencakup Suhu Minimum (TN), Suhu Maksimum (TX), Suhu Rata-rata (TAVG), Kelembapan (RH_AVG), Curah Hujan (RR), Penyinaran Matahari (SS), Kecepatan dan Arah Angin (FF_AVG, DDD_CAR), serta Kasus DBD (CASES). |
| **Skema Partisi Data** | Data dibagi menjadi data latih (80% periode awal) dan data uji (20% periode akhir) berdasarkan urutan waktu tanpa pengacakan (no shuffle) untuk mencegah kebocoran data masa depan (data leakage). |

---

## BAB III. RENCANA PENDEKATAN PEMODELAN DAN EVALUASI

### 3.1 Model Baseline: Linear Regression

Model acuan dasar (baseline) yang digunakan adalah Ordinary Least Squares (OLS) Linear Regression. Model ini dipilih karena bersifat parametrik, sederhana, cepat dilatih, dan koefisien bobotnya mudah diinterpretasikan. Linear Regression berfungsi sebagai tolok ukur awal untuk mengetahui sejauh mana korelasi linear langsung antara parameter cuaca dan angka kasus DBD dapat menjelaskan variansi data target.

Secara matematis, prediksi dihitung melalui kombinasi linear dari variabel input dengan meminimalkan jumlah kuadrat galat (sum of squared residuals). Hasil pengujian model dasar ini akan menjadi pembanding untuk menilai apakah model non-linear dan model sekuensial yang lebih kompleks memberikan peningkatan performa yang berarti.

### 3.2 Model Advanced Machine Learning: XGBoost Regressor

Model machine learning lanjutan yang digunakan adalah XGBoost (Extreme Gradient Boosting) Regressor. XGBoost merupakan algoritma ensemble berbasis pohon keputusan berulang (gradient boosted decision trees) yang terbukti efektif untuk data tabular multivariat. Keunggulan XGBoost adalah kemampuannya memodelkan hubungan non-linear dan interaksi antarfaktor cuaca tanpa memerlukan asumsi kenormalan distribusi data.

XGBoost juga mengintegrasikan fungsi regularisasi L1 (Lasso) dan L2 (Ridge) untuk mengontrol kompleksitas pohon serta meminimalkan risiko overfitting pada data harian. Penyetelan hyperparameter seperti learning rate, kedalaman pohon (max_depth), dan jumlah estimator akan dilakukan untuk memperoleh hasil terbaik.

### 3.3 Model Deep Learning: Long Short-Term Memory (LSTM)

Model deep learning yang diterapkan adalah arsitektur Long Short-Term Memory (LSTM). LSTM merupakan varian Recurrent Neural Network (RNN) yang dirancang khusus untuk menangani data deret waktu panjang tanpa kendala hilangnya gradien (vanishing gradient). Arsitektur ini menggunakan mekanisme gerbang (forget gate, input gate, dan output gate) untuk menyaring informasi mana yang perlu disimpan atau dibuang di sepanjang deret waktu.

Karena penularan DBD memiliki jeda waktu biologis terhadap perubahan cuaca beberapa minggu sebelumnya, input untuk LSTM disusun menggunakan metode jendela geser (sliding window). Dengan cara ini, model dapat mempelajari pola ketergantungan temporal dari riwayat cuaca masa lalu guna memproyeksikan kasus DBD pada hari-hari berikutnya.

### 3.4 Desain Eksperimen dan Metrik Pengujian

Semua model diuji menggunakan protokol eksperimen yang seragam pada partisi data kronologis yang sama tanpa pengacakan data. Evaluasi kuantitatif menggunakan tiga metrik: Mean Absolute Error (MAE) untuk mengukur rata-rata kesalahan absolut, Root Mean Squared Error (RMSE) untuk memberikan bobot penalti lebih besar pada deviasi kesalahan yang tinggi, serta koefisien determinasi (R²) untuk mengukur seberapa baik variansi kasus DBD dapat dijelaskan oleh model.

---

## DAFTAR PUSTAKA

Ichwani, A. S., & Wibawa, H. A. (2019). Prediksi angka kejadian Demam Berdarah Dengue (DBD) berdasarkan faktor cuaca menggunakan metode Extreme Learning Machine (Studi kasus Kecamatan Tembalang). *Jurnal IPTEK*, 23(1), 31-34. https://doi.org/10.31284/j.iptek.2019.v23i1.471

Syahbani, A. N., & Sukendra, D. M. (2020). Peramalan jumlah kasus Demam Berdarah Dengue berdasarkan surveilans kasus dan curah hujan. *HIGEIA Journal of Public Health Research and Development*, 4(1), 84-95. https://doi.org/10.15294/higeia.v4i1/33686

Tarmana, D. (2014). Potensi peluang Demam Berdarah Dengue (DBD) berdasarkan proyeksi perubahan iklim (Studi kasus: DKI Jakarta). *The Indonesian Journal of Infectious Disease*, 1(1), 12-19.
