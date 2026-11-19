## INFORMASI PROYEK

<<<<<<< HEAD
**Judul Proyek:**  
Deteksi Dini Risiko Kecemasan Sedang–Berat pada Gamer Online Berdasarkan Pola Bermain dan Faktor Psikologis Menggunakan Pendekatan Multi-Model (Logistic Regression, Random Forest, dan ResNet)

**Kelompok / Kelas:** Kelompok 2 / TRPL 5B  
**Nama Mahasiswa:**  
1. Mohammad Hanif Huda A (NIM: 244311049)  
2. Muhammad Akbar Fadilah (NIM: 244311051)  

**Program Studi:** Teknologi Rekayasa Perangkat Lunak (TRPL)  
**Mata Kuliah:** Data Science  
**Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.  
**Tahun Akademik:** 2026/2027, Semester Gasal (Semester 5)  
**Link GitHub Repository:** https://github.com/akbrvdv/kel2_datascience_uasproject  
**Link Video Pembahasan:** [URL Video YouTube/Drive - Tersedia pada Tahap Final Evaluasi]  
**SDG yang Dipilih:** SDG 3 – Kehidupan Sehat dan Sejahtera (Good Health and Well-being)  
**Target SDG yang Relevan:** Target 3.4 – Mengurangi sepertiga dari kematian dini akibat penyakit tidak menular melalui pencegahan dan pengobatan, serta meningkatkan kesehatan mental dan kesejahteraan.
=======
| Komponen | Keterangan |
| :--- | :--- |
| **Judul Proyek** | Klasifikasi Risiko Kecemasan pada Gamer Online Berdasarkan Pola Bermain dan Faktor Psikologis |
| **Kelompok / Kelas** | Kelompok 2 / TRPL 5B |
| **Nama Mahasiswa** | 1. Mohammad Hanif Huda A (NIM: 244311049)<br>2. Muhammad Akbar Fadilah (NIM: 244311051)<br> |
| **Program Studi** | Rekayasa Perangkat Lunak (TRPL) |
| **Mata Kuliah** | Data Science |
| **Dosen Pengampu** | Gus Nanang Syaifuddiin, S.Kom., M.Kom. |
| **Tahun Akademik** | 2026/2027, Semester Gasal (Semester 5) |
| **Link Repository GitHub** | https://github.com/akbrvdv/kel2_datascience_uasproject |
| **SDG yang Dipilih** | SDG 3: Kehidupan Sehat dan Sejahtera (Good Health and Well-being) |
| **Target SDG Relevan** | Target 3.4: Mengurangi sepertiga dari kematian dini akibat penyakit tidak menular melalui pencegahan dan pengobatan, serta meningkatkan kesehatan mental dan kesejahteraan. |
>>>>>>> f08428de80c3a7668eb3d4bdae63fb854d80de50

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|---|---|---|---|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

**Catatan:** Setiap bagian diberi label **[L]**, **[M]**, **[O]**, atau **[P]**. Keterkaitan antartahap dijelaskan secara transparan, khususnya bagaimana hasil evaluasi **[P]** memberikan umpan balik perbaikan pada tahap formulasi data **[L]**, arsitektur model **[M]**, dan optimasi **[O]** (Bab 7.5).

---

## 1. LEARNING OUTCOMES

Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan *problem statement* (target, objective, data) secara jelas.
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif.
3. **[L]** Melakukan *data preparation* yang sesuai dengan karakteristik dataset tanpa terjadi kebocoran informasi (*data leakage*).
4. **[M]** Merancang tiga model machine learning dengan justifikasi *hypothesis space* dan *inductive bias* (**WAJIB**):
   - Model baseline: **Logistic Regression**
   - Model machine learning / advanced: **Random Forest Classifier**
   - Model deep learning: **ResNet (Residual Neural Network for Tabular Data)**
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, penanganan *class imbalance*, dan regularisasi).
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML (terutama metrik yang sensitif terhadap ketimpangan kelas seperti Macro-F1, Recall, ROC-AUC, dan PR-AUC) dan menganalisis hasilnya secara kritis.
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis.
8. Mengunggah seluruh kode proyek ke GitHub secara terstruktur (**WAJIB**).
9. Menerapkan prinsip *software engineering* dan reproduktibilitas dalam pengembangan proyek data science.

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang

Permainan daring (*online gaming*) telah bertransformasi menjadi salah satu media hiburan harian, interaksi sosial, dan sarana rekreasi paling masif di era digital. Meskipun aktivitas bermain game dapat memberikan stimulasi kognitif dan relaksasi, keterlibatan waktu yang berlebihan dan kompulsi bermain yang tidak terkendali memiliki konsekuensi psikologis yang signifikan. Pada tahun 2018, *World Health Organization* (WHO) secara resmi mengukuhkan *Gaming Disorder* sebagai salah satu kategori gangguan psikiatri dalam *International Classification of Diseases 11th Revision* (ICD-11). Berbagai kajian epidemiologis dan klinis membuktikan bahwa fenomena ketergantungan game online berbanding lurus dengan peningkatan prevalensi stres emosional dan gangguan kecemasan umum (*Generalized Anxiety Disorder*) (Latubessy et al., 2024; Latubessy et al., 2025).

Dalam protokol klinis baku, evaluasi tingkat keparahan kecemasan umum diukur menggunakan kuesioner terstandarisasi *Generalized Anxiety Disorder-7* (GAD-7) dengan rentang skor $0-21$ (Spitzer et al., 2006). Skor GAD-7 $\ge 10$ menetapkan ambang batas klinis (*clinical cut-off*) yang mengindikasikan bahwa responden mengalami risiko kecemasan tingkat sedang hingga berat (*moderate-to-severe anxiety*), sebuah kondisi yang memerlukan evaluasi atau intervensi medis profesional. Kendala utamanya adalah asesmen konvensional melalui kuesioner mandiri yang panjang membutuhkan kesadaran, waktu, dan kepatuhan aktif dari responden. Banyak pemain game cenderung enggan atau tidak menyadari bahwa kebiasaan bermain mereka mencerminkan eskapisme terhadap kecemasan yang mendasari, sehingga intervensi medis kerap terlambat dilakukan. Oleh karena itu, pengembangan instrumen penapisan dini (*early screening tool*) berbasis kecerdasan buatan (*Machine Learning*) yang memanfaatkan data kebiasaan bermain (*gaming patterns*) serta indikator psikologis ringkas menjadi kebutuhan mendesak.

Penelitian komparasi model klasifikasi pada domain kesehatan mental gamer sebelumnya telah dilakukan oleh Karthik & Ashokkumar (2023) yang membandingkan performa *Recurrent Neural Network* (RNN) dengan pohon keputusan *J48* pada data remaja gamer, di mana arsitektur neural terbukti lebih adaptif terhadap pola data yang kompleks. Selanjutnya, Latubessy et al. (2024, 2025) mengembangkan model *Fine-Tuned Attribute Weighted Naïve Bayes* (FTAWNB) pada dataset *Online Gaming Anxiety Data* di Kaggle dan melaporkan capaian akurasi tinggi. Namun, evaluasi kritis menunjukkan adanya risiko kebocoran label (*label leakage*) apabila butir pertanyaan GAD diikutsertakan sebagai fitur input prediksi, serta tingginya sensitivitas model Naïve Bayes terhadap nilai pencilan (*outliers*) pada jam bermain mingguan.

Guna mengatasi celah penelitian tersebut, proyek ini merancang pendekatan deteksi dini yang **bebas dari kebocoran label (*leakage-free*)** dengan mengeluarkan seluruh butir kuesioner GAD (`GAD1`–`GAD7`) dan total skor GAD dari matriks fitur masukan. Model hanya dilatih menggunakan kombinasi atribut pola bermain (jam bermain, platform, preferensi game, gaya bermain), demografi, serta instrumen psikologis eksternal seperti kepuasan hidup (*Satisfaction with Life* / SWL), fobia sosial (*Social Phobia Inventory* / SPIN), dan narsisisme. 

Secara metodologis, penelitian ini membandingkan tiga paradigma pemodelan dengan karakteristik *inductive bias* yang berbeda:
1. **Model Baseline**: **Logistic Regression** sebagai tolok ukur linear parametrik dasar.
2. **Model Advanced ML**: **Random Forest Classifier** sebagai representasi *ensemble* pohon non-linear yang tangguh terhadap pencilan dan interaksi fitur tabular.
3. **Model Deep Learning**: **ResNet (Residual Neural Network for Tabular Data)** untuk menguji efektivitas koneksi lewati (*residual skip connections*) dalam mengekstraksi representasi fitur laten non-linear bertingkat tanpa mengalami degradasi gradien (Gorishniy et al., 2021; He et al., 2016).

Implementasi proyek ini diharapkan dapat menyediakan instrumen penapisan non-invasif yang akurat sekaligus berkontribusi langsung pada agenda pembangunan berkelanjutan **SDG 3 Target 3.4** dalam mempromosikan kesehatan mental generasi muda.

### 2.2 Keterkaitan dengan SDGs (WAJIB)

Ide proyek dirumuskan oleh mahasiswa secara mandiri dan selaras dengan Tujuan Pembangunan Berkelanjutan (Sustainable Development Goals / SDGs) ke-3, khususnya Target 3.4. Rincian keterkaitan masalah, peran model, dan pertimbangan etika disajikan pada Tabel 2.1:

#### Tabel 2.1: Matriks Keterkaitan Proyek dengan Sustainable Development Goals (SDG 3)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|---|---|---|
| **SDG yang Dipilih** | Goal nomor berapa dan apa namanya? | **SDG 3: Kehidupan Sehat dan Sejahtera (*Good Health and Well-being*)** |
| **Target SDG** | Target mana yang paling relevan? | **Target 3.4**: Pada tahun 2030, mengurangi sepertiga dari kematian dini akibat penyakit tidak menular melalui pencegahan dan pengobatan, serta meningkatkan kesehatan mental dan kesejahteraan. |
| **Masalah Nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Tingginya prevalensi kecemasan (*anxiety*) dan isolasi sosial pada komunitas pemain game online (*online gamers*) yang kerap terabaikan karena stigma sosial atau keengganan mengisi instrumen psikologis mandiri yang panjang. Dampak dirasakan langsung oleh remaja dan dewasa muda berupa penurunan kualitas hidup dan produktivitas. |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? | ML dan Deep Learning mampu mengenali interaksi non-linear yang kompleks antara kebiasaan bermain (durasi, genre, gaya bermain), profil demografi, dan indikator psikologis pendukung secara cepat dan non-invasif. Model berfungsi sebagai *triage tool* (penapisan awal) otomatis untuk mendeteksi individu yang berada pada ambang risiko kecemasan sedang–berat. |
| **Indikator Kontribusi** | Bagaimana hasil model berkontribusi pada target SDG? | Menyediakan sistem penapisan dini yang terukur dengan metrik objektif (Macro-F1 $\ge 0,65$, Recall $\ge 0,70$, ROC-AUC $\ge 0,78$) yang dapat diintegrasikan sebagai modul edukasi/peringatan dini (*self-monitoring*) pada platform gaming atau komunitas digital guna mendorong akses konsultasi psikologis lebih awal. |
| **Pertimbangan Etika** | Risiko bias, privasi, atau dampak salah prediksi? | 1. **Bukan Alat Diagnosis Klinis**: Model berstatus murni sebagai penapis awal (*screening*); diagnosis medis resmi mutlak merupakan wewenang psikolog/psikiater berlisensi.<br>2. **Privasi & Kerahasiaan Data**: Menggunakan dataset terbuka yang telah teranonimkan penuh (bebas dari informasi pribadi/PII).<br>3. **Potensi Bias Sampel**: Dataset didominasi responden pria, penggemar *League of Legends*, platform PC, dan wilayah Barat; generalisasi untuk populasi lokal perlu dilakukan secara hati-hati.<br>4. **Integritas Metodologi**: Seluruh pertanyaan GAD (`GAD1`–`GAD7` dan total skor `GAD_T`) dihapus dari set fitur masukan untuk mencegah *data leakage*. |

**Sumber Ide:**  
Ide proyek muncul dari keprihatinan atas krisis kesehatan mental di kalangan generasi muda yang intens beraktivitas di dunia digital dan game daring, pengakuan resmi WHO mengenai *Gaming Disorder* dalam ICD-11, serta tersedianya dataset terbuka *Online Gaming Anxiety Data* di Kaggle yang kaya akan instrumen psikometrik tervalidasi.

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek ini dilaksanakan mengikuti siklus iteratif **L-M-O-P** secara disiplin dan terstruktur:
Pada tahap **Learning Problem [L]**, permasalahan dirumuskan sebagai tugas *supervised binary classification* untuk memprediksi risiko kecemasan sedang–berat (skor GAD-7 $\ge 10$) menggunakan $13.050$ data gamer online yang terbebas dari *label leakage* dan selaras dengan target SDG 3.4. Pada tahap **Model Design [M]**, dirancang tiga arsitektur dengan *hypothesis space* dan *inductive bias* yang kontras: **Logistic Regression** sebagai *baseline* linear parametrik, **Random Forest** sebagai *advanced machine learning* berbasis *ensemble* pohon keputusan non-linear, dan **ResNet (Tabular ResNet)** sebagai model *deep learning* yang mengintegrasikan *residual skip connections*. Pada tahap **Optimization [O]**, seluruh prosedur pra-pemrosesan (penanganan pencilan jam bermain $<84$ jam/minggu, *one-hot encoding*, imputasi median, dan *robust scaling*) diisolasi ke dalam *pipeline* validasi silang (*Stratified 5-Fold CV*), disertai penerapan pembobotan kelas (*class weighting*) untuk mengatasi ketimpangan kelas target ($\approx 17,5\%$ positif), serta optimasi *learning rate* dan *weight decay*. Pada tahap **Predict & Evaluate [P]**, performa akhir diuji pada *hold-out test set* (20%) terpisah menggunakan metrik Macro-F1, Recall kelas risiko tinggi, ROC-AUC, dan PR-AUC, di mana hasil evaluasi dianalisis untuk memberikan umpan balik perbaikan kembali pada tahap **[L]**, **[M]**, atau **[O]** bila terindikasi *generalization gap*.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements

Berdasarkan latar belakang di atas, rumusan masalah dalam proyek ini adalah:
1. **Pencegahan Kebocoran Label (*Label Leakage*)**: Bagaimana merumuskan tugas klasifikasi risiko kecemasan sedang–berat (skor GAD-7 $\ge 10$) pada pemain game online secara objektif tanpa melibatkan satu pun butir pertanyaan diagnosis GAD sebagai fitur masukan?
2. **Penanganan Karakteristik Data Tabular Kompleks**: Bagaimana menangani ketimpangan kelas target (*class imbalance* dengan rasio $\approx 17,5\%$ kelas minoritas positif) serta keberadaan pencilan (*outliers*) pada fitur jam bermain agar model tidak mengalami bias terhadap kelas mayoritas?
3. **Komparasi Paradigma Tiga Model**: Seberapa besar peningkatan performa klasifikasi yang dapat dicapai saat beralih dari model linear sederhana (**Logistic Regression**), ke model ensemble non-linear berbasis pohon (**Random Forest**), hingga ke model *deep learning* dengan koneksi residual (**ResNet**)?
4. **Interpretasi Determinan Risiko Kecemasan**: Fitur apa saja di antara pola kebiasaan bermain, profil demografi, dan skala psikologis eksternal (kepuasan hidup SWL dan fobia sosial SPIN) yang memberikan kontribusi paling signifikan (*feature importance*) terhadap prediksi risiko kecemasan?

### 3.2 Formulasi Learning Problem (WAJIB)

Formulasi teknis permasalahan pembelajaran mesin disajikan pada Tabel 3.1:

#### Tabel 3.1: Formulasi Teknis Learning Problem

| Komponen | Pertanyaan Panduan | Formulasi Proyek Anda |
|---|---|---|
| **Target** | Apa yang diprediksi? Jenis tugas (klasifikasi biner/multikelas, regresi, forecasting, dll.)? | **Status Risiko Kecemasan Sedang–Berat (`risiko_tinggi`)**: Nilai biner $y \in \{0, 1\}$, di mana $y = 1$ jika total skor $\text{GAD-7} \ge 10$ (risiko kecemasan sedang hingga berat), dan $y = 0$ jika total skor $\text{GAD-7} < 10$ (tingkat kecemasan minimal hingga ringan).<br>**Jenis Tugas:** *Supervised Binary Classification*. |
| **Objective** | Apa yang dioptimalkan/diminimalkan secara matematis? Metrik keberhasilan utama? | **Fungsi Objektif Matematis:** Meminimalkan fungsi rugi *Weighted Binary Cross-Entropy* (pada Logistic Regression dan ResNet) serta meminimalkan *Weighted Gini Impurity* (pada Random Forest) dengan penalti kesalahan lebih besar pada kelas minoritas positif.<br>$$\mathcal{L}_{WCE} = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log(\hat{y}_i) + w_0 (1 - y_i) \log(1 - \hat{y}_i) \right]$$<br>**Metrik Keberhasilan Utama:** Macro-F1 Score $\ge 0,65$, Recall Kelas Risiko Tinggi ($y=1$) $\ge 0,70$, dan ROC-AUC $\ge 0,78$ pada data uji (*test set*). |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data? | **Sumber:** Dataset sekunder publik *Online Gaming Anxiety Data* (Kaggle) oleh Divyansh22.<br>**Ukuran:** $13.464$ baris sampel $\times$ $55$ fitur awal, tersaring menjadi $13.050$ responden berstatus `accept == 'Accept'`.<br>**Jenis Data:** Tabular heterogen (fitur numerik kontinu, diskrit, dan kategorikal multikelas).<br>**Asumsi Data:** Sampel diasumsikan independen dan berdistribusi identik (*independent and identically distributed* / i.i.d.) di antara responden berkategori valid. Distribusi kelas target tidak seimbang ($\approx 17,5\%$ positif, $\approx 82,5\%$ negatif). Seluruh fitur diagnosis GAD dieliminasi guna menjamin integritas prediksi (*leakage-free*). Partisi data menggunakan *Stratified Train-Test Split* (80:20). |

### 3.3 Goals

Tujuan spesifik dan terukur dari proyek ini adalah:
1. **Membangun Model Klasifikasi Berkinerja Tinggi**: Mengembangkan sistem klasifikasi biner yang mampu mendeteksi risiko kecemasan sedang–berat dengan capaian metrik pengujian pada *hold-out test set* berupa **Macro-F1 $\ge 0,65$**, **Recall kelas risiko tinggi $\ge 0,70$**, dan **ROC-AUC $\ge 0,78$**.
2. **Mengevaluasi Komparasi Tiga Paradigma Pemodelan**: Mengukur dan membandingkan secara ilmiah performa tiga pendekatan model:
   - **Model 1 (Baseline)**: Logistic Regression dengan pembobotan kelas.
   - **Model 2 (Advanced ML)**: Random Forest Classifier dengan *hyperparameter tuning*.
   - **Model 3 (Deep Learning)**: ResNet (Residual Neural Network for Tabular Data) dengan *batch normalization* dan *skip connections*.
3. **Menganalisis Kontribusi dan Atribusi Fitur**: Mengidentifikasi 5 fitur teratas (*Top-5 features*) yang paling berpengaruh terhadap kecenderungan kecemasan menggunakan analisis *Feature Importance* (Gini importance) dan interpretasi model, guna membuktikan sejauh mana faktor psikologis eksternal mendominasi pola jam bermain murni.
4. **Mendukung Target SDG 3.4 Secara Terukur**: Menghasilkan prototipe penapisan awal (*screening tool*) yang objektif, transparan, dan *reproducible* untuk membantu deteksi dini masalah kesehatan mental pada komunitas gamer sebelum berkembang menjadi gangguan klinis yang lebih parah.

### 3.4 Solution Approach  **[M]**

Untuk menjawab permasalahan di atas, proyek ini mengimplementasikan tiga model pembelajaran mesin dengan rincian arsitektur, *hypothesis space*, dan *inductive bias* sebagai berikut:

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       MATRIKS FITUR MASUKAN (X)                        │
   │   [Pola Bermain: Hours, Game, Platform, whyplay, Playstyle, streams]   │
   │   [Demografi: Age, Gender, Work, Degree]                               │
   │   [Faktor Psikologis: SWL_T (Kepuasan Hidup), SPIN_T (Fobia Sosial)]   │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 ┌──────────────┐             ┌──────────────────┐           ┌──────────────────┐
 │   MODEL 1    │             │     MODEL 2      │           │     MODEL 3      │
 │   Baseline   │             │   Advanced ML    │           │  Deep Learning   │
 │   Logistic   │             │  Random Forest   │           │     ResNet       │
 │  Regression  │             │    Classifier    │           │  (Tabular DL)    │
 └──────┬───────┘             └────────┬─────────┘           └────────┬─────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                       PREDIKSI RISIKO KECEMASAN (y)
                       0: Minimal-Ringan (GAD < 10)
                       1: Sedang-Berat   (GAD >= 10)
```

#### **Model 1 – Baseline Model: Logistic Regression**

* **Deskripsi Model:**  
  Logistic Regression adalah algoritma pembelajaran statistik terawasi parametrik yang memodelkan probabilitas kemunculan kelas target biner menggunakan fungsi logistik (sigmoid):
  $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
* **Alasan Pemilihan:**  
  Dipilih sebagai *baseline* pembanding standar karena kesederhanaannya, efisiensi komputasi yang tinggi, dan transparansi koefisien bobot (*log-odds*) yang memudahkan analisis arah hubungan antara fitur dan risiko kecemasan. Model ini menetapkan batas bawah (*lower bound*) performa linear yang harus dilampaui oleh model yang lebih kompleks.
* **Hypothesis Space & Inductive Bias:**  
  - *Model Family:* Generalized Linear Model (GLM).
  - *Hypothesis Space:* Himpunan semua bidang pemisah linear (*hyperplanes*) berdimensi $d$ dalam ruang fitur: $\mathcal{H}_{LR} = \{ f(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) \mid \mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R} \}$.
  - *Inductive Bias:* Mengasumsikan batas keputusan (*decision boundary*) bersifat linear dan log-odds probabilitas kelas target merupakan kombinasi linear dari fitur masukan. Tidak mengasumsikan interaksi fitur otomatis kecuali ditambahkan secara eksplisit.
  - *Kesesuaian Data:* Berfungsi sebagai tolok ukur cepat, namun diprediksi kurang optimal menangkap interaksi multi-arah antara durasi bermain dan tekanan fobia sosial yang cenderung non-linear.
* **Strategi Optimasi:**  
  Mengoptimalkan fungsi loss *Weighted Binary Cross-Entropy* dengan penalti regularisasi $L_2$ (*Ridge penalty*), solver `lbfgs`, parameter regularisasi $C=1.0$, dan konfigurasi `class_weight='balanced'` untuk mengimbangi ketimpangan proporsi data.

---

#### **Model 2 – Advanced / ML Model: Random Forest Classifier**

* **Deskripsi Model:**  
  Random Forest (Breiman, 2001) adalah algoritma *ensemble learning* berbasis *bagging* (Bootstrap Aggregating) yang membangun kumpulan ratusan pohon keputusan (*decision trees*) secara simultan. Setiap pohon dilatih menggunakan subset sampel yang diambil secara acak dengan pengembalian (*bootstrap samples*), dan pada setiap *node* pemecahan (*split*), algoritma hanya mempertimbangkan subset acak dari fitur masukan (*feature subsampling*). Prediksi final diperoleh melalui agregasi suara terbanyak (*majority voting*):
  $$\hat{y} = \text{mode} \left( \{ T_b(\mathbf{x}) \}_{b=1}^B \right)$$
* **Alasan Pemilihan:**  
  1. Sangat unggul dan mapan dalam mengolah data tabular heterogen yang menggabungkan fitur numerik kontinu (`Hours`, `Age`, `SPIN_T`, `SWL_T`) dan variabel kategorikal nominal/ordinal.
  2. Memiliki ketahanan alami (*robustness*) terhadap pencilan (*outliers*) pada jam bermain mingguan dan tidak sensitif terhadap transformasi monotonik skala fitur.
  3. Mampu menangkap hubungan non-linear kompleks dan interaksi multivariat tanpa membutuhkan rekayasa fitur polinomial manual.
  4. Mereduksi variansi secara dramatis dibandingkan pohon tunggal (*Decision Tree*), sehingga lebih tahan terhadap risiko *overfitting*.
  5. Menyediakan metrik *Feature Importance* (Gini Impurity reduction) bawaan untuk interpretasi determinan kecemasan.
* **Keunggulan & Kelemahan:**  
  - *Keunggulan:* Performa tinggi pada data tabular, tidak memerlukan normalisasi data, tahan terhadap *noise*, dan mampu mengevaluasi *out-of-bag* (OOB) error.
  - *Kelemahan:* Kompleksitas memori dan waktu inferensi meningkat seiring bertambahnya jumlah pohon; interpretasi individual tidak seringkas model linear.
* **Hypothesis Space & Inductive Bias:**  
  - *Model Family:* Non-parametric Decision Tree Ensemble.
  - *Hypothesis Space:* Ruang fungsi *piecewise constant* yang dibentuk oleh partisi ruang fitur menjadi hiper-persegi panjang (*hyperrectangles*) melalui pemisahan ortogonal sumbu fitur: $\mathcal{H}_{RF} = \left\{ \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}) \right\}$.
  - *Inductive Bias:* Mengasumsikan ruang data dapat dipisahkan secara efektif melalui serangkaian aturan ambang batas (*threshold rules*) biner tegak lurus sumbu fitur, serta asumsi bahwa penggabungan pohon-pohon yang *decorrelated* akan menekan komponen variansi dari kesalahan generalisasi.
  - *Perbandingan dengan Baseline:* Mampu memetakan pola ambang kritis non-linear (misal: jam bermain tinggi HANYA berbahaya jika skor kepuasan hidup rendah), pola yang mustahil dipelajari oleh Logistic Regression tanpa interaksi manual.
* **Strategi Optimasi & Kontrol Overfitting:**  
  - Hyperparameter tuning menggunakan *Grid Search* / *Randomized Search* pada parameter kunci: `n_estimators` (100–300), `max_depth` (8–20), `min_samples_split` (4–10), `min_samples_leaf` (2–6), dan `max_features='sqrt'`.
  - Mengatur `class_weight='balanced_subsample'` untuk memastikan setiap pohon menerima bobot penyeimbang kelas pada setiap sampel bootstrap.

---

#### **Model 3 – Deep Learning Model: ResNet (Residual Neural Network for Tabular Data) (WAJIB)**

* **Jenis Deep Learning:**  
  - [x] **Transfer Learning / ResNet Architecture for Tabular Data** *(Adaptasi Deep Residual Learning He et al., 2016 & Gorishniy et al., 2021)*
  - [x] Multilayer Perceptron (MLP) dengan Residual Skip Connections
* **Deskripsi Model:**  
  ResNet untuk data tabular (Tabular ResNet) mengadaptasi prinsip revolusioner *Deep Residual Learning* ke dalam jaringan saraf tiruan data tabular. Berbeda dengan jaringan *feedforward* sekuensial konvensional yang memaksa setiap lapisan mempelajari pemetaan langsung $\mathcal{H}(\mathbf{x})$, arsitektur ResNet menyusun blok residual di mana lapisan mempelajari fungsi pemetaan sisa (*residual mapping*) $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$. Melalui sambungan lewati identitas (*identity shortcut / skip connection*), representasi blok didefinisikan sebagai:
  $$\mathbf{x}_{l+1} = \text{Activation} \left( \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l) \right)$$
* **Alasan Pemilihan:**  
  1. Menguji temuan empiris mutakhir (seperti dalam studi *Revisiting Deep Learning Models for Tabular Data*, Gorishniy et al., NeurIPS 2021) yang membuktikan bahwa arsitektur berbasis ResNet merupakan salah satu arsitektur *deep learning* terbaik untuk data tabular yang mampu menyaingi model ensemble pohon.
  2. Menghilangkan kendala *vanishing gradient* dan degradasi performa pada jaringan neural yang dalam, sehingga memungkinkan eksplorasi representasi fitur laten yang lebih kaya.
  3. Menguji efektivitas representasi *deep learning* terdiferensiasi (*gradient-based differentiable learning*) dalam memetakan interaksi psikologis gamer dibandingkan pendekatan berbasis pohon (*tree-based partition*).
* **Arsitektur Model, Hypothesis Space & Inductive Bias:**  
  - *Model Family:* Deep Residual Neural Network.
  - *Inductive Bias:* Mengasumsikan bahwa fungsi target mendekati fungsi identitas pada beberapa tingkatan representasi dan bahwa memodifikasi representasi secara aditif ($\mathbf{x} + \mathcal{F}(\mathbf{x})$) jauh lebih stabil dan mudah dioptimalkan melalui *gradient descent* daripada memetakan transformasi penuh dari nol. Koneksi identitas memungkinkan gradien mengalir secara langsung tanpa hambatan (*gradient highway*) ke lapisan awal.
  - *Hypothesis Space:* Himpunan fungsi non-linear kontinu bertingkat berdimensi tinggi yang dibentuk oleh komposisi matriks transformasi afinitas dengan aktivasi non-linear dan akumulasi residual aditif: $\mathcal{H}_{ResNet} = \{ f(\mathbf{x}; \Theta) \mid \mathbf{x}_{l+1} = \sigma(\mathbf{x}_l + \mathbf{W}_{l,2} \cdot \text{Dropout}(\sigma(\mathbf{W}_{l,1} \mathbf{x}_l + \mathbf{b}_{l,1})) + \mathbf{b}_{l,2}) \}$.
* **Desain Rinci Arsitektur (Layer Breakdown):**
  1. **Input Projection & Embedding Layer**: 
     - Fitur numerik diskalakan dan diproyeksikan melalui lapisan Dense linear ke dimensi tersembunyi ($d_{\text{model}} = 128$).
     - Fitur kategorikal berdensitas tinggi diproses menggunakan *Entity Embedding* berdimensi rendah ($8–16$).
  2. **Residual Block 1**:
     - Dense Layer ($128 \to 128$) $\to$ Batch Normalization $\to$ ReLU Activation $\to$ Dropout ($p=0.3$) $\to$ Dense Layer ($128 \to 128$) $\to$ Batch Normalization.
     - Penambahan *Skip Connection*: $\mathbf{x}_{\text{out1}} = \text{ReLU}(\mathbf{x}_{\text{in1}} + \mathcal{F}_1(\mathbf{x}_{\text{in1}}))$.
  3. **Residual Block 2**:
     - Dense Layer ($128 \to 128$) $\to$ Batch Normalization $\to$ ReLU Activation $\to$ Dropout ($p=0.3$) $\to$ Dense Layer ($128 \to 128$) $\to$ Batch Normalization.
     - Penambahan *Skip Connection*: $\mathbf{x}_{\text{out2}} = \text{ReLU}(\mathbf{x}_{\text{out1}} + \mathcal{F}_2(\mathbf{x}_{\text{out1}}))$.
  4. **Classification Head**:
     - Dense Layer ($128 \to 64$) $\to$ ReLU $\to$ Dropout ($p=0.2$) $\to$ Dense Layer ($64 \to 1$) $\to$ Sigmoid Activation untuk estimasi probabilitas risiko $P(y=1|\mathbf{x})$.
* **Strategi Optimasi Deep Learning:**  
  - *Optimizer:* AdamW (learning rate awal $\eta = 10^{-3}$, weight decay $= 10^{-4}$).
  - *Loss Function:* *Weighted Binary Cross-Entropy* dengan rasio pembobotan $w_1 = \frac{N_{\text{total}}}{2 \times N_{\text{pos}}}$.
  - *Learning Rate Schedule:* `ReduceLROnPlateau` (faktor $0.5$, kesabaran $3$ epoch).
  - *Epochs & Batch Size:* Training minimal $25$ epoch dengan batch size $64$.
  - *Callbacks:* `EarlyStopping` (patience = 6 epoch pada validation loss) dan `ModelCheckpoint` untuk menyimpan bobot terbaik.

---

## 4. DATA UNDERSTANDING  **[L]**

### 4.1 Informasi Dataset

Dataset yang digunakan dalam penelitian ini merupakan dataset sekunder publik yang dihimpun melalui survei daring berskala internasional pada komunitas pemain game daring (*online gamers*) di platform Reddit.

* **Sumber Dataset:**  
  Platform Kaggle: *Online Gaming Anxiety Data* oleh Divyansh22.  
  **URL:** [https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data](https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data)
* **Deskripsi Spesifikasi Dataset:**
  - **Jumlah Baris (Awal):** $13.464$ baris sampel responden.
  - **Jumlah Baris (Valid):** $13.050$ baris sampel (setelah menerapkan kriteria inklusi verifikasi status `accept == 'Accept'`).
  - **Jumlah Kolom (Fitur Awal):** $55$ kolom variabel.
  - **Tipe Data:** Tabular heterogen (fitur numerik kontinu, diskrit ordinal kuesioner, dan kategorikal nominal/teks bebas).
  - **Ukuran Berkas:** $\approx 4,26$ MB (pada penyimpanan lokal/CSV).
  - **Format Berkas:** Comma-Separated Values (`.csv`).

### 4.2 Deskripsi Fitur

Berikut adalah rincian kamus data (*data dictionary*) untuk fitur-fitur yang dianalisis dalam proyek ini:

| Kelompok Fitur | Nama Fitur | Tipe Data | Deskripsi Fitur | Contoh Nilai |
|---|---|---|---|---|
| **Filter & Kontrol** | `S. No.` | Integer | Nomor urut indeks responden dalam kuesioner survei. | 1, 2, 3 |
| | `accept` | Categorical | Status verifikasi integritas pengisian kuesioner (`Accept` / `NA`). | `Accept` |
| **Pola Bermain** | `Hours` | Float / Continuous | Estimasi rata-rata jam bermain game per minggu yang dilaporkan responden. | 15.0, 20.0, 42.0 |
| | `streams` | Float / Continuous | Jam per minggu yang dihabiskan untuk menonton siaran langsung game (*live streaming*). | 0.0, 2.0, 5.0 |
| | `Game` | Categorical | Judul permainan game daring utama yang paling sering dimainkan. | `League of Legends`, `Counter Strike`, `Starcraft 2` |
| | `Platform` | Categorical | Perangkat keras utama yang digunakan untuk bermain game. | `PC`, `Console (PS, Xbox, ...)`, `Smartphone / Tablet` |
| | `whyplay` | Categorical | Alasan / motivasi primer responden dalam bermain game. | `having fun`, `improving`, `winning`, `relaxing` |
| | `Playstyle` | Categorical | Preferensi gaya bermain dan interaksi sosial dalam game. | `Multiplayer - online - with real life friends`, `Singleplayer` |
| **Demografi** | `Age` | Integer / Discrete | Usia responden dalam satuan tahun (minimal 18 tahun). | 18, 21, 25 |
| | `Gender` | Categorical | Jenis kelamin yang dilaporkan responden. | `Male`, `Female`, `Other` |
| | `Work` | Categorical | Status pekerjaan atau aktivitas kesibukan utama responden. | `Student at college / university`, `Employed`, `Unemployed` |
| | `Degree` | Categorical | Tingkat pendidikan formal terakhir yang telah diselesaikan. | `High school diploma`, `Bachelor (or equivalent)` |
| **Faktor Psikologis** | `SWL_T` | Integer / Discrete | Total skor skala *Satisfaction with Life* (rentang 5–35; mengukur kepuasan hidup global). | 16, 20, 28 |
| | `SPIN_T` | Integer / Discrete | Total skor instrumen *Social Phobia Inventory* (rentang 0–68; mengukur kecemasan sosial). | 11, 19, 42 |
| | `Narcissism` | Integer / Discrete | Skor penilaian mandiri terhadap sifat narsisisme (rentang 1–5). | 1, 2, 4 |
| **Variabel Diagnosis** | `GAD1` s/d `GAD7` | Integer (0–3) | Butir respons individual instrumen GAD-7 (**WAJIB dieksklusi dari fitur**). | 0, 1, 2, 3 |
| | `GAD_T` | Integer / Discrete | Total akumulasi skor instrumen GAD-7 (rentang 0–21). | 2, 8, 14 |
| | `GADE` | Categorical | Tingkat kesulitan fungsional akibat kecemasan (**Dieksklusi dari fitur**). | `Not difficult at all`, `Somewhat difficult` |
| **Target Prediksi** | `risiko_tinggi` | Binary (0 / 1) | **Target Klasifikasi**: 1 jika total skor $\text{GAD-7} \ge 10$ (Risiko Sedang–Berat), 0 jika $< 10$. | 0, 1 |

> **PERINGATAN INTEGRITAS (PENCEGAHAN LABEL LEAKAGE):**  
> Seluruh variabel diagnosa klinis GAD (`GAD1` s/d `GAD7`, total skor `GAD_T`, dan interpretasi keparahan `GADE`) **dikeluarkan sepenuhnya** dari matriks fitur masukan $X$. Model murni dilatih hanya menggunakan pola kebiasaan bermain, demografi, dan skala psikologis eksternal (SWL, SPIN, Narsisisme).

### 4.3 Kondisi Data

Berdasarkan pemeriksaan mendalam terhadap dataset, ditemukan beberapa karakteristik dan anomali data sebagai berikut:

1. **Missing Values (Nilai Hilang):**
   - Kolom `highestleague` memiliki nilai kosong $100\%$ ($13.464$ baris) pada seluruh dataset, sehingga wajib dieliminasi secara permanen.
   - Kolom `League` memiliki $1.797$ nilai hilang ($13,35\%$).
   - Skala psikologis `SPIN_T` memiliki $618$ baris nilai hilang ($4,74\%$) pada subset responden `Accept`. Penanganan dilakukan menggunakan imputasi median terisolasi pada data latih.
   - Fitur `Work` dan `Degree` memiliki sejumlah nilai hilang minor ($<0,5\%$) yang ditangani dengan imputasi modus / kategori `'Unknown'`.
2. **Duplicate Data (Data Duplikat):**
   - Pemeriksaan terhadap baris tanpa kolom nomor indeks (`S. No.`) menunjukkan hanya terdapat 2 baris duplikasi identik yang telah dibersihkan.
3. **Outliers (Pencilan Ekstrem):**
   - Pada fitur `Hours` (jam bermain mingguan), nilai kuartil berada pada $Q_1 = 12$, Median = $20$, dan $Q_3 = 28$ jam/minggu. Namun, nilai maksimum data mentah mencapai **$8.000$ jam/minggu** (secara fisik tidak mungkin karena 1 minggu hanya terdiri dari 168 jam).
   - Terdapat $42$ responden dengan nilai jam bermain $> 84$ jam/minggu ($> 12$ jam per hari secara konsisten). Untuk menjaga validitas representasi, nilai di atas ambang logis $84$ jam/minggu dipangkas (*capped* / difilter) pada tahap pembersihan.
4. **Imbalanced Data (Ketimpangan Kelas Target):**
   - Dari $13.050$ responden valid:
     - **Kelas 0 (Risiko Rendah / GAD-7 < 10):** $10.765$ responden ($82,49\%$).
     - **Kelas 1 (Risiko Tinggi / GAD-7 $\ge$ 10):** $2.285$ responden ($17,51\%$).
   - Rasio ketimpangan kelas tercatat sebesar **$1 : 4,71$**. Kondisi ini mengharuskan penerapan pembobotan kelas (*cost-sensitive class weighting*) dan penggunaan metrik Macro-F1 serta Recall.
5. **Noise & Inkonsistensi Kategori:**
   - Variabel teks bebas seperti `Playstyle` dan `whyplay` memuat puluhan variasi isian manual yang hanya muncul $1$ kali. Nilai-nilai minoritas ini dikelompokkan ke dalam kategori `'Other'`.
6. **Asumsi Data:**
   - Sampel data diasumsikan terdistribusi secara independen dan identik (*independent and identically distributed* / i.i.d.) di antara gamer responden kuesioner. Skema pemisahan data menerapkan *Stratified Split* untuk menjaga rasio distribusi kelas target yang identik antara data latih dan data uji.

### 4.4 Exploratory Data Analysis (EDA)

Analisis eksplorasi data komprehensif dilakukan untuk mengungkap pola tersembunyi antardimensi psikologis dan kebiasaan bermain game. Tiga visualisasi utama disajikan sebagai berikut:

#### Visualisasi 1: Distribusi Target & Analisis Ketimpangan Kelas

![Distribusi Target](reports/figures/01_distribusi_target.svg)

*Gambar 4.1: Distribusi Frekuensi dan Proporsi Kelas Target Risiko Kecemasan (GAD-7 $\ge$ 10).*

**Insight Visualisasi 1:**  
- Proporsi kelas risiko tinggi hanya mencakup $17,51\%$ dari keseluruhan data, menegaskan bahwa mayoritas gamer berada pada spektrum kecemasan normal hingga ringan ($82,49\%$).
- Ketimpangan kelas ini membuktikan bahwa metrik akurasi (*Accuracy*) akan bias terhadap kelas mayoritas. Pengujian model wajib memprioritaskan metrik **Recall Kelas 1** (untuk meminimalkan risiko *false negative* pada gamer berisiko) dan **Macro-F1 Score** (menyeimbangkan ketepatan kedua kelas).

---

#### Visualisasi 2: Distribusi Jam Bermain & Deteksi Outliers

![Pola Jam Bermain Outlier](reports/figures/02_jam_bermain_outlier.svg)

*Gambar 4.2: Pemeriksaan Outlier Jam Bermain Mingguan Sebelum dan Sesudah Pemangkasan Ambang Logis.*

**Insight Visualisasi 2:**  
- Data mentah terdistorsi oleh *troll responses* atau kesalahan input ekstrem (hingga $8.000$ jam/minggu dengan standar deviasi $71,34$).
- Setelah membatasi ambang batas logis maksimal $84$ jam/minggu (ekuivalen $12$ jam/hari), distribusi jam bermain menunjukkan pola kemencengan ke kanan (*right-skewed*) yang wajar dengan nilai tengah (median) $20$ jam/minggu dan rerata $21,28$ jam/minggu. Mayoritas gamer meluangkan waktu antara $12$ hingga $28$ jam setiap minggunya.

---

#### Visualisasi 3: Matriks Korelasi Faktor Psikologis vs Tingkat Kecemasan

![Matriks Korelasi](reports/figures/03_korelasi_psikologis_gad.svg)

*Gambar 4.3: Matriks Korelasi Pearson Antara Fitur Kunci, Skala Psikologis, dan Risiko Kecemasan.*

**Insight Visualisasi 3:**  
- **Korelasi Positif Kuat Fobia Sosial (`SPIN_T`)**: Menunjukkan korelasi linear tertinggi terhadap skor kecemasan ($r = +0,45$) dan target risiko tinggi ($r = +0,35$). Rerata skor fobia sosial pada kelompok berisiko tinggi mencapai $30,22$, hampir dua kali lipat dibanding kelompok risiko rendah ($17,66$).
- **Korelasi Negatif Kuat Kepuasan Hidup (`SWL_T`)**: Menunjukkan korelasi negatif yang sangat signifikan ($r = -0,40$). Rerata kepuasan hidup pada individu berisiko tinggi merosot ke angka $14,89$ (dibandingkan $20,84$ pada kelompok normal).
- **Jam Bermain Bukan Prediktor Tunggal**: Korelasi jam bermain mingguan (`Hours`) terhadap kecemasan tercatat sangat rendah ($r = +0,10$). Selisih rerata jam bermain kelompok cemas ($23,77$ jam/minggu) hanya terpaut $\approx 3$ jam lebih tinggi dibanding kelompok normal ($20,77$ jam/minggu). Hal ini membuktikan bahwa durasi bermain game saja tidak memadai untuk memprediksi gangguan kecemasan tanpa memperhitungkan faktor psikologis yang melatarbelakanginya.

---

## 5. DATA PREPARATION  **[L]**

Tahap pra-pemrosesan data dirancang secara sistematis dengan memastikan tidak adanya kebocoran informasi (*data leakage*) antara data latih dan data uji. Seluruh transformasi inferensial dipelajari murni dari data training (*fit*) dan diaplikasikan secara konsisten pada data evaluasi (*transform*).

### 5.1 Data Cleaning

Langkah pembersihan data meliputi tindakan berikut:
1. **Penerapan Kriteria Inklusi**: Memfilter data survei dan hanya menyertakan responden yang terverifikasi berstatus `accept == 'Accept'` ($13.050$ responden).
2. **Eksklusi Fitur Bocor (*Label Leakage Elimination*)**:
   - Seluruh variabel penentu diagnosis kecemasan klinis (`GAD1` s/d `GAD7`, `GAD_T`, dan `GADE`) dihapus dari matriks fitur.
   - Kolom penanda administratif non-informatif (`S. No.`, `Timestamp`, `accept`, `highestleague`, `League`, `Birthplace`, `Residence`, `Reference`, `Residence_ISO3`, `Birthplace_ISO3`) dieliminasi.
3. **Pembersihan Pencilan (*Outlier Truncation*)**:
   - Baris dengan nilai `Hours > 84` jam/minggu difilter/dipangkas untuk mengeliminasi respons palsu yang mendistorsi variansi data.
4. **Imputasi Nilai Hilang (*Missing Values Imputation*)**:
   - **Fitur Numerik**: Variabel `SPIN_T` dan `Hours` diimputasi menggunakan nilai **median** data training karena distribusinya memiliki kemencengan (*skewed*). Median lebih kokoh (*robust*) terhadap nilai ekstrem dibanding rata-rata (*mean*).
   - **Fitur Kategorikal**: Variabel kategorikal diimputasi menggunakan nilai **modus** (frekuensi terbanyak) pada data training.

### 5.2 Feature Engineering

Untuk meningkatkan daya pembeda (*discriminative power*) model, dilakukan rekayasa fitur sebagai berikut:
1. **Binarisasi Target**: Membentuk label klasifikasi biner $y \in \{0, 1\}$ berdasarkan formula ambang batas klinis:
   $$y = \begin{cases} 1, & \text{jika } \text{GAD\_T} \ge 10 \text{ (Risiko Kecemasan Sedang–Berat)} \\ 0, & \text{jika } \text{GAD\_T} < 10 \text{ (Risiko Kecemasan Minimal–Ringan)} \end{cases}$$
2. **Penyederhanaan Kategori Berfrekuensi Rendah**:
   - Fitur `Game`: Mempertahankan 4 game terpopuler (`League of Legends`, `Counter Strike`, `Starcraft 2`, `World of Warcraft`), sedangkan seluruh game lain dikelompokkan ke dalam kategori `'Other'`.
   - Fitur `Playstyle`: Mengelompokkan isian teks bebas yang beragam ke dalam 4 format utama (`Multiplayer with real life friends`, `Multiplayer with strangers`, `Multiplayer with online teammates`, `Singleplayer`), dan kategori sisanya menjadi `'Other'`.
   - Fitur `whyplay`: Mengonsolidasikan motivasi bermain ke dalam 4 alasan utama (`having fun`, `improving`, `winning`, `relaxing`) dan `'Other'`.
3. **Pembuatan Fitur Rasio Psikologis (`SPIN_SWL_Ratio`)**:
   - Membentuk fitur interaksi rasio antara fobia sosial terhadap kepuasan hidup:
     $$\text{SPIN\_SWL\_Ratio} = \frac{\text{SPIN\_T}}{\text{SWL\_T} + 1}$$
   - Rasio ini memperkuat sinyal kerentanan psikologis gamer (skor fobia sosial tinggi yang disertai kepuasan hidup rendah menghasilkan nilai rasio yang sangat tinggi).

### 5.3 Data Transformation

Transformasi data dilakukan menggunakan `sklearn.compose.ColumnTransformer` secara terisolasi:

1. **Untuk Fitur Kategorikal (Encoding):**
   - Menggunakan **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) untuk fitur nominal: `Platform`, `Gender`, `Work`, `Degree`, `Game_Clean`, `Playstyle_Clean`, dan `whyplay_Clean`.
   - Menghasilkan representasi biner ortogonal yang ramah terhadap model linear (Logistic Regression) dan deep learning (ResNet) tanpa memaksakan urutan ordinal buatan.
2. **Untuk Fitur Numerik (Scaling):**
   - Menggunakan **RobustScaler** (`RobustScaler()`) untuk variabel numerik kontinu: `Hours`, `streams`, `Age`, `SWL_T`, `SPIN_T`, `Narcissism`, dan `SPIN_SWL_Ratio`.
   - Penskalaan ini menggunakan median dan rentang interkuartil ($IQR = Q_3 - Q_1$), sehingga tidak terpengaruh oleh sisa pencilan dan menjaga stabilitas gradien pada pelatihan ResNet dan konvergensi Logistic Regression.

### 5.4 Data Splitting

Pemisahan data dilakukan dengan skema yang ketat dan *reproducible*:
- **Rasio Partisi**: $80\%$ Data Latih ($10.418$ sampel) dan $20\%$ Data Uji *Hold-out* ($2.605$ sampel).
- **Stratified Splitting**: Diterapkan fungsi `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` untuk mempertahankan proporsi kelas target yang identik ($\approx 17,51\%$ positif) di kedua partisi.
- **Validasi Silang Data Latih**: Pada data latih, diterapkan **Stratified 5-Fold Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) untuk optimasi hyperparameter dan pemilihan ambang batas keputusan.
- **Integritas Anti-Leakage**: Data uji (*test set*) dikunci rapat dan tidak pernah digunakan dalam proses `fit` transformator maupun pelatihan model.

```
   DATASET TOTAL (13.023 sampel bersih)
   ├── DATA LATIH (80% / 10.418 sampel) ────────────► Stratified 5-Fold CV (Model Tuning)
   └── DATA UJI HOLD-OUT (20% / 2.605 sampel) ─────► Evaluasi Akhir (Bab 7) [Hanya Disentuh 1x]
```

### 5.5 Data Balancing

Untuk mengatasi ketimpangan kelas ($1 : 4,71$), strategi penyeimbangan diterapkan **HANYA pada data training**:

1. **Cost-Sensitive Class Weighting (Strategi Utama):**
   - Menghitung bobot penalti kesalahan kelas secara terbalik terhadap frekuensi kemunculannya:
     $$w_c = \frac{N_{\text{total}}}{2 \times N_c}$$
   - Didapatkan bobot: **Kelas 0 = $0,606$** dan **Kelas 1 = $2,855$**.
   - Bobot ini diintegrasikan ke parameter `class_weight='balanced'` pada **Logistic Regression** dan **Random Forest**, serta ke fungsi loss *Weighted Binary Cross-Entropy* pada **ResNet**.
2. **Synthetic Minority Over-sampling Technique (SMOTE - Opsional/Eksperimen):**
   - Pada notebook persiapan data, disiapkan pipeline SMOTE yang disintesis **murni pada `X_train`** untuk eksperimen pembanding, memastikan data uji tetap mempertahankan distribusi asli dunia nyata tanpa interpolasi artifisial.

### 5.6 Ringkasan Data Preparation (Apa–Mengapa–Bagaimana)

Ringkasan komprehensif seluruh rangkaian pra-pemrosesan data disajikan pada Tabel 5.1:

#### Tabel 5.1: Matriks Ringkasan Rangkaian Data Preparation

| No | Tahapan | **Apa** yang Dilakukan? | **Mengapa** Penting Dilakukan? | **Bagaimana** Implementasinya? |
|---|---|---|---|---|
| 1 | **Filtering Responden** | Memfilter baris survei hanya dengan nilai `accept == 'Accept'`. | Menghapus respons survei yang tidak valid atau ditolak oleh verifikasi integritas data Kaggle. | `df = df_raw[df_raw['accept'] == 'Accept']` menghasilkan $13.050$ responden. |
| 2 | **Eksklusi Label Leakage** | Menghapus kolom `GAD1`–`GAD7`, `GAD_T`, dan `GADE` dari set fitur. | Mencegah kebocoran target yang dapat menyebabkan model mengalami *overfitting* palsu dan tidak realistis saat inferensi. | `df.drop(columns=['GAD1', ..., 'GAD_T', 'GADE'])` sebelum pembentukan matriks masukan $X$. |
| 3 | **Pembersihan Outliers** | Memangkas batas atas jam bermain pada $Hours \le 84$ jam/minggu. | Mengeliminasi data input palsu/mustahil (seperti $8.000$ jam/minggu) yang mendistorsi skala fitur dan variansi model. | `df = df[df['Hours'] <= 84]` memangkas 42 baris data ekstrem. |
| 4 | **Imputasi Missing Values** | Mengisi nilai kosong pada fitur numerik (`SPIN_T`) dengan median, dan kategorikal dengan modus. | Algoritma ML (terutama Logistic Regression & ResNet) tidak dapat menerima matriks masukan dengan nilai `NaN`. Median dipilih karena resisten terhadap *skewness*. | Menggunakan `SimpleImputer(strategy='median')` dan `SimpleImputer(strategy='most_frequent')` di dalam `Pipeline`. |
| 5 | **Feature Engineering** | Membentuk rasio psikologis `SPIN_SWL_Ratio` dan merampingkan kategori `Game`, `Playstyle`, dan `whyplay`. | Mengurangi dimensi *sparsity* berlebih dari teks bebas serta menangkap interaksi non-linear antara fobia sosial dan ketidakpuasan hidup. | Fungsi `apply()` untuk memetakan kategori ke `'Other'` dan formula rasio $\frac{\text{SPIN\_T}}{\text{SWL\_T} + 1}$. |
| 6 | **Penskalaan Numerik** | Menerapkan penskalaan `RobustScaler` pada seluruh fitur kontinu. | Menyamakan rentang skala antarfitur agar optimizer gradien (ResNet) dan koefisien L2 (Logistic Regression) tidak didominasi fitur berskala besar. | `RobustScaler()` yang menghitung $IQR$ dan median hanya pada `X_train`. |
| 7 | **Encoding Kategorikal** | Mengonversi fitur teks nominal menjadi format biner melalui `OneHotEncoder`. | Mengubah representasi teks kategorikal menjadi vektor numerik tanpa memaksakan relasi ordinalitas semu. | `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` di dalam `ColumnTransformer`. |
| 8 | **Stratified Splitting** | Membagi data menjadi $80\%$ Train dan $20\%$ Test dengan pengacakan berstrata (`random_state=42`). | Menjamin representasi proporsi kelas minoritas ($17,5\%$) seimbang dan identik di kedua set data uji dan latih. | `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`. |
| 9 | **Penanganan Imbalance** | Menghitung bobot penyeimbang kelas (*class weights*) pada data latih. | Mencegah bias model terhadap kelas mayoritas (risiko rendah $82,5\%$) dan mendongkrak metrik Recall kelas risiko tinggi. | `compute_class_weight(class_weight='balanced', ...)` dengan bobot $w_0 \approx 0,61$ dan $w_1 \approx 2,85$. |
| 10 | **Isolasi Pipeline Anti-Leakage** | Mengenkapsulasi transformasi dalam `ColumnTransformer` dan `Pipeline`. | Menjamin bahwa parameter transformasi (`fit`) murni dipelajari dari `X_train` tanpa mencemari data uji `X_test`. | `preprocessor.fit_transform(X_train)` dilanjutkan `preprocessor.transform(X_test)`. |

---

## 6. MODELING  **[M + O]**

> *Akan diimplementasikan pada Checkpoint 2 dan Checkpoint 3 (Minggu 3 s/d 5).*

### 6.1 Model 1 — Baseline Model: Logistic Regression
- Implementasi `sklearn.linear_model.LogisticRegression(class_weight='balanced', max_iter=500)`
- Optimasi solver L-BFGS dengan regularisasi $L_2$.

### 6.2 Model 2 — ML / Advanced Model: Random Forest Classifier
- Implementasi `sklearn.ensemble.RandomForestClassifier(class_weight='balanced_subsample', random_state=42)`
- Penyetelan hyperparameter: `n_estimators`, `max_depth`, `min_samples_split`, dan `max_features` melalui *GridSearchCV*.

### 6.3 Model 3 — Deep Learning Model: ResNet (Tabular ResNet)
- Implementasi arsitektur PyTorch / TensorFlow Keras dengan blok residual aditif, *Batch Normalization*, *Dropout*, dan *Entity Embedding*.
- Konfigurasi pelatihan: Optimizer AdamW, fungsi loss *Weighted Binary Cross-Entropy*, *learning rate schedule*, dan *Early Stopping*.

---

## 7. EVALUATION  **[P]**

> *Akan dieksekusi dan dilaporkan pada Minggu ke-6.*

### 7.1 Metrik Evaluasi
Mengingat distribusi kelas yang tidak seimbang ($\approx 17,5\%$ positif), evaluasi menitikberatkan pada:
- **Macro-F1 Score**: Menilai keseimbangan performa model secara adil pada kedua kelas tanpa bias mayoritas.
- **Recall (Kelas Risiko Tinggi)**: Meminimalkan angka *False Negative* (pasien berisiko tinggi yang terlewat).
- **Precision (Kelas Risiko Tinggi)**: Memastikan intervensi rekomendasi tepat sasaran.
- **ROC-AUC & PR-AUC**: Mengukur kemampuan diskriminasi probabilitas model pada berbagai nilai ambang batas (*threshold*).

### 7.2 Hasil Evaluasi Model (pada Test Set)
*(Tabel hasil uji performa akhir ketiga model akan dicatat setelah eksperimen tuntas).*

### 7.3 Perbandingan Ketiga Model

| Model | Accuracy | Precision | Recall | F1-Score (Macro) | ROC-AUC | Training Time |
|---|---|---|---|---|---|---|
| **Baseline (Logistic Regression)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Advanced (Random Forest)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Deep Learning (ResNet)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 8. CONCLUSION

> *Akan dirumuskan pada laporan akhir berdasarkan temuan empiris eksperimen.*

---

## 9. FUTURE WORK (Opsional)

- [ ] Pengembangan antarmuka web interaktif (*Streamlit / Gradio*) untuk penapisan mandiri secara langsung.
- [ ] Pengumpulan data primer representatif pada populasi gamer lokal di Indonesia.
- [ ] Eksplorasi arsitektur transformer untuk data tabular (misal: *TabTransformer* atau *FT-Transformer*).

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository & Struktur Folder
Repository publik: [https://github.com/akbrvdv/kel2_datascience_uasproject](https://github.com/akbrvdv/kel2_datascience_uasproject)

Struktur direktori dirancang mengikuti siklus *ML Lifecycle*:
```
kel2_datascience_uasproject/
├── data/                                 # [L] Direktori penyimpanan dataset
│   └── GamingStudy_data.csv
├── notebooks/                            # Notebook eksperimen bertahap
│   ├── 01_learning_problem_eda.ipynb     # [L] Eksplorasi data awal
│   ├── 02_data_preparation.ipynb         # [L] Pipeline pembersihan & transformasi
│   ├── 03_baseline_random_forest.ipynb   # [M+O] Pemodelan Logistic Regression & Random Forest
│   └── 04_tabular_resnet_dl.ipynb        # [M+O] Arsitektur & pelatihan ResNet
├── models/                               # [O] Bobot dan artefak model tersimpan
├── reports/figures/                      # [P] Plot kurva ROC, PR, dan loss
├── requirements.txt                      # Daftar dependensi library
└── README.md                             # Laporan komprehensif proyek
```

### 10.2 Environment & Dependencies
```
python>=3.10
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.1.0        # Framework Deep Learning ResNet
torchvision>=0.16.0
```

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

| Minggu | Fokus Lifecycle | Aktivitas Utama | Deliverable / Checkpoint |
|---|---|---|---|
| **1** | **L – Learning Problem** | Perumusan ide SDG 3.4, pemilihan dataset Kaggle, studi literatur, formulasi problem statement & learning problem (Bab 1–3) | Formulir ide + proposal singkat di README. **Checkpoint 1** |
| **2** | **L – Data** | Data understanding, EDA, pembersihan pencilan, encoding, dan partisi data (Bab 4–5) | Notebook data preparation + update repo |
| **3** | **M – Model Design** | Rancang & latih Model 1 (Logistic Regression) dan Model 2 (Random Forest) (Bab 6.1–6.2) | Hasil baseline & advanced pada validation set. **Checkpoint 2** |
| **4** | **M → O** | Tuning Random Forest; rancang arsitektur Model 3 (ResNet) dan training awal (Bab 6.2–6.3) | Model DL berjalan end-to-end |
| **5** | **O – Optimization** | Optimasi ResNet (lr schedule, loss bobot, regularisasi dropout/batchnorm, konvergensi loss) | Model DL terlatih + kurva loss/metrik. **Checkpoint 3** |
| **6** | **P – Predict & Evaluate** | Evaluasi test set, komparasi 3 model, analisis error, perbaikan, video presentasi, finalisasi repo | Laporan final + video + presentasi |

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|---|---|---|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur, training $\ge 10$ epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

---

## DAFTAR PUSTAKA

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Divyansh22. (n.d.). *Online gaming anxiety data* [Data set]. Kaggle. https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems (NeurIPS 2021)*, 34, 18932–18943.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778).

Karthik, M. A., & Ashokkumar, S. (2023). Predicting the link between online gaming and mental disorders in youth: RNN vs J48 decision tree. *Proceedings of the International Conference on Health Informatics*, 403–408. SciTePress. https://doi.org/10.5220/0012504900003739

Latubessy, A., Wardoyo, R., Musdholifah, A., & Kusrohmaniah, S. (2024). Fine tuning attribute weighted naïve Bayes model for detecting anxiety disorder levels of online gamers. *International Journal of Electrical and Computer Engineering*, 14(3), 3277–3286. https://doi.org/10.11591/ijece.v14i3.pp3277-3286

Latubessy, A., Wardoyo, R., Musdholifah, A., & Kusrohmaniah, S. (2025). Fine-tuned attribute weighted naïve Bayes with modified partial instances reduction for gaming disorder classification. *HighTech and Innovation Journal*, 6(1), 1–14. https://doi.org/10.28991/HIJ-2025-06-01-05

Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006). A brief measure for assessing generalized anxiety disorder: The GAD-7. *Archives of Internal Medicine*, 166(10), 1092–1097. https://doi.org/10.1001/archinte.166.10.1092

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

| SDG | Nama (Indonesia) | Contoh Arah Ide ML (Inspirasi) |
|---|---|---|
| 1 | Tanpa Kemiskinan | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular) |
| 2 | Tanpa Kelaparan | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular) |
| **3** | **Kehidupan Sehat dan Sejahtera** | **Deteksi dini risiko kecemasan pada gamer online (tabular); prediksi risiko klinis** |
| 4 | Pendidikan Berkualitas | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran |
| 5 | Kesetaraan Gender | Deteksi ujaran kebencian atau bias gender pada teks |
| 6 | Air Bersih dan Sanitasi Layak | Prediksi kelayakan/kualitas air minum (tabular) |
| 7 | Energi Bersih dan Terjangkau | Forecasting konsumsi atau produksi energi (time series) |
| 8 | Pekerjaan Layak dan Pertumbuhan Ekonomi | Klasifikasi lowongan kerja atau prediksi pengangguran |
| 9 | Industri, Inovasi, dan Infrastruktur | Deteksi kerusakan jalan (image); predictive maintenance |
| 10 | Berkurangnya Kesenjangan | Pengenalan bahasa isyarat (image); analisis akses layanan publik |
| 11 | Kota dan Permukiman Berkelanjutan | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas |
| 12 | Konsumsi & Produksi Bertanggung Jawab | Klasifikasi jenis sampah (image); prediksi limbah makanan |
| 13 | Penanganan Perubahan Iklim | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan |
| 14 | Ekosistem Lautan | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang |
| 15 | Ekosistem Daratan | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra |
| 16 | Perdamaian, Keadilan, Kelembagaan | Deteksi hoaks/berita palsu (teks); deteksi fraud transaksi |
| 17 | Kemitraan untuk Mencapai Tujuan | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|---|---|
| **Nama / NIM** | 1. Mohammad Hanif Huda A (NIM: 244311049)<br>2. Muhammad Akbar Fadilah (NIM: 244311051) |
| **Kelas / Kelompok** | TRPL 5B / Kelompok 2 |
| **Judul Ide** | Deteksi Dini Risiko Kecemasan Sedang–Berat pada Gamer Online Berdasarkan Pola Bermain dan Faktor Psikologis Menggunakan Pendekatan Multi-Model (Logistic Regression, Random Forest, dan ResNet) |
| **SDG & Target yang Dipilih** | SDG 3 (Kehidupan Sehat dan Sejahtera), Target 3.4 (Pencegahan penyakit dan peningkatan kesehatan mental) |
| **Masalah Nyata yang Diangkat** | Peningkatan prevalensi kecemasan dan stres pada pemain game online yang sulit dideteksi sejak dini karena keengganan mengisi instrumen kuesioner psikologis panjang, sehingga membutuhkan alat penapis mandiri (*triage*) non-invasif. |
| **Tugas ML (Target & Jenis Tugas)** | Supervised Binary Classification: Memprediksi `risiko_tinggi` ($1 = \text{GAD-7} \ge 10$, $0 = \text{GAD-7} < 10$) |
| **Objective & Metrik Utama** | Meminimalkan *Weighted Binary Cross-Entropy* dengan target evaluasi Macro-F1 $\ge 0,65$, Recall kelas positif $\ge 0,70$, dan ROC-AUC $\ge 0,78$. |
| **Dataset (Sumber, URL, Ukuran, Jenis Data)** | Kaggle *Online Gaming Anxiety Data* (Divyansh22), URL: https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data, $13.050$ baris berstatus `Accept`, $55$ kolom awal, data tabular campuran. |
| **Rencana Tiga Model** | 1. **Baseline**: Logistic Regression (Linear Model)<br>2. **Advanced ML**: Random Forest Classifier (Tree-based Ensemble)<br>3. **Deep Learning**: ResNet for Tabular Data (Residual Neural Network dengan Skip Connections) |
| **Risiko & Pertimbangan Etika** | Model hanya sebagai alat penapisan awal (bukan diagnosis klinis definitif). Data telah teranonimkan penuh (bebas PII). Seluruh pertanyaan diagnosis GAD1–GAD7 dan total skor dieksklusikan dari fitur input untuk menjamin tidak adanya kebocoran label (*leakage-free*). |
| **Referensi Awal (Minimal 2)** | 1. Spitzer et al. (2006) - *A brief measure for assessing generalized anxiety disorder: The GAD-7*.<br>2. Latubessy et al. (2024) - *Fine tuning attribute weighted naïve Bayes model for detecting anxiety disorder levels of online gamers*.<br>3. Gorishniy et al. (2021) - *Revisiting deep learning models for tabular data (ResNet on tabular data)*. |

**Kriteria Persetujuan Dosen:**
- [x] Ide berasal dari mahasiswa dan tidak duplikat
- [x] Selaras jelas dengan satu SDG dan targetnya (SDG 3 Target 3.4)
- [x] Dapat dirumuskan sebagai masalah ML dengan target yang jelas (Supervised Binary Classification)
- [x] Dataset tersedia, legal/etis digunakan, dan cukup besar ($13.050$ baris)
- [x] Layak untuk tiga model termasuk deep learning dalam 6 minggu
