## INFORMASI PROYEK

**Judul Proyek:** Deteksi Dini Risiko Kecemasan Sedang–Berat pada Gamer Online Berdasarkan Pola Bermain dan Faktor Psikologis Menggunakan Pendekatan Multi-Model (Logistic Regression, Random Forest, dan ResNet)

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

Bab ini mendokumentasikan pemahaman awal atas data sebelum pemodelan. Eksplorasi yang dapat dijalankan ulang tersedia pada [notebooks/01_learning_problem_eda.ipynb](notebooks/01_learning_problem_eda.ipynb).

### 4.1 Informasi Dataset

Dataset yang digunakan adalah *Online Gaming Anxiety Data* dari Kaggle (Divyansh22). Dataset ini merupakan hasil survei terbuka yang telah dianonimkan, sehingga unit analisisnya adalah seorang responden gamer online, bukan rekaman aktivitas permainan secara langsung.

| Aspek | Keterangan |
|---|---|
| Berkas sumber | `data/GamingStudy_data.csv` |
| Format dan encoding | CSV, dibaca dengan encoding `latin1` |
| Ukuran awal | 13.464 baris x 55 kolom; sekitar 8,26 MB di memori |
| Kriteria inklusi | Hanya responden dengan `accept == 'Accept'` |
| Sampel valid hasil inklusi | 13.050 responden |
| Sampel setelah aturan kualitas `Hours <= 84` | 12.981 responden |
| Jenis data | Tabular campuran: numerik diskret/kontinu serta kategorikal nominal |
| Target turunan | `risiko_tinggi = 1` bila `GAD_T >= 10`; selain itu `0` |

Penyaringan status persetujuan mengeluarkan 414 baris yang tidak memiliki nilai `accept`. Tahap berikutnya mengeluarkan 42 jawaban dengan `Hours > 84` serta baris dengan nilai `Hours` kosong melalui aturan filter yang sama. Batas 84 jam per minggu dipilih sebagai batas kewajaran (rata-rata maksimum 12 jam per hari), bukan sebagai batas klinis.

### 4.2 Deskripsi Fitur

Tabel berikut merangkum kelompok fitur pada data mentah. Kolom yang digunakan sebagai masukan model ditetapkan kembali pada Bab 5 agar pemilihan fitur dan pencegahan *leakage* dapat ditelusuri dengan jelas.

| Kelompok | Kolom utama | Tipe | Peran dalam proyek |
|---|---|---|---|
| Identitas dan administrasi | `S. No.`, `Timestamp`, `accept`, `Reference` | Numerik/kategorikal | Identitas, waktu pengisian, status persetujuan, dan sumber rujukan; tidak menjadi fitur model. |
| Kecemasan GAD-7 | `GAD1`–`GAD7`, `GADE`, `GAD_T` | Numerik/kategorikal | `GAD_T` membentuk target; seluruh kelompok ini dikeluarkan dari masukan karena dapat membocorkan label. |
| Kepuasan hidup | `SWL1`–`SWL5`, `SWL_T` | Numerik | Indikator kepuasan hidup; `SWL_T` dipakai sebagai indikator psikologis agregat. |
| Fobia sosial | `SPIN1`–`SPIN17`, `SPIN_T` | Numerik | Indikator fobia sosial; `SPIN_T` dipakai sebagai indikator psikologis agregat. |
| Pola bermain | `Game`, `Platform`, `Hours`, `earnings`, `whyplay`, `League`, `highestleague`, `streams`, `Playstyle` | Numerik/kategorikal | Menggambarkan kebiasaan dan preferensi bermain; kategori relevan diringkas pada Bab 5. |
| Demografi | `Gender`, `Age`, `Work`, `Degree` | Numerik/kategorikal | Variabel karakteristik responden yang dapat mendukung pola risiko. |
| Lokasi | `Birthplace`, `Residence`, `Birthplace_ISO3`, `Residence_ISO3` | Kategorikal | Tidak digunakan agar model tidak bergantung pada lokasi dan karena tingkat kategori sangat beragam. |
| Skala lain | `Narcissism` | Numerik | Indikator psikologis pendukung. |

### 4.3 Kondisi dan Kualitas Data

**Missing values.** Pada 13.050 responden berstatus `Accept`, nilai kosong tidak tersebar merata. `highestleague` kosong pada seluruh baris sehingga dieliminasi. Kolom `League` kosong pada 1.800 baris (13,79%), `Degree` pada 1.536 baris (11,77%), `GADE` pada 631 baris (4,84%), dan `SPIN_T` pada 618 baris (4,74%). Pada fitur yang tetap dipakai, kekosongan ditangani nanti melalui imputasi yang hanya di-*fit* pada data latih. Kolom `Hours` memiliki 27 nilai kosong dan `streams` 95 nilai kosong; keduanya juga ditangani sesuai pipeline Bab 5 setelah pemisahan data.

**Duplikasi.** `S. No.` dan `Timestamp` adalah identitas/pencatat waktu survei, bukan sinyal perilaku. Pemeriksaan dilakukan pada rekaman responden sebelum pemodelan; tidak ada proses penghapusan duplikasi yang diterapkan dalam notebook karena tidak ditemukan kebutuhan deduplikasi yang dapat dipertanggungjawabkan tanpa berisiko menghapus responden berbeda dengan jawaban serupa.

**Outlier dan noise.** Fitur `Hours` mengandung nilai ekstrem hingga 8.000 jam per minggu, padahal satu minggu hanya memiliki 168 jam. Sebanyak 42 responden valid mempunyai `Hours > 84`, sehingga diperlakukan sebagai jawaban tidak masuk akal/kesalahan input dan dikeluarkan. Setelah pemangkasan, distribusi `Hours` lebih masuk akal: Q1 = 12, median = 20, dan Q3 = 28 jam per minggu. Penskalaan yang tahan pencilan (*RobustScaler*) tetap digunakan pada fitur numerik yang masuk ke model.

**Ketimpangan kelas.** Dari data berstatus `Accept` sebelum filter `Hours`, kelas risiko rendah berjumlah 10.765 (82,49%) dan kelas risiko sedang–berat berjumlah 2.285 (17,51%), atau sekitar 1 : 4,71. Karena itu, akurasi tidak dipakai sebagai satu-satunya indikator; proyek menggunakan Macro-F1, recall kelas positif, ROC-AUC, dan PR-AUC serta pembobotan kelas.

### 4.4 Exploratory Data Analysis (EDA)

Tiga visualisasi berikut dihasilkan pada notebook EDA. Visualisasi dipakai untuk menetapkan keputusan persiapan data, bukan untuk memasukkan variabel GAD sebagai fitur.

| Visualisasi | Temuan utama | Implikasi |
|---|---|---|
| Distribusi target `risiko_tinggi` | Kelas positif hanya 17,51%; prediktor yang selalu memilih kelas 0 dapat tampak akurat tetapi tidak berguna untuk skrining. | Gunakan split terstratifikasi, *class weight*, dan metrik yang sensitif pada kelas minoritas. |
| Histogram/boxplot `Hours` | Terdapat 42 nilai di atas 84 jam/minggu dan nilai maksimum mentah 8.000 jam/minggu; setelah pemangkasan, mayoritas responden bermain sekitar 2–4 jam per hari. | Terapkan batas kualitas `Hours <= 84` sebelum split dan gunakan penskalaan robust. |
| Matriks korelasi faktor psikologis dengan kecemasan | `SPIN_T` berkorelasi positif paling kuat dengan `GAD_T` (r = +0,45) dan `risiko_tinggi` (r = +0,35); `SWL_T` berkorelasi negatif dengan `GAD_T` (r = -0,40); korelasi linear `Hours` dengan `GAD_T` lemah (r = +0,10). | Risiko tidak cukup dijelaskan oleh durasi bermain saja. Model perlu menggabungkan pola bermain, demografi, dan indikator psikologis eksternal yang bebas dari butir GAD. |

**Ringkasan Bab 4.** Data layak untuk klasifikasi biner setelah menerapkan kriteria inklusi dan aturan kualitas jam bermain. Risiko utama yang teridentifikasi ialah *label leakage* melalui keluarga GAD, *class imbalance*, nilai `Hours` ekstrem, dan nilai hilang. Keempatnya ditangani secara eksplisit pada Bab 5.

---

## 5. DATA PREPARATION  **[L]**

Seluruh proses pada bab ini dapat direproduksi melalui [notebooks/02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb). *Random seed* ditetapkan ke `42` agar pembagian data dapat diulang secara konsisten.

### 5.1 Pembersihan Data dan Pembentukan Target

Urutan pembersihan dilakukan sebagai berikut:

1. Membaca CSV dengan `latin1`, lalu hanya mempertahankan `accept == 'Accept'` (13.050 baris).
2. Menerapkan aturan kualitas `Hours <= 84`, sehingga tersisa 12.981 baris untuk pemodelan.
3. Membentuk target biner `risiko_tinggi` dari `GAD_T`: 1 untuk `GAD_T >= 10` dan 0 untuk nilai di bawahnya.
4. Mengeluarkan 19 kolom dari kandidat fitur: seluruh kolom GAD (`GAD1`–`GAD7`, `GAD_T`, `GADE`) sebagai sumber *label leakage*; serta `S. No.`, `Timestamp`, `accept`, `highestleague`, `League`, `Birthplace`, `Residence`, `Reference`, `Residence_ISO3`, dan `Birthplace_ISO3` yang berupa identitas, metadata, sangat kosong, atau berpotensi membuat model terlalu spesifik pada lokasi.

Hasil langkah ini adalah 36 fitur kandidat sebelum rekayasa fitur. Target `GAD_T` hanya digunakan satu kali untuk membentuk `risiko_tinggi` dan tidak ikut ke dalam matriks `X`.

### 5.2 Feature Engineering dan Seleksi Fitur

Untuk membatasi kardinalitas kategori serta membuat representasi lebih stabil, fitur tekstual direkayasa menjadi berikut:

| Fitur asal | Transformasi | Hasil |
|---|---|---|
| `Game` | Mempertahankan `League of Legends`, `Counter Strike`, `Starcraft 2`, `World of Warcraft`, dan `Other`; kategori lain digabung sebagai `Other`. | `Game_Clean` |
| `Playstyle` | Mempertahankan empat gaya bermain terumum (tiga variasi multiplayer dan `Singleplayer`); sisanya menjadi `Other`. | `Playstyle_Clean` |
| `whyplay` | Mempertahankan `having fun`, `improving`, `winning`, dan `relaxing`; sisanya menjadi `Other`. | `whyplay_Clean` |
| `SPIN_T`, `SWL_T` | Membentuk rasio fobia sosial terhadap kepuasan hidup: `SPIN_T / (SWL_T + 1.0)`. Konstanta 1 mencegah pembagian nol. | `SPIN_SWL_Ratio` |

Kolom teks asal `Game`, `Playstyle`, dan `whyplay` kemudian diganti dengan versi bersihnya. Matriks kandidat setelah rekayasa berisi 37 kolom. Pipeline model memilih tujuh fitur numerik (`Hours`, `streams`, `Age`, `SWL_T`, `SPIN_T`, `Narcissism`, `SPIN_SWL_Ratio`) dan tujuh fitur kategorikal (`Platform`, `Gender`, `Work`, `Degree`, `Game_Clean`, `Playstyle_Clean`, `whyplay_Clean`). Skor agregat `SWL_T` dan `SPIN_T` diprioritaskan agar butir kuesioner individual yang sangat berkorelasi tidak mendominasi representasi.

### 5.3 Transformasi Data Tanpa Leakage

Transformasi dipisahkan berdasarkan tipe fitur dan dibungkus dalam `ColumnTransformer`:

| Kelompok | Langkah transformasi | Alasan |
|---|---|---|
| Numerik | `SimpleImputer(strategy='median')` → `RobustScaler()` | Median tahan terhadap pencilan; `RobustScaler` menggunakan median dan IQR sehingga lebih stabil untuk distribusi miring seperti jam bermain. |
| Kategorikal | `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` | Mengisi nilai kosong dengan kategori paling umum dan mengubah kategori nominal tanpa memberi urutan palsu; kategori baru di data uji tidak menyebabkan kegagalan transformasi. |

Penting: `fit_transform()` dilakukan **hanya pada `X_train`**. Parameter imputasi, median/IQR scaler, dan daftar kategori encoder tidak pernah mempelajari `X_test`; data uji hanya menerima `transform()`. Hasilnya berbentuk 36 dimensi: `X_train_proc` berukuran `(10.384, 36)` dan `X_test_proc` berukuran `(2.597, 36)`. Verifikasi notebook menunjukkan tidak ada nilai NaN tersisa pada kedua matriks hasil transformasi.

### 5.4 Pembagian Data

Data dibagi menggunakan `train_test_split` dengan konfigurasi `test_size=0.20`, `stratify=y`, dan `random_state=42`.

| Partisi | Jumlah sampel | Proporsi kelas positif | Penggunaan |
|---|---:|---:|---|
| Data latih | 10.384 | 17,38% | Pelatihan model, *fitting* preprocessing, dan validasi silang/tuning. |
| Data uji hold-out | 2.597 | 17,40% | Evaluasi akhir saja; tidak digunakan untuk *fit*, pemilihan fitur, scaling, tuning, atau balancing. |

Stratifikasi menjaga prevalensi risiko tinggi hampir sama pada kedua partisi. Untuk tahap tuning selanjutnya, notebook menyiapkan `StratifiedKFold` sehingga rasio kelas juga dapat dijaga pada setiap *fold* data latih.

### 5.5 Penanganan Class Imbalance

Strategi utama adalah *cost-sensitive learning*, bukan *oversampling* pada data uji. Bobot dihitung hanya dari `y_train` menggunakan `compute_class_weight(class_weight='balanced')` dan menghasilkan:

| Kelas | Makna | Bobot |
|---|---|---:|
| 0 | Risiko rendah/minimal | 0,6052 |
| 1 | Risiko sedang–berat | 2,8765 |

Bobot tersebut digunakan sebagai `class_weight='balanced'`/ekuivalennya pada Logistic Regression dan Random Forest, serta sebagai bobot *Weighted Binary Cross-Entropy* pada ResNet. Notebook juga menyediakan demonstrasi SMOTE sebagai opsi, tetapi pustaka `imblearn` belum tersedia pada eksekusi saat ini; karena itu, hasil pipeline yang dipakai tetap pembobotan kelas. Jika SMOTE dipakai pada eksperimen berikutnya, resampling hanya boleh dilakukan setelah split dan hanya pada data/fold latih.

### 5.6 Ringkasan Keputusan Preparation

| Risiko yang ditemukan | Keputusan | Jaminan integritas |
|---|---|---|
| Kebocoran label dari GAD | Keluarkan `GAD1`–`GAD7`, `GAD_T`, dan `GADE` dari `X`. | Model tidak menerima butir pembentuk target. |
| Jam bermain tidak masuk akal | Filter `Hours <= 84`. | Menghilangkan 42 jawaban ekstrem yang tidak kredibel. |
| Missing values | Imputasi median/modus di dalam pipeline. | Parameter imputasi dipelajari dari data latih saja. |
| Skala dan distribusi numerik | `RobustScaler`. | Lebih tahan terhadap nilai ekstrem tersisa. |
| Kategori baru/beragam | Penggabungan kategori langka dan `OneHotEncoder(handle_unknown='ignore')`. | Dimensi fitur konsisten pada data uji. |
| Kelas tidak seimbang | Bobot kelas dari `y_train`. | Kelas positif diberi penalti kesalahan lebih tinggi tanpa menyentuh data uji. |

Dengan demikian, keluaran Bab 5 adalah dua matriks fitur siap latih yang bersih, konsisten, dan bebas *data leakage*, beserta target serta bobot kelas yang siap dipakai pada Bab 6.

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
