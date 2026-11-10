## INFORMASI PROYEK

| Komponen | Keterangan |
| :--- | :--- |
| **Judul Proyek** | Deteksi Dini Risiko Kecemasan Sedang–Berat pada Gamer Online Berdasarkan Pola Bermain dan Faktor Psikologis Menggunakan Machine Learning |
| **Kelompok / Kelas** | Kelompok 2 / TRPL 5B |
| **Nama Mahasiswa** | 1. Mohammad Hanif Huda A (NIM: 244311049)<br>2. Muhammad Akbar Fadilah (NIM: 244311051)<br> |
| **Program Studi** | Rekayasa Perangkat Lunak (TRPL) |
| **Mata Kuliah** | Data Science |
| **Dosen Pengampu** | Gus Nanang Syaifuddiin, S.Kom., M.Kom. |
| **Tahun Akademik** | 2026/2027, Semester Gasal (Semester 5) |
| **Link Repository GitHub** | https://github.com/akbrvdv/kel2_datascience_uasproject |
| **SDG yang Dipilih** | SDG 3: Kehidupan Sehat dan Sejahtera (Good Health and Well-being) |
| **Target SDG Relevan** | Target 3.4: Mengurangi sepertiga dari kematian dini akibat penyakit tidak menular melalui pencegahan dan pengobatan, serta meningkatkan kesehatan mental dan kesejahteraan. |

---

## BAB I. PENDAHULUAN

### 1.1 Latar Belakang

Permainan daring (*online gaming*) telah menjadi salah satu aktivitas hiburan harian paling populer bagi jutaan orang di seluruh dunia. Kendati menawarkan sarana relaksasi dan interaksi sosial virtual, intensitas bermain yang berlebihan berpotensi memicu konsekuensi psikologis yang merugikan. Pada tahun 2018, *World Health Organization* (WHO) secara resmi mengklasifikasikan *gaming disorder* sebagai salah satu bentuk gangguan psikiatri dalam *International Classification of Diseases* edisi ke-11 (ICD-11). Berbagai studi empiris menunjukkan bahwa tingginya prevalensi gangguan kecemasan umum memiliki korelasi erat dengan pola *Internet Gaming Disorder* (IGD) (Latubessy et al., 2024).

Dalam asesmen psikologis standar, tingkat kecemasan umum umumnya diukur menggunakan instrumen baku kuesioner *Generalized Anxiety Disorder-7* (GAD-7) dengan rentang skor $0-21$ (Spitzer et al., 2006). Individu dengan skor GAD-7 $\ge 10$ dikategorikan mengalami risiko kecemasan tingkat sedang hingga berat yang memerlukan evaluasi klinis lebih lanjut. Akan tetapi, proses pengisian kuesioner mandiri yang panjang membutuhkan kesediaan, waktu, dan partisipasi aktif dari responden. Oleh karena itu, pengembangan sistem deteksi dini (*early screening tool*) berbasis model machine learning yang memanfaatkan indikator pola bermain dan faktor psikologis ringkas dapat menjadi instrumen penapisan awal yang sangat berguna.

Penelitian terdahulu yang mengkaji data gamer sebagian besar berfokus pada komparasi performa algoritma. Misalnya, penelitian Karthik dan Ashokkumar (2023) membandingkan Recurrent Neural Network (RNN) dengan Decision Tree J48 pada dataset Kaggle (20.200 sampel) untuk menganalisis keterkaitan game online terhadap gangguan mental remaja, di mana RNN menunjukkan efektivitas yang lebih unggul. Selain itu, Latubessy et al. (2024) mengklasifikasikan kecemasan pemain game menjadi empat kategori tingkat keparahan menggunakan metode *Fine-Tuned Attribute Weighted Naïve Bayes* (FTAWNB) dengan klaim akurasi mencapai 99,22%. Namun, capaian akurasi setinggi itu perlu dievaluasi secara kritis terhadap risiko kebocoran label (*label leakage*) apabila item kuesioner GAD dimasukkan ke dalam fitur prediksi. Studi lanjutan oleh Latubessy et al. (2025) juga menemukan bahwa algoritma Naïve Bayes sangat rentan terhadap *outlier*, sehingga prosedur pra-pemrosesan data yang tepat mutlak diperlukan.

Meskipun demikian, terdapat beberapa celah penting yang perlu diinvestigasi lebih lanjut. Pertama, belum ada kejelasan mengenai sejauh mana variabel pola bermain saja mampu memprediksi kecemasan dibandingkan jika diintegrasikan dengan faktor psikologis. Eksplorasi awal pada 13.050 responden menunjukkan bahwa korelasi jam bermain terhadap skor GAD-7 relatif rendah ($r \approx 0,10$). Model yang hanya menggunakan fitur pola bermain menghasilkan nilai ROC-AUC sekitar $0,58-0,60$, sedangkan penggabungan faktor psikologis seperti kepuasan hidup (*Satisfaction with Life* / SWL), kecemasan sosial (*Social Phobia Inventory* / SPIN), dan narsisisme mampu meningkatkan ROC-AUC hingga mencapai $\approx 0,79$. Kedua, penelitian komparatif yang menguji secara langsung model statistik parametrik dasar (Logistic Regression), model ensemble pohon (XGBoost / Random Forest), dan jaringan saraf tiruan (Multi-Layer Perceptron / MLP) pada dataset tabular gamer yang mengalami ketimpangan kelas (*imbalanced data*) masih sangat terbatas.

Berdasarkan permasalahan tersebut, penelitian ini bertujuan untuk membandingkan kinerja tiga model, yaitu Logistic Regression sebagai *baseline*, XGBoost / Random Forest sebagai representasi *advanced machine learning*, serta Multi-Layer Perceptron (MLP) sebagai arsitektur *deep learning* dalam memprediksi risiko kecemasan sedang–berat pada gamer online. Seluruh butir pertanyaan GAD dikeluarkan dari variabel fitur untuk mencegah *label leakage*. Evaluasi model dilakukan dengan menangani ketidakseimbangan kelas ($\approx 17,5\%$ kelas risiko tinggi) melalui metrik Macro-F1, Recall, ROC-AUC, dan PR-AUC. Hasil penelitian ini diharapkan dapat memberikan bukti empiris mengenai determinan kecemasan pada gamer serta menyediakan instrumen deteksi dini yang selaras dengan SDG 3 Target 3.4.

### 1.2 Rumusan Masalah

1. Bagaimana merumuskan deteksi risiko kecemasan sedang–berat (skor GAD-7 $\ge 10$) pada gamer online sebagai permasalahan klasifikasi biner tanpa menyertakan item-item kuesioner GAD sebagai fitur masukan (*leakage-free*)?
2. Seberapa besar perbedaan kemampuan prediktif model antara penggunaan variabel pola bermain saja, penambahan atribut demografis, dan integrasi bersama faktor psikologis?
3. Model manakah di antara Logistic Regression, XGBoost / Random Forest, dan Multi-Layer Perceptron (MLP) yang memberikan performa klasifikasi terbaik pada kondisi dataset yang tidak seimbang (*class imbalance*)?
4. Fitur-fitur apa saja yang memiliki kontribusi paling signifikan (*feature importance*) terhadap prediksi risiko kecemasan pada pemain game berdasarkan analisis interpretasi model?

### 1.3 Tujuan Penelitian

1. Membangun dan melatih model klasifikasi biner dengan target metrik evaluasi pada data uji berupa Macro-F1 $\ge 0,65$, recall kelas risiko tinggi $\ge 0,70$, dan ROC-AUC $\ge 0,78$.
2. Menganalisis dan melaporkan komparasi performa model pada tiga skenario kombinasi fitur: Skenario A (Pola Bermain), Skenario B (Pola Bermain + Demografis), dan Skenario C (Pola Bermain + Demografis + Faktor Psikologis).
3. Membandingkan performa tiga arsitektur model (Logistic Regression, XGBoost / Random Forest, dan MLP) menggunakan protokol evaluasi terstandarisasi berbasis *Stratified 5-Fold Cross-Validation*.
4. Mengidentifikasi lima fitur teratas (*Top-5 features*) yang paling berpengaruh terhadap estimasi risiko kecemasan menggunakan metode *Permutation Importance* atau *SHapley Additive exPlanations* (SHAP).
5. Menyusun simpulan hasil pemodelan sebagai acuan empiris dan rekomendasi sistem penapisan dini (*early screening tool*) guna mendukung pencapaian SDG 3 Target 3.4 dalam mempromosikan kesehatan mental.

### 1.4 Keterkaitan dengan SDGs dan Pertimbangan Etika

Rincian keterkaitan penelitian ini dengan Sustainable Development Goals (SDG 3 Target 3.4) serta aspek etika pengolahan data disajikan pada Tabel 1.1:

#### Tabel 1.1: Keterkaitan dengan SDGs dan Pertimbangan Etika

| Komponen Evaluasi | Deskripsi Keterkaitan dan Implementasi |
| :--- | :--- |
| **SDG yang Dipilih** | SDG 3: Kehidupan Sehat dan Sejahtera (Good Health and Well-being). |
| **Target SDG Relevan** | Target 3.4: Mengurangi sepertiga dari kematian dini akibat penyakit tidak menular melalui pencegahan dan pengobatan, serta meningkatkan kesehatan mental dan kesejahteraan. |
| **Masalah Nyata** | Tingginya prevalensi kecemasan pada komunitas pemain game online yang berkorelasi dengan fenomena *gaming disorder* (ICD-11 WHO 2018), di mana penapisan konvensional membutuhkan kuesioner panjang yang memakan waktu dan bergantung pada partisipasi aktif responden. |
| **Peran Machine Learning** | Machine learning dan deep learning dimanfaatkan untuk mengenali pola non-linear dan interaksi multivariat antara kebiasaan bermain, profil demografis, dan indikator psikologis, guna memprediksi status risiko kecemasan sedang–berat secara cepat dan non-invasif sebagai instrumen penapisan awal (*triage*). |
| **Indikator Kontribusi** | Tersedianya perbandingan performa model yang terukur (Macro-F1, Recall, ROC-AUC, PR-AUC) yang memberikan landasan empiris bahwa intervensi kesehatan mental pada gamer tidak boleh semata-mata berfokus pada pembatasan jam bermain, melainkan harus menyentuh faktor psikologis yang mendasarinya. |
| **Pertimbangan Etika** | 1. **Bukan Alat Diagnosis Klinis**: Model berfungsi semata-mata sebagai alat bantu penapisan awal (*early screening*), di mana vonis dan keputusan klinis mutlak berada di bawah wewenang tenaga medis dan psikolog profesional.<br>2. **Privasi & Kerahasiaan Data**: Dataset survei bersifat publik, teranonimkan sepenuhnya tanpa data identitas pribadi (bebas PII), dan hanya responden berstatus *Accept* yang digunakan dalam analisis.<br>3. **Keterbatasan dan Bias Sampel**: Data didominasi oleh pemain *League of Legends*, platform PC, berjenis kelamin pria, dan berasal dari negara-negara Barat (hanya 1 baris dari Indonesia), sehingga temuan tidak dapat langsung digeneralisasi untuk populasi lokal.<br>4. **Pencegahan Label Leakage**: Seluruh pertanyaan GAD (GAD1–GAD7, skor total, dan level GAD) dieliminasi dari daftar fitur untuk menjamin integritas model. |
| **Sumber Ide** | Perhatian global terhadap isu kesehatan mental di kalangan gamer yang diakui oleh WHO serta tersedianya dataset terbuka *Online Gaming Anxiety Data* di Kaggle. |

---

## BAB II. LANDASAN KERANGKA KERJA DAN FORMULASI DATASET

### 2.1 Alur Kerangka Kerja ML Lifecycle (L-M-O-P)

Pelaksanaan proyek ini dirancang mengikuti empat tahapan terstruktur *Machine Learning Lifecycle*: tahap **Learning Problem (L)** berfokus pada formulasi tugas klasifikasi biner, telaah mendalam dataset terhadap potensi kebocoran label (*label leakage*), pemetaan target SDG 3.4, serta penentuan metrik evaluasi yang sensitif terhadap ketidakseimbangan kelas (*class imbalance*); tahap **Model Design (M)** menetapkan pemilihan arsitektur tiga tingkatan model (Logistic Regression sebagai *baseline*, XGBoost / Random Forest sebagai *advanced ensemble*, dan Multi-Layer Perceptron sebagai *deep learning*); tahap **Optimization (O)** mencakup pemfilteran kriteria responden `Accept`, pemangkasan outlier jam bermain ($< 84$ jam/minggu), pengelompokan teks kategori bebas, imputasi nilai hilang dan penskalaan fitur numerik yang terisolasi di dalam *pipeline* validasi, penanganan ketimpangan kelas menggunakan pembobotan kelas (*class weighting*), serta penyesuaian ambang batas klasifikasi (*threshold tuning*); serta tahap **Predict and Evaluate (P)** melakukan evaluasi komparatif menggunakan *Stratified 5-Fold Cross-Validation* dan pengujian akhir pada *hold-out test set* (20%) berbasis Macro-F1, Recall, ROC-AUC, PR-AUC, Confusion Matrix, dan interpretasi fitur menggunakan SHAP / Permutation Importance.

### 2.2 Formulasi Learning Problem dan Spesifikasi Dataset

Rumusan teknis masalah pembelajaran mesin dan spesifikasi dataset dirangkum pada Tabel 2.1:

#### Tabel 2.1: Formulasi Learning Problem dan Spesifikasi Dataset

| Komponen Masalah | Formulasi Formal dan Spesifikasi Teknis |
| :--- | :--- |
| **Target Prediksi** | Status risiko kecemasan biner: `risiko_tinggi` = 1 jika total skor $\text{GAD-7} \ge 10$ (tingkat kecemasan sedang hingga berat), dan `risiko_tinggi` = 0 jika $\text{GAD-7} < 10$ (tingkat kecemasan minimal hingga ringan). Jenis tugas adalah Supervised Binary Classification (dengan opsi klasifikasi 4 kelas ordinal sebagai eksperimen tambahan). |
| **Objective Matematis** | Memaksimalkan skor Macro-F1 dan Recall pada kelas risiko tinggi ($\approx 17,5\%$ proporsi data) dengan meminimalkan fungsi loss *Weighted Binary Cross-Entropy* untuk mengompensasi ketidakseimbangan kelas. Evaluasi komprehensif didukung oleh metrik ROC-AUC, PR-AUC, dan Confusion Matrix pada data uji. |
| **Karakteristik Dataset** | Dataset sekunder tabular *Online Gaming Anxiety Data* yang dihimpun melalui survei daring sukarela oleh Divyansh22 di platform Kaggle (total $13.464$ baris $\times$ $55$ variabel, tersaring menjadi $13.050$ baris setelah pemfilteran kriteria inklusi status responden `Accept`). URL: https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data. Fitur terpakai mencakup Pola Bermain (`Hours`, `streams`, `Game`, `Platform`, `whyplay`, `Playstyle`), Demografis (`Age`, `Gender`, `Work`, `Degree`), serta Faktor Psikologis (`SWL_T` / Kepuasan Hidup, `SPIN_T` / Fobia Sosial, dan `Narcissism`). Fitur yang dieksklusikan demi mencegah *data leakage* mencakup `GAD1`–`GAD7`, `GAD_T`, `GADE`, `highestleague` (banyak nilai hilang), `Timestamp`, serta atribut penanda lokasi. |
| **Skema Partisi Data** | Data dibagi menjadi data latih (80%) dan data uji (*hold-out test set*, 20%) menggunakan *Stratified Split* untuk menjaga rasio distribusi kelas target. Pada data latih, diterapkan *Stratified 5-Fold Cross-Validation* untuk penyetelan hyperparameter. Seluruh tahapan imputasi, penskalaan, dan pra-pemrosesan dienkapsulasi di dalam *fold pipeline* guna mencegah kebocoran informasi (*data leakage*). |

---

## BAB III. RENCANA PENDEKATAN PEMODELAN DAN EVALUASI

### 3.1 Model Baseline: Logistic Regression

Model acuan dasar (*baseline*) yang digunakan adalah Logistic Regression dengan konfigurasi bobot kelas seimbang (`class_weight='balanced'`). Model ini dipilih karena bersifat parametrik, sederhana, efisien dalam komputasi, dan koefisien bobotnya mudah diinterpretasikan dalam bentuk *log-odds* untuk menganalisis arah korelasi fitur terhadap risiko kecemasan. Logistic Regression berfungsi sebagai tolok ukur awal guna menilai sejauh mana batas keputusan linear dapat memisahkan kedua kelas risiko sebelum beralih ke model yang lebih rumit.

Secara matematis, model menghitung kombinasi linear dari fitur masukan dan memetakannya melalui fungsi sigmoid ke dalam probabilitas kontinu antara 0 dan 1. Hasil pengujian model dasar ini akan menjadi pembanding utama untuk membuktikan apakah algoritma non-linear dan jaringan saraf tiruan memberikan peningkatan akurasi dan recall yang signifikan. Sebagai pelengkap pada tahap eksplorasi awal, algoritma Gaussian Naïve Bayes juga dapat diikutsertakan sebagai pembanding tambahan karena kerap dirujuk dalam literatur klasifikasi kecemasan terdahulu.

### 3.2 Model Advanced Machine Learning: XGBoost / Random Forest Classifier

Model machine learning tingkat lanjut yang diterapkan adalah algoritma ensemble berbasis pohon, yaitu XGBoost (*Extreme Gradient Boosting*) Classifier atau Random Forest Classifier. Algoritma ensemble berbasis pohon sangat unggul dalam memproses data tabular campuran (fitur numerik dan kategorikal) serta memiliki ketahanan alami terhadap perbedaan skala fitur dan keberadaan pencilan (*outliers*).

XGBoost secara khusus mampu menangkap hubungan non-linear kompleks dan interaksi multivariat antarfaktor psikologis dan pola bermain tanpa memerlukan asumsi normalitas distribusi data. Selain itu, XGBoost mengintegrasikan fungsi regularisasi L1 (*Lasso*) dan L2 (*Ridge*) untuk mengontrol kompleksitas pohon keputusan dan mencegah *overfitting*. Optimasi hyperparameter seperti *learning rate*, kedalaman pohon (*max_depth*), jumlah pohon (*n_estimators*), dan penyesuaian *scale_pos_weight* akan dilakukan menggunakan validasi silang pada data latih.

### 3.3 Model Deep Learning: Multi-Layer Perceptron (MLP)

Model *deep learning* yang diaplikasikan adalah arsitektur Multi-Layer Perceptron (MLP) dengan 2 hingga 3 lapisan tersembunyi (*hidden layers*). Arsitektur ini dirancang untuk menguji efektivitas representasi fitur bertingkat pada data tabular berukuran menengah ($13.050$ sampel). Setiap lapisan tersembunyi dilengkapi fungsi aktivasi non-linear (seperti ReLU), normalisasi *Batch Normalization*, serta lapisan *Dropout* dengan rasio $0,2$ hingga $0,5$ untuk mereduksi risiko *co-adaptation* neuron.

Pelatihan model memanfaatkan optimizer Adam dengan penyesuaian tingkat pembelajaran (*learning rate schedule*) dan fungsi loss *Weighted Binary Cross-Entropy* untuk memitigasi dampak ketidakseimbangan kelas. Mekanisme *Early Stopping* berbasis loss validasi diterapkan untuk menghentikan pelatihan saat model mulai mengalami *overfitting*. Implementasi ini bertujuan menguji secara empiris apakah arsitektur *deep learning* mampu memberikan generalisasi yang lebih baik atau sebanding dengan model ensemble pohon pada data tabular kesehatan mental.

### 3.4 Desain Eksperimen dan Metrik Pengujian

Seluruh model dievaluasi secara konsisten menggunakan protokol eksperimental yang seragam:
1. **Tiga Skenario Fitur (Ablation Study)**: Pengujian dilakukan pada Skenario A (Pola Bermain saja), Skenario B (Pola Bermain + Demografis), dan Skenario C (Pola Bermain + Demografis + Faktor Psikologis) untuk mengisolasi seberapa besar daya prediksi masing-masing kelompok variabel.
2. **Protokol Validasi Silang**: Diterapkan *Stratified 5-Fold Cross-Validation* pada 80% data latih (dilaporkan dalam rata-rata $\pm$ standar deviasi) untuk memilih arsitektur terbaik dan menyetel ambang batas keputusan (*optimal decision threshold*), dilanjutkan dengan pengujian final pada 20% *hold-out test set*.
3. **Metrik Evaluasi**: Mengingat distribusi kelas yang tidak seimbang ($\approx 17,5\%$ kelas positif), evaluasi menitikberatkan pada metrik **Macro-F1 Score** untuk keseimbangan performa antarkelas, **Recall (Kelas Risiko Tinggi)** untuk meminimalkan *false negative* (pasien berisiko yang terlewat), serta **ROC-AUC** dan **PR-AUC** untuk menilai performa diskriminasi probabilitas secara menyeluruh.
4. **Analisis Kepentingan Fitur**: Menentukan lima fitur teratas (*Top-5 features*) yang paling berpengaruh terhadap keluaran model menggunakan analisis *Permutation Feature Importance* atau nilai *SHapley Additive exPlanations* (SHAP).

---

## DAFTAR PUSTAKA

Divyansh22. (n.d.). *Online gaming anxiety data* [Data set]. Kaggle. https://www.kaggle.com/datasets/divyansh22/online-gaming-anxiety-data

Karthik, M. A., & Ashokkumar, S. (2023). Predicting the link between online gaming and mental disorders in youth: RNN vs J48 decision tree. *Proceedings of the International Conference on Health Informatics*, 403–408. SciTePress. https://doi.org/10.5220/0012504900003739

Latubessy, A., Wardoyo, R., Musdholifah, A., & Kusrohmaniah, S. (2024). Fine tuning attribute weighted naïve Bayes model for detecting anxiety disorder levels of online gamers. *International Journal of Electrical and Computer Engineering*, 14(3), 3277–3286. https://doi.org/10.11591/ijece.v14i3.pp3277-3286

Latubessy, A., Wardoyo, R., Musdholifah, A., & Kusrohmaniah, S. (2025). Fine-tuned attribute weighted naïve Bayes with modified partial instances reduction for gaming disorder classification. *HighTech and Innovation Journal*, 6(1), 1-14. https://doi.org/10.28991/HIJ-2025-06-01-05

Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006). A brief measure for assessing generalized anxiety disorder: The GAD-7. *Archives of Internal Medicine*, 166(10), 1092–1097. https://doi.org/10.1001/archinte.166.10.1092
