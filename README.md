# 🏆 NER Olahraga Indonesia: Identifikasi Entitas Teks Menggunakan Deep Learning

**Implementasi Arsitektur LSTM dengan Fitur Part-of-Speech Tagging untuk Teks Berbahasa Indonesia**

---

!(images/sports_text_data_analysis.jpg)

## 📌 1. Domain Proyek: Natural Language Processing (NLP) & Analisis Olahraga

Dunia olahraga menghasilkan volume data tekstual yang masif setiap harinya, mulai dari berita pertandingan, ulasan performa atlet, statistik, hingga percakapan di media sosial. Di dalam teks-teks ini terkandung informasi berharga seperti **nama atlet, tim, lokasi pertandingan, dan nama liga** yang menjadi kunci untuk berbagai aplikasi, termasuk analisis performa, jurnalisme otomatis, sistem rekomendasi, dan platform *fantasy sports*. Untuk mengekstrak informasi ini secara efisien, diperlukan teknologi **Named Entity Recognition (NER)**.

Namun, penerapan NER pada domain olahraga, khususnya untuk teks berbahasa Indonesia, menghadapi tantangan unik. Model NER umum yang dilatih pada data berita atau Wikipedia seringkali gagal mengenali entitas spesifik olahraga. Nama seperti "Persib Bandung" (tim), "Stadion Gelora Bung Karno" (lokasi), atau "Liga 1" (liga) bersifat sangat kontekstual dan tidak umum ditemukan dalam korpus data generik (Seti et al., 2020). Akibatnya, sistem NER standar seringkali memberikan hasil yang tidak akurat, menghambat potensi analisis data olahraga secara mendalam (Gunawan et al., 2018).

Proyek ini bertujuan untuk mengatasi kesenjangan tersebut dengan mengembangkan model NER yang dirancang khusus untuk domain olahraga berbahasa Indonesia. Dengan melatih model pada dataset yang relevan, penelitian ini berupaya membangun sistem yang mampu mengidentifikasi dan mengklasifikasikan entitas kunci dalam teks olahraga dengan lebih akurat, membuka jalan bagi pemanfaatan data tekstual olahraga yang lebih canggih.

---

## 🎯 2. Business Understanding

### 🔍 Problem Statements

1.  Bagaimana cara membangun sistem *Named Entity Recognition* (NER) yang dapat secara efektif mengidentifikasi entitas spesifik domain olahraga (atlet, tim, stadion, liga) dari teks berbahasa Indonesia?
2.  Mengapa model NER konvensional menunjukkan performa yang rendah pada teks olahraga, dan apa tantangan utama yang terkait dengan variasi dan keunikan entitas di domain ini?
3.  Pendekatan apa yang dapat diusulkan untuk meningkatkan akurasi sistem NER, meskipun dihadapkan pada keterbatasan jumlah data pelatihan yang telah dilabeli secara manual?

### 🎯 Objectives

1.  Mengembangkan dan mengimplementasikan model NER menggunakan arsitektur **Long Short-Term Memory (LSTM)** yang mampu mengenali delapan kategori entitas spesifik dalam teks olahraga.
2.  Menganalisis performa model yang hanya menggunakan **Part-of-Speech (POS) tagging** sebagai fitur input utama untuk memahami batas kemampuannya.
3.  Mengidentifikasi kelemahan model berdasarkan hasil evaluasi dan memberikan rekomendasi teknis yang konkret untuk perbaikan di masa mendatang, seperti penambahan data, rekayasa fitur, dan eksplorasi arsitektur yang lebih canggih.

### 💡 Solusi yang Diusulkan

Mengimplementasikan model **Long Short-Term Memory (LSTM)** *from scratch* menggunakan PyTorch. Model ini dipilih karena kemampuannya dalam memproses data sekuensial seperti teks dan menangkap dependensi antar kata. Sebagai fitur input utama, model akan memanfaatkan **Part-of-Speech (POS) tag** dari setiap kata, yang diekstrak menggunakan *library* Stanza, untuk mempelajari pola gramatikal yang dapat membantu identifikasi entitas.

---

## 📁 3. Dataset Overview

* **Sumber**: *Scraping* dari berbagai portal berita olahraga Indonesia (CNN Indonesia, Detik Sport), artikel/jurnal online, dan media sosial (X).
* **Jumlah Entri**: 331 kalimat.
* **Proses Pelabelan**: Dilakukan secara manual dengan format *plain tagging*.
* **Bahasa**: Indonesia.

---

## 📋 4. Fitur & Label Dataset

### 📥 Fitur Input Utama

* **Part-of-Speech (POS) Tag**: Setiap kata dalam kalimat diubah menjadi label kelas katanya (misal: NOUN, VERB, PROPN) menggunakan *library* Stanza. Fitur inilah yang menjadi input bagi model LSTM.

### 🏷️ Kategori Label Entitas (Target)

| Label | Deskripsi Entitas | Contoh |
| :--- | :--- | :--- |
| `PER` | Nama Orang (Atlet, Pelatih, dll.) | "Lionel Messi" |
| `ORG` | Organisasi atau Tim | "Paris Saint-Germain" |
| `LOC` | Lokasi (Stadion, Kota, Negara) | "Parc des Princes" |
| `EVT` | Nama Acara atau Pertandingan | "Final NBA 2024" |
| `DATE` | Tanggal, Bulan, atau Tahun | "7 Juni 2024" |
| `SPORT`| Nama Cabang Olahraga | "Sepak bola" |
| `EQUIP`| Nama Alat Olahraga | "Bola", "Raket" |
| `LEAGUE`| Nama Liga atau Kompetisi | "La Liga", "Bundesliga" |
| `O` | *Outside* (Bukan entitas) | "mencetak", "di" |

---

## 🔍 5. Data Understanding & EDA

### ⚖️ **Distribusi Label Entitas**

Analisis awal pada dataset menunjukkan adanya **ketidakseimbangan kelas (class imbalance)** yang signifikan. Label **'O' (Outside)** jauh lebih dominan dibandingkan dengan delapan label entitas lainnya. Hal ini merupakan karakteristik umum dalam tugas NER, di mana sebagian besar kata dalam teks bukanlah entitas bernama.

* **Insight**: Ketidakseimbangan ini berisiko membuat model menjadi bias dan cenderung memprediksi label 'O'. Oleh karena itu, diperlukan strategi penanganan khusus seperti **pembobotan kelas (class weighting)** pada fungsi *loss* saat pelatihan.

### ✍️ **Contoh Data dan Pelabelan**

Untuk memberikan gambaran yang jelas, berikut adalah contoh satu baris data dari dataset:

| Kalimat Asli | Final | NBA | 2024 | akan | mulai | digulirkan | pada | 7 | Juni | 2024 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Label Entitas** | O | B-Event | I-Event | O | O | O | O | B-Date | I-Date | I-Date |

*Catatan: `B-` menandakan awal entitas dan `I-` menandakan kelanjutan entitas.*

---

## 🧹 6. Data Preparation

Proses persiapan data sangat krusial untuk memastikan input yang berkualitas bagi model. Berikut adalah ringkasan langkah-langkah yang dilakukan:

| Langkah | Penjelasan |
| :--- | :--- |
| **Tokenisasi & POS Tagging** | Setiap kalimat dipecah menjadi token (kata), dan setiap token diberi label POS-nya menggunakan *library* **Stanza**. Contoh: "Celtics" → `PROPN` (Proper Noun). |
| **Pemetaan ke Indeks** | Semua POS tag unik dan label entitas unik dipetakan ke indeks numerik (integer). Contoh: `PROPN` → `1`, `NOUN` → `2`; `B-PER` → `1`, `I-PER` → `2`. |
| **Padding** | Semua sekuens (kalimat) dibuat memiliki panjang yang sama (misalnya, 100 token) dengan menambahkan nilai *padding* (indeks 0). Ini diperlukan agar data dapat diproses dalam *batch* oleh model. |
| **Pembobotan Kelas** | Bobot dihitung untuk setiap kelas label berdasarkan frekuensi terbaliknya. Label 'O' diberi bobot yang sangat rendah, sementara label entitas lainnya diberi bobot yang lebih tinggi untuk memaksa model lebih memperhatikan entitas langka. |

---

## ⚙️ 7. Model Development: Arsitektur LSTM

Model NER ini dibangun menggunakan arsitektur **Long Short-Term Memory (LSTM)**, sebuah varian dari Recurrent Neural Network (RNN) yang sangat efektif untuk tugas pemrosesan sekuens.

### ✅ Alasan Pemilihan LSTM

* **Memahami Konteks**: LSTM mampu mengingat informasi dari langkah-langkah sebelumnya dalam sebuah sekuens (kalimat), memungkinkannya memahami konteks sebuah kata berdasarkan kata-kata di sekitarnya.
* **Dependensi Jangka Panjang**: LSTM dirancang khusus untuk mengatasi masalah *vanishing gradient*, sehingga dapat menangkap hubungan antara kata-kata yang berjauhan dalam satu kalimat.

### 🏗️ Arsitektur Model

Model ini terdiri dari tiga lapisan utama:
1.  **Embedding Layer**: Mengubah input berupa indeks numerik dari POS tag menjadi vektor padat (*dense vector*) dengan dimensi tertentu (misalnya, 20). Vektor ini merepresentasikan POS tag dalam ruang semantik.
2.  **Bidirectional LSTM Layer**: Lapisan inti yang memproses sekuens vektor embedding. Penggunaan *bidirectional* berarti data diproses dari dua arah (kiri-ke-kanan dan kanan-ke-kiri), memungkinkan model untuk menangkap konteks sebelum dan sesudah setiap kata secara bersamaan.
3.  **Fully Connected (Linear) Layer**: Lapisan output yang mengambil hasil dari LSTM dan memproyeksikannya ke dalam ruang dimensi yang sama dengan jumlah total label entitas. Lapisan ini menghasilkan skor (*logits*) untuk setiap kemungkinan label pada setiap token.

### 📉 Fungsi Loss

* **Cross-Entropy Loss dengan Pembobotan**: Digunakan untuk menghitung kesalahan prediksi. Dengan menerapkan **bobot kelas** yang telah dihitung sebelumnya, *loss* untuk kesalahan pada entitas langka akan lebih besar, mendorong model untuk belajar mengenalinya.

---

## 📏 8. Evaluation

### 🎯 Tujuan Evaluasi

Evaluasi bertujuan untuk mengukur secara kuantitatif seberapa baik model yang telah dilatih dapat mengidentifikasi dan mengklasifikasikan entitas dengan benar pada data yang belum pernah dilihat sebelumnya (*test set*).

### 📐 Metrik Evaluasi

* **Akurasi (Accuracy)**: Metrik utama yang digunakan dalam penelitian ini. Akurasi dihitung sebagai persentase prediksi label yang benar dari total semua prediksi yang dibuat pada *test set*.

$$\text{Akurasi} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Prediksi}} \times 100\%$$

Meskipun sederhana, metrik ini dapat menyesatkan pada dataset yang tidak seimbang. Namun, untuk laporan awal ini, akurasi digunakan sebagai indikator performa dasar.

---

## 📊 9. Hasil Evaluasi & Analisis

### 📉 Hasil Kuantitatif

Setelah dilatih selama 2000 epoch, model dievaluasi pada dataset uji dan mencapai hasil sebagai berikut:

> **Akurasi: 22.69%**

Akurasi yang sangat rendah ini secara jelas menunjukkan bahwa model memiliki performa yang buruk dan gagal mempelajari pola yang memadai untuk mengenali entitas olahraga secara akurat.

### 🔬 Analisis Kualitatif (Contoh Kesalahan)

Analisis manual pada output prediksi menunjukkan beberapa pola kesalahan yang umum terjadi:

| Kalimat Uji | Entitas Sebenarnya | Prediksi Model | Analisis Kesalahan |
| :--- | :--- | :--- | :--- |
| "...pada **7 Juni 2024**" | `DATE` | `LEAGUE` | Salah klasifikasi total. Model salah mengasosiasikan angka dengan entitas Liga. |
| "**Celtics** menjadi penguasa..." | `ORG` | `O` | Kegagalan mengenali entitas. Model tidak mampu mengidentifikasi nama tim sebagai sebuah organisasi. |
| "...sejak **regular season**" | `EVT` | `O` | Kegagalan mengenali entitas. Istilah spesifik olahraga tidak dikenali sebagai sebuah *event*. |

Kesalahan-kesalahan ini mengonfirmasi bahwa model tidak hanya salah mengklasifikasikan tetapi juga seringkali gagal mendeteksi keberadaan entitas sama sekali.

---

## 💡 10. Pembahasan & Rekomendasi

Hasil akurasi 22.69% yang sangat rendah menandakan bahwa pendekatan yang digunakan saat ini tidak memadai. Berikut adalah analisis mendalam mengenai penyebab kegagalan dan rekomendasi untuk perbaikan.

### 📉 **Analisis Penyebab Performa Rendah**

1.  **Keterbatasan Dataset**: Dengan hanya 331 kalimat, volume data terlalu kecil untuk melatih model *deep learning* seperti LSTM secara efektif. Model tidak memiliki cukup contoh untuk mempelajari variasi entitas yang luas.
2.  **Fitur POS Tag yang Kurang Informatif**: Mengandalkan HANYA pada POS tag sebagai input adalah kelemahan utama. Kata "Celtics" dan "Bandung" sama-sama memiliki POS tag `PROPN` (Proper Noun). Tanpa informasi semantik dari kata itu sendiri, model tidak dapat membedakan mana yang merupakan tim dan mana yang lokasi.
3.  **Arsitektur Model Sederhana**: Arsitektur LSTM dasar, meskipun baik untuk sekuens, mungkin terlalu sederhana untuk menangkap nuansa bahasa yang kompleks tanpa fitur tambahan yang kaya.
4.  **Ketidakseimbangan Kelas Ekstrem**: Meskipun telah menggunakan pembobotan kelas, dominasi label 'O' kemungkinan masih membuat model cenderung bermain aman dan tidak memprediksi entitas.

### 🚀 **Rekomendasi Peningkatan**

1.  **Perkaya Dataset (Data Augmentation & Collection)**: Langkah paling krusial adalah memperbesar dataset pelatihan secara signifikan. Kumpulkan lebih banyak data dan lakukan pelabelan, atau gunakan teknik augmentasi data untuk membuat variasi kalimat baru.
2.  **Gunakan Fitur yang Lebih Kuat (Word Embeddings)**: Ganti atau kombinasikan input POS tag dengan **Word Embeddings** (seperti Word2Vec, FastText, atau GloVe). Fitur ini menangkap makna semantik kata, memungkinkan model untuk memahami bahwa "Messi" mirip dengan "Ronaldo" (keduanya pemain).
3.  **Eksplorasi Arsitektur Modern (Pre-trained Models)**: Daripada melatih model *from scratch*, manfaatkan model bahasa *pre-trained* yang canggih seperti **IndoBERT** atau **spaCy**. Lakukan *fine-tuning* pada model ini dengan dataset olahraga. Pendekatan *transfer learning* ini telah terbukti sangat efektif dan merupakan standar industri saat ini.
4.  **Optimalkan Penanganan Imbalance**: Selain pembobotan kelas, eksplorasi teknik *resampling* seperti *oversampling* pada kelas minoritas atau *undersampling* pada kelas mayoritas ('O').

---

## ✅ 11. Kesimpulan

Penelitian ini berhasil mengimplementasikan model NER berbasis LSTM untuk teks olahraga berbahasa Indonesia, namun dengan hasil performa yang sangat rendah, yaitu **akurasi 22.69%**. Hasil ini secara tegas menunjukkan bahwa penggunaan dataset yang kecil (331 kalimat) dan fitur yang terbatas (hanya POS tag) tidak cukup untuk membangun sistem NER yang andal dan fungsional.

Meskipun model gagal mencapai akurasi yang diharapkan, penelitian ini memberikan kontribusi penting sebagai *proof-of-concept* yang menyoroti tantangan-tantangan utama dalam NER domain spesifik. Kegagalan ini menjadi dasar untuk merumuskan serangkaian rekomendasi yang jelas dan terarah, yaitu perlunya memperkaya dataset secara masif, menggunakan fitur semantik seperti *word embeddings*, dan mengadopsi arsitektur model *transformer* berbasis *transfer learning* (seperti IndoBERT) untuk mencapai performa yang lebih baik.

---

## 📚 12. Referensi

Nadeau, D. & Sekine, S., 2007. A survey of named entity recognition and classification. Lingvisticae Investigationes, 30(1), pp.3–26. Available at: https://nlp.cs.nyu.edu/sekine/papers/li07.pdf 

Popovski, G., Seljak, B.K. & Eftimov, T., 2023. A survey of named-entity recognition methods for food information extraction. IEEE Access, 11, pp.56789–56801. Available at: https://www.researchgate.net/publication/339215947_A_Survey_of_Named-Entity_Recognition_Methods_for_Food_Information_Extraction .

Wicaksono, A.F., Purwarianti, A. & Suhartono, D., 2023. Named entity recognition on Indonesian legal documents: a dataset and study using transformer-based models. Journal of Big Data, 10(1), https://www.researchgate.net/publication/339215947_A_Survey_of_Named-Entity_Recognition_Methods_for_Food_Information_Extraction ].

Gunawan, W., Suhartono, D., Purnomo, F. & Ongko, A., 2018. Named-Entity Recognition for Indonesian Language using Bidirectional LSTM-CNNs.3rd International Conference on Computer Science and Computational Intelligence 2018 , 135 , pp.425–432. Available at: https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050918X0012X/1-s2.0-S1877050918314832/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFg4A%2BNhqZVPkcrjBt9f9KYYt5oSKi4qEs5gDfwb6Jc2AiBOvY2OQj%2BrtHKKlIH3Us0S0qwn1HhgvUMgA%2F5Ftj1zMyq8BQiQ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIM9OTPZAIUFt0CuIf%2BKpAFf0DaWh8ZdsHE%2F692uCIQWPMa8%2Bg5XLdPcF1UHpYBcN43Va46EJgycP6kQWlciaum3DKCqctCeVDSrfkhg6965FtgAFwHbBq1GQb6m3hykFSLh3NFsN4T0U%2BpRG7yyD%2BOtBGyDUTx2%2FZMZQOMJnW2a56j9wcJ0YZ5WebiX6uyBHATodT%2B%2FdglMfjyMf05EWxpmrxXDgK31rsFq53NYR3SvRvHrSd5smhZg5FylqFwZu7GL75Bhu1p%2Blo%2FUEq22f%2Fq7yV2hNYSLTARr1Wt%2F%2Fl7KyocYCNItGEzCfo%2FuYN1qm1dPoCq2ofwbqlF9ir%2FEIMl62FAn%2BQ1FfPNAqx5TX1Odcohj0SdVTWN5zuSvxx%2BxxgDeLjaOlxC1O0KKrPP2oBNWymkf4B%2F0W2hDnm9Y3TrSs%2FUSqxG609%2B7Do0tL00XKfAht4srC0kwpDeA4CNVzB%2Fkudc6slMsQ6t8xxIuiUA38iWkKJ1JjafAuExDZkINzFitW4G075Mn8zA9Xy3wsR2xEO0hamdoDdKVhf9Oee3xwBFgfr6MN19i7f4dUv%2BP2fs4OL4ZaLyNmSziOA8%2FO6N2THLI3OQyfflhASp%2FxD%2B8G2b%2FAg7FVeEuxVTNBNPN1oVJfH%2FOrTHRTgNrq0KDsBvfJ%2Fk0sk7h1x1qdDQyunRygYki0yrMrzSPNaXTJ8RZkULGuz6n2Wcfj4PvKp62jxT5eYeNEj3vH3eunTnU3Hi6ziGeWXg3wyZPx9tNVp2FC51FUT6b%2FI4HwKv1Gme9rGNiOVazXoKiNcC6QO%2F0ZycZW6yHPLOx%2FdnmKBoufiyWGu4hbGRaOGVCu370xFa%2BJ3gwL35de2oqdR7g51gMx6RHRgCp2EWDJaeYaYnFaVaIwcwj9WsugY6sgFh1m8%2FvQJqFxIXL3FTyqXPM3NTGpfMHEq6vJJ4MoKyR%2BeUNYWs5UaiMeUTTKapkDEkxbKfNcYaQaisUdvbo%2Be91cKQvx9QQYbR1xudTQSKjOpR%2F23oqucjZmylQwoPpWe1O%2B9VVvI7GTJggL0cIofDuqHehS%2F6qx8vFlKr06PVz%2BxGBHTUh%2FoI9FikCZjME7sDP369Asg9rjd9fDf2pN%2B7uXf6IT0j48E8Nw%2B6BzQ1vzY8&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241130T161904Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUP2DI6MG%2F20241130%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e044426e631f62a90217ca24ab4ef553ba187676da0a64852489f0825fe1b22f&hash=8c563e051ee2e529d2c05c7a1fa3815a39c19c8df992f23ee46f810bab18d094&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050918314832&tid=spdf-717373fa-b3ee-4499-b3bc-75d6ceda46e3&sid=c225f3f17750964a778ba48472a572029f15gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f045e0107015d5d5053&rr=8eac28300e52409c&cc=id .

Jehangir, B., Radhakrishnan, S. & Agarwal, R., 2023. A survey on named entity recognition—datasets, tools, and methodologies. Expert Systems with Applications, 215, p.119431. Available at: https://www.sciencedirect.com/science/article/pii/S2949719123000146 .

Li, J., Sun, A., Han, J. & Li, C., 2022. A survey on deep learning for named entity recognition. IEEE Transactions on Knowledge and Data Engineering, 34(1), pp.50–70. Available at: https://ieeexplore.ieee.org/document/10184827 .
Seti, X. et al., 2020. Named-entity recognition in sports field based on a character-level graph convolutional network. Information, 11(1), p.30. Available at: https://www.mdpi.com/2078-2489/11/1/30 .


---
## Development Team
* I Putu Paramaananda Tanaya
* Muhammad Aldy Naufal Fadhilah 
* Jonathan Young
