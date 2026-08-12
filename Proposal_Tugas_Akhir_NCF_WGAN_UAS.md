# PROPOSAL TUGAS AKHIR
## PENGEMBANGAN SISTEM REKOMENDASI *E-COMMERCE* MELALUI INTEGRASI WASSERSTEIN GENERATIVE ADVERSARIAL NETWORKS (WGANRS) DAN NEURAL COLLABORATIVE FILTERING (NCF) UNTUK MENGATASI KELANGKAAN DATA (*DATA SPARSITY*)

---

**Diajukan untuk Memenuhi Syarat Kelulusan Mata Kuliah Ujian Akhir Semester (UAS)**  
**Mata Kuliah: Metodologi Penelitian Teknologi Informasi (MPTI) - IF-701**  

**Dosen Pengampu:**  
**Dr. Ucuk Darusalam, S.T., M.T.**  

---

**Disusun Oleh:**  
**Muhammad Salman Ramadhan Alfadhlih**  
**NIM: 220401010280**  

<br>

<div align="center">
  <img src="https://unsia.ac.id/wp-content/uploads/2021/09/Logo-UNSIA.png" alt="Logo Universitas Siber Asia" width="250"/>
  <br>
  <h3>PROGRAM STUDI PJJ INFORMATIKA</h3>
  <h3>FAKULTAS TEKNOLOGI INFORMASI</h3>
  <h3>UNIVERSITAS SIBER ASIA</h3>
  <h3>JAKARTA</h3>
  <h3>2026</h3>
</div>

---

## DAFTAR ISI
- [BAB I: PENDAHULUAN](#bab-i-pendahuluan)
  - [1.1 Latar Belakang](#11-latar-belakang)
  - [1.2 Rumusan Masalah](#12-rumusan-masalah)
  - [1.3 Batasan Masalah](#13-batasan-masalah)
  - [1.4 Tujuan Penelitian](#14-tujuan-penelitian)
  - [1.5 Manfaat Penelitian](#15-manfaat-penelitian)
- [BAB II: TINJAUAN PUSTAKA](#bab-ii-tinjauan-pustaka)
  - [2.1 Sistem Rekomendasi dan Collaborative Filtering (CF) Klasik](#21-sistem-rekomendasi-dan-collaborative-filtering-cf-klasik)
  - [2.2 Neural Collaborative Filtering (NCF)](#22-neural-collaborative-filtering-ncf)
  - [2.3 Fenomena Mode Collapse pada Generative Adversarial Networks (GAN)](#23-fenomena-mode-collapse-pada-generative-adversarial-networks-gan)
  - [2.4 Wasserstein Generative Adversarial Networks untuk Sistem Rekomendasi (WGANRS)](#24-wasserstein-generative-adversarial-networks-untuk-sistem-rekomendasi-wganrs)
  - [2.5 Evaluasi Kinerja Sistem Informasi](#25-evaluasi-kinerja-sistem-informasi)
- [BAB III: METODOLOGI PENELITIAN](#bab-iii-metodologi-penelitian)
  - [3.1 Diagram Alir dan Alur Kerja Sistem Usulan](#31-diagram-alir-dan-alur-kerja-sistem-usulan)
  - [3.2 Tahapan Operasional Sistem](#32-tahapan-operasional-sistem)
  - [3.3 Formulasi Matematika Model Usulan](#33-formulasi-matematika-model-usulan)
  - [3.4 Desain Pengujian dan Metrik Evaluasi Kinerja](#34-desain-pengujian-dan-metrik-evaluasi-kinerja)
- [DAFTAR PUSTAKA](#daftar-pustaka)

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Di era ledakan informasi digital saat ini, sistem rekomendasi (*recommender systems*) telah menjadi pilar utama dalam mengatasi kelebihan pilihan (*information overload*) pada berbagai platform web seperti *e-commerce*, layanan streaming, dan media sosial [4, 7]. Perusahaan besar seperti Netflix, Spotify, Amazon, dan Google sangat bergantung pada sistem rekomendasi untuk mempersonalisasi pengalaman pengguna, meningkatkan keterlibatan, dan mendorong konversi penjualan [7, 11, 240, 241, 328, 329]. Diperkirakan bahwa sistem rekomendasi bertanggung jawab atas sekitar 35% dari total pendapatan pada platform *e-commerce* global [235].

Metode penyaringan kolaboratif (*Collaborative Filtering* / CF) adalah salah satu strategi yang paling populer dan akurat [242, 243]. CF bekerja dengan memanfaatkan data interaksi historis pengguna (seperti klik, rating, atau riwayat pembelian) untuk memprediksi item yang diminati oleh pengguna aktif berdasarkan kesamaan preferensi dengan pengguna lain [18, 222, 331, 336]. Namun, algoritma CF berbasis memori klasik (seperti *K-Nearest Neighbors* / KNN) menghadapi hambatan kritis terkait **skalabilitas** dan **latensi inferensi** [223, 234, 243]. Ketika jumlah pengguna ($m$) dan item ($n$) mencapai jutaan, komputasi kemiripan matriks interaksi menjadi sangat berat dengan kompleksitas waktu sebesar $O(m \times n)$ [224, 234]. Hal ini memicu latensi respons yang tinggi (>2 detik), padahal studi industri menunjukkan setiap penambahan latensi sebesar 1 detik dapat menurunkan kepuasan pengguna sebesar 7% [225, 234].

Untuk mengatasi masalah tersebut, pada penelitian Ujian Tengah Semester (UTS) sebelumnya telah diusulkan metode **Neural Collaborative Filtering (NCF)** [226, 232]. NCF menggantikan operasi *dot-product* linier pada *Matrix Factorization* (MF) tradisional dengan arsitektur jaringan saraf dalam (*deep neural network*) yang terdiri dari *Embedding Layer* dan *Multi-layer Perceptron* (MLP) [10, 226, 227, 244]. Langkah ini secara fundamental mereduksi masalah skalabilitas karena inferensi dilakukan pada ruang embedding berdimensi rendah ($64$ hingga $256$ dimensi) dengan latensi yang sangat rendah dalam hitungan milidetik [228, 236]. Selain itu, MLP mampu menangkap interaksi non-linear yang kompleks antara pengguna dan item, yang dibuktikan dengan peningkatan Hit Rate (HR@10) sebesar 4,93% dan Normalized Discounted Cumulative Gain (NDCG@10) sebesar 4,08% pada dataset benchmark [227, 230, 237].

Meskipun model *deep learning* seperti NCF memiliki kinerja operasional yang sangat baik, model ini menderita kelemahan bawaan yang kritis: **bersifat sangat lapar data (*data-hungry*)** [30, 256]. Kinerja akurasi NCF merosot tajam ketika berhadapan dengan situasi **kelangkaan data (*data sparsity*)** dan **pengguna/item baru (*cold start*)** [56, 117, 219, 338]. Pada platform e-commerce nyata, matriks interaksi pengguna-item umumnya sangat kosong (tingkat kekosongan mencapai lebih dari 99%) [224, 235, 336]. Matriks yang sangat kosong ini mengakibatkan metrik evaluasi akurasi seperti *Root Mean Square Error* (RMSE) meningkat secara signifikan dan metrik *Precision@K* serta *Recall@K* menurun tajam karena model tidak memiliki cukup data sampel untuk melatih parameter jaringan sarafnya yang melimpah [225, 234, 235, 345].

Sebagai solusi mutakhir untuk mengatasi hambatan kelangkaan data tersebut, diusulkan integrasi teknologi **Generative AI** untuk melakukan augmentasi data (*data augmentation*) [2, 219, 240, 246]. Jurnal bereputasi internasional dari Springer (*Applied Intelligence*, 2024) oleh Jesús Bobadilla & Abraham Gutiérrez mengenalkan metode arsitektur **Wasserstein Generative Adversarial Networks for Recommender Systems (WGANRS)** [2, 3, 216]. WGANRS dirancang khusus untuk menghasilkan dataset penyaringan kolaboratif sintetis yang padat dan berkualitas tinggi guna mensimulasikan berbagai skenario stres operasional dan menyediakan data latih yang melimpah [3, 240, 256]. 

Berbeda dengan model GAN standar yang mengambil data mentah (*raw data*) yang sangat kosong dan diskret sehingga rentan mengalami kegagalan pelatihan berupa **Mode Collapse** (kondisi di mana generator menghasilkan data tiruan yang seragam dan tidak bervariasi) [248, 251, 252, 291], WGANRS beroperasi dengan melatih generator pada ruang laten embedding kontinu yang dense dan kompak yang dihasilkan oleh model kompresi *Deep Matrix Factorization* (DeepMF) [240, 253, 256, 257]. Lebih lanjut, WGANRS mengadopsi fungsi kerugian berbasis *Wasserstein Distance* (Earth Mover's Distance) dengan batasan 1-Lipschitz (melalui *weight clipping*) untuk melatih model penilai (*critic*) [3, 251, 276]. Eksperimen ilmiah membuktikan bahwa pendekatan ini mampu mengurangi *mode collapse* secara drastis, sehingga meningkatkan ukuran dataset sintetis valid yang bervariasi hingga ~200% dibandingkan baseline GAN standar (213% pada variasi pengguna dan 191% pada variasi item) [292].

Oleh karena itu, proposal Tugas Akhir ini mengusulkan sebuah model hibrida mutakhir yang mengintegrasikan **WGANRS sebagai mesin generator data sintetis padat** dan **NCF sebagai model rekomendasi akhir**. Dengan menggabungkan kedua metode AI ini, sistem rekomendasi diharapkan mampu mempertahankan akurasi prediksi yang tinggi dan latensi inferensi yang rendah, bahkan dalam kondisi matriks interaksi riil yang sangat kosong atau situasi pengguna/item baru (*cold start*).

---

### 1.2 Rumusan Masalah
Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana merancang arsitektur diagram alir integrasi metode Generative AI (WGANRS) dan deep learning (NCF) untuk mengatasi masalah kelangkaan data (*data sparsity*) pada sistem rekomendasi *e-commerce*?
2. Bagaimana cara kerja operasional pemrosesan data dari bentuk matriks interaksi riil yang kosong (*sparse*) melalui tahap kompresi embedding, generasi data sintetis, dekompresi/diskretisasi, hingga pelatihan model NCF?
3. Bagaimana merumuskan fungsi kerugian Wasserstein distance dan batasan 1-Lipschitz secara matematis untuk meminimalkan fenomena *mode collapse* pada proses pembuatan profil pengguna sintetis?
4. Sejauh mana integrasi WGANRS mampu meningkatkan volume dataset sintetis valid dan menjaga kemiripan distribusi rating riil dibandingkan dengan model GAN standar?

---

### 1.3 Batasan Masalah
Penelitian ini dibatasi oleh beberapa ruang lingkup berikut:
1. **Dataset Pengujian:** Implementasi dan pengujian model menggunakan dataset benchmark terbuka yang diakui secara internasional, yaitu **MovieLens 1M** (mengandung 1.000.209 rating dari 6.040 pengguna pada 3.900 item) [282, 283].
2. **Karakter Data:** Fokus pada data interaksi eksplisit (*explicit feedback*) berupa nilai rating berskala integer 1 hingga 5 bintang [266, 331].
3. **Arsitektur Generatif:** Arsitektur generative yang digunakan adalah modifikasi Wasserstein GAN dengan pembatasan bobot (*weight clipping*) untuk menjamin kondisi 1-Lipschitz [3, 251, 276].
4. **Model Rekomendasi:** Model rekomendasi akhir dibatasi pada arsitektur Neural Collaborative Filtering (NCF) yang menggabungkan General Matrix Factorization (GMF) dan Multi-layer Perceptron (MLP) [227, 236].

---

### 1.4 Tujuan Penelitian
Tujuan yang ingin dicapai melalui penelitian Tugas Akhir ini adalah:
1. Merancang arsitektur sistem informasi rekomendasi terintegrasi yang memanfaatkan kekuatan generatif WGANRS untuk melakukan augmentasi data dan kekuatan prediktif NCF untuk menyajikan rekomendasi non-linear [3, 226, 227].
2. Mengimplementasikan alur kerja otomatisasi data dari pemrosesan awal, kompresi spasial embedding (*DeepMF Coder*), generasi berbasis *WGAN Kernel*, hingga pemetaan kembali ke ID diskret menggunakan algoritma *K-Means Decoder* [3, 257, 263].
3. Membuktikan secara empiris keunggulan metode usulan WGANRS dalam meminimalkan *mode collapse* sehingga mampu menghasilkan ragam data latih sintetis yang lebih bervariasi dan bervolume lebih besar dibandingkan model GAN konvensional [251, 292].
4. Mengevaluasi metrik akurasi rekomendasi akhir (Precision@K, Recall@K, Hit Rate HR@10, dan NDCG@10) untuk memastikan kualitas data sintetis yang dihasilkan memiliki validitas pola yang setara dengan data riil [234, 285].

---

### 1.5 Manfaat Penelitian
Hasil dari penelitian ini diharapkan memberikan kontribusi sebagai berikut:
1. **Manfaat Akademis:** Memberikan kontribusi ilmiah dalam literatur penggabungan metode Generative AI dan sistem rekomendasi berbasis *deep learning*, khususnya mengenai metodologi penanganan kelangkaan data menggunakan representasi laten kontinu [30, 240, 256].
2. **Manfaat Praktis bagi Pengembang Sistem:** Menyediakan panduan implementasi kode pemrograman dan arsitektur yang siap pakai untuk merekayasa data sintetis berkualitas tinggi guna menguji ketangguhan sistem rekomendasi e-commerce dalam menghadapi lonjakan drastis jumlah pengguna atau item (*stress testing*) [240, 245, 256].
3. **Manfaat Sosial Ekonomi:** Meningkatkan relevansi rekomendasi produk pada platform e-commerce yang baru dirintis (yang masih kekurangan data interaksi), sehingga membantu meningkatkan omzet penjualan platform dan kepuasan pengalaman belanja pengguna [7, 225, 329].

---

## BAB II: TINJAUAN PUSTAKA

### 2.1 Sistem Rekomendasi dan Collaborative Filtering (CF) Klasik
Sistem rekomendasi didefinisikan sebagai sistem informasi yang memperkirakan preferensi pengguna terhadap item tertentu dan secara proaktif menyajikan item yang paling relevan [18, 122]. Secara umum, sistem rekomendasi dibagi menjadi tiga kategori utama berdasarkan jenis data input yang digunakan: *Collaborative Filtering* (CF), *Content-Based* (CB), dan model Hibrida (*Hybrid*) [18, 242, 333].

*Collaborative Filtering* (CF) adalah metode yang paling banyak diterapkan karena tidak memerlukan informasi eksternal (meta-data) mengenai pengguna maupun item [336]. CF murni mengandalkan matriks interaksi historis pengguna-item [336]. CF klasik terbagi menjadi dua kelompok:
1. **Memory-Based (Neighborhood-Based):** Menggunakan kesamaan antar-pengguna (*User-Based*) atau antar-item (*Item-Based*) menggunakan algoritma KNN [9, 223, 243]. Kelemahannya adalah kompleksitas komputasi yang tinggi dan ketidakmampuan beroperasi pada data yang sangat renggang (*sparse matrix*) [224, 243].
2. **Model-Based (Matrix Factorization):** Mengompresi matriks interaksi $R$ berdimensi besar menjadi dua matriks faktor laten berdimensi rendah $P \in \mathbb{R}^{M \times k}$ (untuk mewakili pengguna) dan $Q \in \mathbb{R}^{N \times k}$ (untuk mewakili item), di mana rating prediksi dihitung melalui operasi perkalian titik linier (*dot-product*) [10, 19, 243, 337]:
   $$\hat{r}_{ui} = p_u^T q_i$$
   Meskipun MF sangat sukses, asumi linearitas perkalian titik membatasi kapasitas ekspresif model dalam menangkap hubungan non-linear yang rumit antara pengguna dan item [27, 226].

---

### 2.2 Neural Collaborative Filtering (NCF)
Neural Collaborative Filtering (NCF) yang diajukan oleh He et al. (2017) mengatasi batasan linearitas MF dengan menggunakan arsitektur jaringan saraf dalam untuk memodelkan interaksi pengguna-item [53, 226]. Arsitektur NCF terdiri dari empat lapisan utama [236]:
1. **Input Layer:** Menerima representasi *one-hot encoding* dari ID pengguna ($u$) dan ID item ($i$) [40, 236].
2. **Embedding Layer:** Lapisan proyeksi yang mengonversi vektor *one-hot* berdimensi tinggi menjadi vektor representasi kontinu berdimensi rendah (latent vectors) [227, 236].
3. **Neural CF Layers (MLP):** Lapisan tersembunyi (*hidden layers*) dengan fungsi aktivasi non-linear (seperti ReLU) yang menumpuk transformasi non-linear untuk mempelajari representasi interaksi yang kompleks [21, 27, 236].
4. **Output Layer:** Memprediksi skor relevansi $\hat{y}_{ui}$ menggunakan fungsi aktivasi Sigmoid untuk mengembalikan nilai dalam rentang $[0,1]$ untuk umpan balik implisit atau rating ternormalisasi [227, 236]:
   $$\hat{y}_{ui} = \sigma\left(\phi_{out}\left(\phi_L\left(\dots\phi_1([p_u \oplus q_i])\dots\right)\right)\right)$$
   di mana $\oplus$ adalah operasi penggabungan (*concatenation*) atau perkalian elemen (*element-wise product*) antara embedding pengguna $p_u$ dan item $q_i$, serta $\phi$ merupakan lapisan saraf [236].

---

### 2.3 Fenomena Mode Collapse pada Generative Adversarial Networks (GAN)
Generative Adversarial Networks (GAN) adalah paradigma generative AI yang terdiri dari dua model jaringan saraf yang saling bersaing dalam permainan minimax [22, 274]:
*   **Generator ($G$):** Mencoba menghasilkan sampel data sintetis realistis dari vektor kebisingan acak (*stochastic noise*) $z \sim p_{\text{latent}}$ untuk mengelabui Diskriminator [106, 274].
*   **Discriminator ($D$):** Mencoba membedakan apakah sampel input berasal dari data riil ($x \sim p_{\text{data}}$) atau dari Generator ($G(z)$) [106, 274].

Formulasi permainan minimax GAN standar didefinisikan sebagai [274]:
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_{z}}[\log(1 - D(G(z)))]$$

Hambatan utama dalam melatih GAN standar pada data sistem rekomendasi adalah **Mode Collapse** [30, 240, 252]. Mode collapse terjadi ketika Generator menemukan sejumlah kecil pola data tiruan yang mampu mengelabui Diskriminator dengan mudah, sehingga Generator terus-menerus memproduksi profil tiruan yang seragam dan repetitif [251, 252, 291]. Pada sistem rekomendasi, hal ini berarti Generator hanya menghasilkan profil pengguna sintetis dengan selera item yang sangat seragam (misalnya, hanya merekomendasikan item populer yang sama), sehingga membatasi diversifikasi dataset sintetis yang dihasilkan [252, 291].

---

### 2.4 Wasserstein Generative Adversarial Networks untuk Sistem Rekomendasi (WGANRS)
Untuk mengatasi masalah ketidakstabilan pelatihan dan mode collapse pada GAN tradisional, Arsitektur **WGANRS** mengadopsi konsep *Wasserstein GAN* (WGAN) [3, 27, 248]. Perubahan mendasar pada WGANRS meliputi:
1.  **Penggunaan Wasserstein Distance (Earth Mover's Distance):** Berfungsi mengukur jarak antara distribusi riil ($P_r$) dan distribusi sintetis ($P_g$) [251, 276]. Keunggulan jarak ini adalah ia tetap kontinu dan dapat diturunkan (*differentiable*) bahkan ketika kedua distribusi tidak saling tumpang tindih, sehingga memberikan gradien yang stabil sepanjang proses pelatihan [251, 253].
2.  **Model Critic:** Menggantikan peran Diskriminator klasik [251, 276]. Jaringan penilai (*critic*) $f_w$ bertugas memperkirakan nilai jarak Wasserstein tanpa melakukan klasifikasi biner beraliran probabilitas [251, 276].
3.  **Batasan 1-Lipschitz:** Untuk memastikan fungsi penilai memenuhi syarat kedekatan metrik Wasserstein, parameter bobot *critic* dibatasi agar bernilai kompak, yang secara praktis dilakukan melalui pembatasan nilai bobot (*weight clipping*) atau penalti gradien (*gradient penalty*) [3, 251, 276].

Melalui WGANRS, Generator dipaksa untuk memproduksi embedding pengguna dan item yang sangat bervariasi, yang secara dramatis meminimalkan duplikasi sampel pasca-diskritsasi dan meningkatkan volume representasi unik [251, 290].

---

### 2.5 Evaluasi Kinerja Sistem Informasi
Kinerja sistem rekomendasi diukur menggunakan beberapa dimensi ukuran kinerja:
1.  **Metrik Akurasi Rating:** Menggunakan *Root Mean Square Error* (RMSE) dan *Mean Absolute Error* (MAE) pada data pengujian untuk melihat seberapa dekat nilai prediksi rating dengan rating sebenarnya [225, 334, 343].
2.  **Metrik Akurasi Klasifikasi / Rekomendasi Top-N:**
    *   **Precision@K:** Proporsi item relevan yang direkomendasikan dalam daftar Top-K [234, 345, 363].
    *   **Recall@K:** Proporsi item relevan dari data uji yang berhasil disajikan dalam daftar Top-K [234, 345, 363].
    *   **Hit Rate (HR@K):** Probabilitas sistem menampilkan setidaknya satu item relevan dalam daftar Top-K [230, 234].
    *   **Normalized Discounted Cumulative Gain (NDCG@K):** Mengukur relevansi item dengan memperhitungkan posisi penempatan item dalam daftar rekomendasi (item yang lebih relevan harus berada di posisi teratas) [230, 234, 334].
3.  **Metrik Keberagaman Generatif:** Diukur dari jumlah sampel unik pasca-eliminasi data duplikat (*deleted profiles*) [264, 292]. Semakin sedikit sampel sintetis yang terbuang saat diskretisasi, semakin tinggi kemampuan model generatif dalam menangani mode collapse [290, 292].

---

## BAB III: METODOLOGI PENELITIAN

### 3.1 Diagram Alir dan Alur Kerja Sistem Usulan
Model arsitektur hibrida usulan (WGANRS + NCF) bekerja dalam lima tahapan proses sekuensial yang saling berkesinambungan [256]. Alur sistem secara keseluruhan digambarkan pada diagram berikut:

```
[ Data Interaksi Riil (Sparse) ] 
               │
               ▼
   [ Tahap 1: DeepMF Coder ] ───► Ekstraksi Embedding Pengguna & Item (Dense & Kontinu)
               │
               ▼
   [ Tahap 2: WGAN Kernel ]  ───► Generasi Embedding Sintetis Kontinu via Wasserstein Loss
               │
               ▼
[ Tahap 3: K-Means Decoder ] ───► Dekompresi & Diskretisasi ke ID Pengguna & Item Riil
               │
               ▼
 [ Dataset Sintetis (Padat) ] ───► Eliminasi Duplikasi Sampel Tabrakan (Collision Removal)
               │
               ▼
   [ Tahap 4: Pelatihan NCF ]───► Optimasi Parameter Embedding & MLP Rekomendasi Akhir
               │
               ▼
[ Evaluasi Kinerja (Top-N) ] ───► HR@10, NDCG@10, Precision@K, Recall@K
```

Diagram alir di atas secara eksplisit menggambarkan transisi dari data mentah riil yang sangat kosong (*sparse raw data*) [252], dikompresi menjadi representasi dense kontinu di *latent space* [256], digenerasikan secara stabil oleh generator generatif berbasis Wasserstein distance [3, 277], dipetakan kembali ke format diskret yang valid [261, 281], dan akhirnya digunakan untuk melatih model penentu keputusan rekomendasi (NCF) [256].

---

### 3.2 Tahapan Operasional Sistem

Pemrosesan operasional sistem usulan dibagi menjadi tahapan terperinci berikut:

#### **Tahap 1: Kompresi Spasial Data (DeepMF Coder)**
*   **Proses Input:** Mengambil dataset penyaringan kolaboratif riil $S$ yang sangat renggang (*sparse*) [256, 273].
*   **Proses Inti:** Dataset diumpankan ke model jaringan saraf dalam *Deep Matrix Factorization* (DeepMF) [244, 257]. DeepMF dilatih menggunakan fungsi optimasi kesalahan kuadrat (*Mean Squared Error*) [269, 271]. Setelah pelatihan konvergen, ID pengguna ($u$) dimasukkan secara umpan maju (*feedforward*) ke lapisan embedding pengguna untuk mengekstrak representasi vektor laten kontinu $\vec{e}_u = f^{eu}(u)$ berukuran dimensi kecil $E$ (umumnya bernilai $5$ hingga $15$ neuron) [258, 272]. Hal yang sama dilakukan untuk mengekstrak embedding item $\vec{e}_i = f^{ei}(i)$ [258, 272].
*   **Proses Output:** Menghasilkan matriks embedding terkompresi $U \times E$ untuk pengguna dan $I \times E$ untuk item [258]. Representasi ini mendefinisikan dataset baru berbasis embedding $R$ yang kontinu, dense, dan tidak kosong [256, 273].

#### **Tahap 2: Generasi Data Sintetis (WGAN Kernel)**
*   **Proses Input:** Matriks embedding kontinu $R$ bertindak sebagai distribusi referensi riil [259, 273, 277]. Generator ($G$) menerima vektor masukan berupa *stochastic noise* Gaussian $z$ [3, 260].
*   **Proses Inti:** Generator $G$ dilatih bersama penilai (*critic* / $D$) menggunakan fungsi kerugian Wasserstein distance [3, 251]. Penilai memperkirakan *Earth Mover's Distance* untuk mengevaluasi seberapa realistis embedding tiruan yang dihasilkan [251]. Batasan kekompakan parameter 1-Lipschitz ditegakkan melalui *weight clipping* pada rentang $[ -c, c ]$ di setiap langkah pembaruan bobot penilai [3, 251].
*   **Proses Output:** Generator yang telah terlatih digunakan secara mandiri untuk menghasilkan ribuan profil embedding sintetis baru berupa tuple representasi kontinu: $\langle \text{user\_embedding}, \text{item\_embedding}, \text{rating} \rangle$ [260, 261].

#### **Tahap 3: Dekompresi dan Diskretisasi (K-Means Decoder)**
*   **Proses Input:** Batches dari tuple embedding sintetis kontinu yang dihasilkan pada Tahap 2 [261].
*   **Proses Inti:** Algoritma clustering **K-Means** diterapkan secara terpisah pada ruang embedding pengguna riil untuk menetapkan $K^*$ buah centroid cluster, dan pada ruang embedding item riil untuk menetapkan $K^{**}$ buah centroid cluster [263, 278, 279]. Untuk setiap embedding pengguna sintetis kontinu yang dihasilkan Generator, fungsi penjelajah $h^*(u)$ menghitung jarak Euclidean terdekat ke centroid cluster pengguna dan mengonversinya menjadi ID diskret pengguna $c \in \{1,\dots,K^*\}$ [263, 280]. Proses yang sama dilakukan oleh fungsi $h^{**}(i)$ untuk mengonversi embedding item sintetis menjadi ID diskret item $c \in \{1,\dots,K^{**}\}$ [263, 280].
*   **Proses Output:** Menghasilkan dataset diskret sintetis baru $H$ yang berisi baris bernilai integer $\langle \text{user\_ID}, \text{item\_ID}, \text{rating} \rangle$ [261, 280, 281].

#### **Tahap 4: Eliminasi Duplikasi Pola (Collision Removal)**
*   Akibat pemetaan dari representasi kontinu ke integer diskret, sangat mungkin terjadi "tabrakan" pola di mana baris sintetis menghasilkan kombinasi ID pengguna dan ID item yang sama namun dengan rating berbeda [264, 281].
*   Sistem secara otomatis mendeteksi tabrakan ini menggunakan fungsi eliminasi $G'$ untuk menyaring dan menghapus baris duplikat, menyisakan hanya baris unik berkualitas tinggi untuk membentuk dataset akhir $S^*$ [264, 282].

#### **Tahap 5: Pelatihan Rekomendasi Akhir (NCF Training)**
*   Dataset sintetis padat hasil augmentasi $S^*$ digunakan sebagai input latih utama untuk melatih model **Neural Collaborative Filtering (NCF)** [256].
*   NCF dilatih menggunakan lapisan MLP non-linear untuk mengonstruksi pembuat rekomendasi yang sangat akurat dan tangguh terhadap kelangkaan data eksternal platform [227, 236].

---

### 3.3 Formulasi Matematika Model Usulan

Pemodelan formal matematis arsitektur terintegrasi dirumuskan melalui persamaan-persamaan berikut:

#### **1. Kompresi Spasial DeepMF**
Fungsi kompresi untuk memetakan ID pengguna $u$ dan ID item $i$ ke dalam representasi kontinu berdimensi $E$ dirumuskan sebagai [268, 269]:
$$f^{eu}(u) = \vec{e}_u = [e_{u,0}, e_{u,1}, \dots, e_{u,E}] \in \mathbb{R}^{E+1}$$
$$f^{ei}(i) = \vec{e}_i = [e_{i,0}, e_{i,1}, \dots, e_{i,E}] \in \mathbb{R}^{E+1}$$

Di mana nilai taksiran rating prediksi $\hat{y}_j$ pada tahap prapelatihan DeepMF dihitung menggunakan perkalian titik dari representasi spasial tersebut [270, 271]:
$$\hat{y}_j = f^{eu}(u) \cdot f^{ei}(i) = \vec{e}_u \cdot \vec{e}_i = \sum_{d=0}^{E} e_{u,d} \cdot e_{i,d}$$

Optimasi bobot parameter jaringan DeepMF dikendalikan oleh fungsi penilai kesalahan kuadrat terkecil (*least square error loss*) [271]:
$$\min_{\Theta} \mathcal{L}_{\text{DeepMF}} = \frac{1}{2} \sum_{j \in S} \left(y_j - \hat{y}_j\right)^2$$

#### **2. Optimasi Generatif WGANRS**
Fungsi tujuan permainan adversarial minimax berbasis Wasserstein distance dengan fungsionalitas penilai (*critic*) $f_w$ dirumuskan sebagai [276, 277]:
$$\min_G \max_{D \in \mathcal{D}} f(D, G) = \mathbb{E}_{R \sim p_{\text{data}}}[f_w(R)] + \mathbb{E}_{z \sim p_{z}}[f_w(G(z))]$$

Di mana $R$ mewakili distribusi dataset spasial kontinu hasil kompresi DeepMF [259, 273, 277], $z$ merupakan kebisingan Gaussian [3, 260], dan $f_w$ harus memenuhi kondisi batasan 1-Lipschitz kontinu [3, 276]:
$$\left| f_w(x_1) - f_w(x_2) \right| \leq \| x_1 - x_2 \|, \quad \forall x_1, x_2 \in \text{latent space}$$

#### **3. Dekompresi dan Diskretisasi K-Means**
Fungsi pemetaan centroid K-Means untuk mentranslasikan koordinat vektor pengguna sintetis kontinu $\vec{e}_{u,\text{fake}}$ ke ID integer diskret $c \in \{1,\dots,K^*\}$ menggunakan pencarian jarak minimum dirumuskan sebagai [263, 280]:
$$h^*(u) = \arg\min_{c \in \{1,\dots,K^*\}} \| \vec{e}_{u,\text{fake}} - \vec{\mu}_c \|^2$$

Di mana $\vec{\mu}_c$ mewakili koordinat vektor centroid cluster pengguna ke-$c$ [263, 280]. Proses translasi ID item diskret $c \in \{1,\dots,K^{**}\}$ didefinisikan secara analog [263, 280]:
$$h^{**}(i) = \arg\min_{c \in \{1,\dots,K^{**}\}} \| \vec{e}_{i,\text{fake}} - \vec{\nu}_c \|^2$$

Di mana $\vec{\nu}_c$ mewakili koordinat vektor centroid cluster item ke-$c$ [263, 280].

#### **4. Pelatihan Akhir Model NCF**
Fungsi prediksi rating akhir $\hat{r}_{ui}$ oleh arsitektur NCF dirumuskan sebagai kombinasi non-linear MLP [236]:
$$\hat{r}_{ui} = \sigma\left(W^T \cdot \phi_{\text{MLP}}([p_u \oplus q_i]) + b\right)$$

Pengoptimalan parameter model NCF dilakukan dengan meminimalkan nilai fungsi kehilangan entropi silang biner (*binary cross-entropy loss*) [41]:
$$\mathcal{L}_{\text{NCF}} = -\sum_{(u,i) \in \mathcal{O} \cup \mathcal{O}^-} r_{ui} \log \hat{r}_{ui} + (1 - r_{ui}) \log (1 - \hat{r}_{ui})$$

---

### 3.4 Desain Pengujian dan Metrik Evaluasi Kinerja
Eksperimen evaluasi dilakukan dengan membandingkan tiga jenis dataset secara komparatif [284, 286]:
1.  **Dataset Riil (MovieLens 1M):** Sebagai acuan batas atas (*ground-truth* / ML) [282, 294].
2.  **Dataset Sintetis Baseline GANRS:** Dataset sintetis yang dihasilkan model GAN standar [284, 291].
3.  **Dataset Sintetis WGANRS (Metode Usulan):** Dataset sintetis hasil usulan penelitian [284, 291].

Evaluasi performa akan mengukur empat aspek kriteria pengujian berikut:

#### **1. Volume Sampel Unik Pasca-Diskretisasi (Mode Collapse Check)**
Sifat ketahanan model generatif terhadap mode collapse dipantau melalui jumlah volume sampel riil non-duplikat yang dihasilkan setelah proses eliminasi tabrakan [290, 292]. Metrik ukuran keberhasilan dirumuskan sebagai:
$$\text{Yield Ratio} = \frac{|S^*|}{|S_{\text{initial}}|}$$
WGANRS ditargetkan menghasilkan peningkatan volume sampel bervariasi hingga ~200% dibanding baseline GAN standar [292].

#### **2. Kemiripan Distribusi Rating (Rating Distribution Fit)**
Mengukur kemiripan statistik kurva distribusi frekuensi rating hasil generasi dibandingkan dengan data riil MovieLens menggunakan evaluasi visual histogram dan perhitungan selisih persentase deviasi rating bintang 4 dan 5 [285, 294]. Model usulan WGANRS ditargetkan mencapai kecocokan bentuk kurva Gaussian yang hampir sempurna (peningkatan akurasi kecocokan sebesar ~271.21% dibandingkan GAN standar) [294, 295].

#### **3. Distribusi Kepadatan Rating Pengguna dan Item**
Kurva kepadatan (*density curve*) rating per pengguna dan item diplot untuk memverifikasi bahwa model usulan tidak hanya menghasilkan rata-rata yang mirip, tetapi juga memiliki variansi standar deviasi yang menyebar secara alami menyamai bentuk persebaran alami dataset MovieLens [285, 295].

#### **4. Akurasi Rekomendasi Top-N (Precision & Recall)**
Data sintetis yang dihasilkan oleh GANRS dan WGANRS digunakan sebagai data latih untuk melatih model NCF [296]. Selanjutnya, NCF diuji pada data uji riil untuk mengukur performa metrik akurasi berikut dengan variasi nilai rekomendasi $N = 2, 4, 6, 8, 10$ [296, 302]:
*   **Precision@N:**
    $$\text{Precision@N} = \frac{|\text{Item Relevan} \cap \text{Top-N Rekomendasi}|}{N}$$
*   **Recall@N:**
    $$\text{Recall@N} = \frac{|\text{Item Relevan} \cap \text{Top-N Rekomendasi}|}{|\text{Total Item Relevan Pengguna}|}$$

Jika kurva evolusi Precision dan Recall dari model hasil latih WGANRS menunjukkan tren pergerakan yang selaras dan berhimpitan dengan kurva hasil latih data riil MovieLens, hal ini memberikan justifikasi ilmiah kuat bahwa pola-pola laten data riil berhasil direplikasi secara sempurna oleh sistem kecerdasan buatan generatif usulan [289, 296, 297].

---

## DAFTAR PUSTAKA

[1] Bobadilla, J., & Gutiérrez, A. (2024). Wasserstein GAN-based architecture to generate collaborative filtering synthetic datasets. *Applied Intelligence*, 54(1), 2472–2490. https://doi.org/10.1007/s10489-024-05313-4 [216, 240, 243].

[2] He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW 2017)*, 173–182. https://doi.org/10.1145/3038912.3052569 [228, 236, 444].

[3] Munson, J., Cuezze, T., Nesar, S., & Zosso, D. (2025). A review of large language models and the recommendation task. *Discover Artificial Intelligence*, 5(203), 1–25. https://doi.org/10.1007/s44163-025-00334-5 [218, 328].

[4] Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys (CSUR)*, 52(1), 1–38. https://doi.org/10.1145/3285029 [222, 232, 568].
