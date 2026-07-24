# Linguistic and Language Pattern Comparison (for clickbait news headline)

Penelitian ini bertujuan untuk melakukan klasifikasi judul berita clickbait dan non-clickbait berbahasa Indonesia menggunakan beberapa algoritma machine learning dan deep learning.

Project ini membandingkan performa model tradisional (shallow learning) seperti Naive Bayes dan Support Vector Machine (SVM) dengan model berbasis Transformer, yaitu IndoBERT, pada dataset berita berbahasa Indonesia.

## Latar Belakang

Perkembangan media online menyebabkan penggunaan judul clickbait semakin meningkat. Judul clickbait dirancang untuk menarik perhatian pembaca, namun sering kali tidak sepenuhnya merepresentasikan isi berita.

Dengan memanfaatkan teknik Natural Language Processing (NLP), penelitian ini mencoba mengidentifikasi pola linguistik pada judul berita serta membandingkan performa beberapa algoritma klasifikasi dalam mendeteksi clickbait. Pendekatan NLP memungkinkan komputer memahami dan memproses bahasa manusia secara otomatis.

## Tujuan Penelitian

- Mengklasifikasikan judul berita menjadi kategori clickbait dan non-clickbait.
- Membandingkan performa algoritma Naive Bayes, SVM, dan IndoBERT.
- Menganalisis pola linguistik yang muncul pada judul berita clickbait.
- Mengevaluasi efektivitas pendekatan shallow learning dan deep learning pada kasus klasifikasi clickbait.

## Metode yang Digunakan

### Preprocessing

Tahapan preprocessing yang digunakan meliputi:

- Case Folding
- Cleaning
- Tokenization
- Stopword Removal
- Normalization

### Feature Extraction

- TF-IDF Vectorization (untuk Naive Bayes dan SVM)
- Transformer Embedding (untuk IndoBERT)

### Algoritma

- Multinomial Naive Bayes
- Support Vector Machine (Linear SVM)
- IndoBERT

Naive Bayes dan SVM merupakan algoritma klasifikasi yang banyak digunakan pada text classification karena sederhana dan efektif. Sementara itu, BERT merupakan model Transformer yang saat ini banyak digunakan pada berbagai tugas NLP.

## Dataset

Dataset yang digunakan berasal dari beberapa sumber dataset berita berbahasa Indonesia yang telah melalui proses pembersihan dan pelabelan.

Kelas yang digunakan:

- Clickbait
- Non-Clickbait

## Evaluasi

Evaluasi model dilakukan menggunakan beberapa metrik:

- Accuracy
- Precision
- Recall
- F1-Score

## Struktur Project

```bash
├── data/
├── notebooks/
├── models/
├── results/
├── preprocessing/
├── training/
├── evaluation/
└── README.md
```

## Tools dan Library

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Transformers
- Hugging Face
- Matplotlib
- Seaborn

## Hasil

Hasil penelitian menunjukkan bahwa model berbasis Transformer secara konsisten mengungguli metode shallow learning pada tugas klasifikasi clickbait bahasa Indonesia:

| Model | Accuracy | F1-Score |
|---|---|---|
| IndoBERT-p1 (best single model) | 95.39% | 94.88% |
| Weighted Ensemble (IndoBERT-p1 + p2 + XLM-RoBERTa) | 92.47% | - |
| Naive Bayes / SVM (TF-IDF) | lebih rendah dari model Transformer | - |

IndoBERT-p1 menjadi model dengan performa terbaik secara individual, mengungguli baseline shallow learning maupun kombinasi ensemble.

## Status Publikasi

Penelitian ini telah diterima (accepted) di **JOISER** dengan Letter of Acceptance (LOA), dan saat ini berada pada tahap copy editing menuju publikasi.

## Cara Menjalankan

Clone repository:

```bash
git clone https://github.com/kwan6/Linguistic-and-Language-Pattern-Comparison.git
```

Masuk ke folder project:

```bash
cd Linguistic-and-Language-Pattern-Comparison
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan notebook atau script sesuai kebutuhan.

## Penulis

**Muhammad Noer Attalah Dzahkwan**

Universitas Amikom Yogyakarta

## Lisensi

Project ini dibuat untuk keperluan penelitian dan akademik.
