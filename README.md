# UAS Deep Learning – NLP with Transformers

## Purpose of the Repository

Repositori ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS)** mata kuliah **Deep Learning**, dengan fokus menerapkan dan mengeksplorasi model **Transformer** pada berbagai task **Natural Language Processing (NLP)**.

## Project Overview

Proyek ini berisi beberapa eksperimen fine-tuning model pre-trained dari HuggingFace Transformers pada beberapa dataset NLP populer, mencakup:

- Text summarization (XSUM).
- Sequence-to-sequence question answering.
- News topic classification (AG News).
- Multi-label emotion classification (GoEmotions).
- Natural Language Inference (MNLI).

Setiap eksperimen didokumentasikan dalam Jupyter Notebook, mulai dari loading data, preprocessing, training, evaluasi, hingga demo inference.

## Models and Metrics

### Ringkasan Model & Metrik

| Task / Notebook          | Dataset   | Jenis Model / Tipe Task              | Metrik Utama (contoh hasil)                                                 |
|--------------------------|-----------|--------------------------------------|------------------------------------------------------------------------------|
| `xsum-DL.ipynb`          | XSUM      | Summarization (seq2seq Transformer)  | Training loss menurun, evaluasi via `Trainer.evaluate()` di test set.       |
| `Seq2seq-DL.ipynb`       | Custom QA | Seq2Seq QA (T5-style)                | Training loss per step (sekitar 0.25 di akhir), kualitas jawaban secara kualitatif. |
| `Ag_News_DL-1-1.ipynb`   | AG News   | Text classification                   | Accuracy ≈ 0.94–0.95, F1 Macro ≈ 0.94–0.95 di validation.                    |
| `Go_emotion_DL-2.ipynb`  | GoEmotions| Multi-label emotion classification    | F1 Micro naik dari ≈0.53 ke ≈0.57, F1 Macro ≈0.31→0.40, Exact Match ≈0.41→0.45. |
| `MNLI_DL-1.ipynb`        | MNLI      | Natural Language Inference (3 class) | Accuracy ≈0.83–0.84, F1 Micro/Macro hampir setara (≈0.83–0.84).             |

### Deskripsi Singkat per Notebook

- **`xsum-DL.ipynb` – Text Summarization**  
  Fine-tuning model summarization pada dataset XSUM untuk menghasilkan ringkasan satu kalimat dari teks berita, dengan monitoring training loss dan evaluasi menggunakan `Trainer.evaluate()`.

- **`Seq2seq-DL.ipynb` – Sequence-to-Sequence Question Answering**  
  Model seq2seq (T5-style) untuk menjawab pertanyaan berbasis konteks, dengan tabel training loss per step dan fungsi `ask_question(question, context)` untuk demo tanya-jawab.

- **`Ag_News_DL-1-1.ipynb` – AG News Classification**  
  Klasifikasi news topic AG News, menampilkan tabel per epoch (Training Loss, Validation Loss, Accuracy, F1 Macro) dan evaluasi akhir dengan metrik `eval_accuracy` dan `eval_f1_macro`.

- **`Go_emotion_DL-2.ipynb` – GoEmotions Emotion Classification**  
  Multi-label emotion classification dengan F1 Micro dan F1 Macro, analisis `trainer.state.log_history` menggunakan pandas, serta visualisasi training & validation loss dan validation F1.

- **`MNLI_DL-1.ipynb` – MNLI Natural Language Inference**  
  NLI tiga kelas (entailment, neutral, contradiction) dengan tabel metrik per epoch dan demo inference untuk beberapa pasangan premise–hypothesis.

## How to Navigate the Repository / Notebooks

Struktur utama (diasumsikan):

- `Notebook/`  
  - `xsum-DL.ipynb`  
  - `Seq2seq-DL.ipynb`  
  - `Ag_News_DL-1-1.ipynb`  
  - `Go_emotion_DL-2.ipynb`  
  - `MNLI_DL-1.ipynb`  

Rekomendasi alur baca/notebook:

1. Mulai dari **`Ag_News_DL-1-1.ipynb`** untuk melihat pipeline klasifikasi teks yang relatif sederhana.
2. Lanjut ke **`Go_emotion_DL-2.ipynb`** untuk contoh multi-label classification dan visualisasi metrik.
3. Baca **`MNLI_DL-1.ipynb`** untuk memahami task NLI dan penggunaan metrik F1 Micro/Macro.
4. Eksplor **`Seq2seq-DL.ipynb`** untuk task question answering dengan model seq2seq serta fungsi helper inference.
5. Terakhir, lihat **`xsum-DL.ipynb`** untuk task summarization yang lebih kompleks.

Untuk menjalankan:

```bash
cd Deeplearn-UAS
pip install -r requirements.txt
cd Notebook
jupyter notebook


Kelompok : 13
Anggota :
- Darryl Satria Wibowo
- Fakhriza Bondan P.
