# Deeplearn-UAS



\# UAS Deep Learning – NLP with Transformers



Repositori ini berisi kumpulan Jupyter Notebook untuk tugas akhir mata kuliah \*\*Deep Learning\*\*, dengan fokus pada berbagai task \*\*Natural Language Processing (NLP)\*\* menggunakan arsitektur \*\*Transformer\*\* dan library \*\*HuggingFace Transformers\*\*. 



\## Tujuan Proyek



Proyek ini dibuat untuk:

\- Menerapkan konsep deep learning modern (Transformer) pada beberapa task NLP berbeda (summarization, question answering, text classification, emotion classification, dan natural language inference). 

\- Mengeksplorasi proses \*\*fine-tuning\*\* model pre-trained, monitoring metrik training/evaluasi, dan membuat demo inference yang dapat dicoba secara interaktif. 

\- Mendokumentasikan workflow end-to-end mulai dari loading dataset, preprocessing, training, evaluasi, sampai visualisasi dan penggunaan model. 



\## Struktur Notebook



Semua notebook berada di folder `Notebook/` dengan daftar sebagai berikut. 



\### 1. `xsum-DL.ipynb` – Text Summarization (XSUM)



Notebook ini melakukan fine-tuning model summarization pada dataset \*\*XSUM\*\* untuk menghasilkan ringkasan satu kalimat dari artikel berita. 



Fitur utama:

\- Menggunakan model dan tokenizer dari HuggingFace untuk abstractive summarization. 

\- Training dengan `Trainer` dan menampilkan ringkasan training menggunakan objek `TrainOutput` yang berisi `global\_step`, `training\_loss`, `train\_runtime`, dan `total\_flos`. 

\- Evaluasi menggunakan `Trainer.evaluate()` di test set yang sudah di-tokenisasi (`test\_tok`) dengan konfigurasi `TrainingArguments` khusus untuk evaluasi. 



\### 2. `Seq2seq-DL.ipynb` – Sequence-to-Sequence Question Answering



Notebook ini mengimplementasikan model \*\*sequence-to-sequence\*\* (gaya T5) untuk tugas \*\*question answering berbasis konteks\*\*. 



Fitur utama:

\- Monitoring \*\*training loss per step\*\* dalam bentuk tabel (kolom `Step` dan `Training Loss`) hingga puluhan ribu step, serta ringkasan `TrainOutput` (misalnya `global\_step=21900`, `training\_loss≈0.254`). 

\- Fungsi helper `ask\_question(question, context)` yang:

&nbsp;- Menggabungkan pertanyaan dan konteks ke dalam format input T5 (`"question: ... context: ..."`). 

&nbsp;- Melakukan tokenisasi dan inference dengan `model.generate()` lalu melakukan decoding jawaban. \[file:12]

&nbsp;- Demo tanya-jawab dengan konteks tentang \*\*Gunung Tangkuban Parahu\*\* dan pertanyaan “Di mana letak Gunung Tangkuban Parahu?”, di mana model menghasilkan jawaban “utara Kota Bandung”. 



\### 3. `Ag\_News\_DL-1-1.ipynb` – News Topic Classification (AG News)



Notebook ini melakukan fine-tuning model klasifikasi teks pada dataset \*\*AG News\*\* untuk mengelompokkan berita ke dalam beberapa kategori topik. 



Fitur utama:

\- Training dan evaluasi model klasifikasi dengan `Trainer`. 

\- Tabel metrik per epoch yang memuat:

&nbsp; - `Epoch`, `Training Loss`, `Validation Loss`, `Accuracy`, dan `F1 Macro`. 

&nbsp; - Contoh hasil: akurasi validasi sekitar 0.94–0.95 dengan F1 macro yang sebanding. 

\- Ringkasan evaluasi akhir yang memuat `eval\_loss`, `eval\_accuracy`, `eval\_f1\_macro`, `eval\_runtime`, `eval\_samples\_per\_second`, dan `eval\_steps\_per\_second` untuk epoch terbaik. 



\### 4. `Go\_emotion\_DL-2.ipynb` – Multi-label Emotion Classification (GoEmotions)



Notebook ini melakukan fine-tuning model untuk \*\*multi-label emotion classification\*\* pada dataset \*\*GoEmotions\*\*. 



Fitur utama:

\- Mengambil `trainer.state.log\_history` dan mengonversinya menjadi `pandas.DataFrame` (`df\_logs`) lalu memisahkan:

&nbsp; - `df\_train` berisi baris dengan kolom `loss` (log training). 

&nbsp; - `df\_eval` berisi baris dengan kolom `eval\_loss` (log evaluasi). 

\- Menambahkan kolom `step` jika belum ada, lalu memvisualisasikan:

&nbsp; - Kurva \*\*Training Loss vs Validation Loss\*\* terhadap step. 

&nbsp; - Kurva metrik \*\*F1 micro\*\* (`eval\_f1\_micro`) yang diperlakukan sebagai “validation accuracy”. \[file:14]

\- Demo inference beberapa kalimat, misalnya:

&nbsp; - “I am so happy today!” → emosi dominan `joy` dengan skor sekitar 0.85. 

&nbsp; - “I feel really sad and disappointed.” → emosi `sadness` dengan skor tinggi. 

&nbsp; - “That was so rude and unfair.” → kombinasi `anger` dan `annoyance`. 



\### 5. `MNLI\_DL-1.ipynb` – Natural Language Inference (MNLI)



Notebook ini melakukan fine-tuning model \*\*Natural Language Inference (NLI)\*\* pada dataset \*\*MNLI\*\*, dengan tiga kelas utama: entailment, neutral, dan contradiction. 



Fitur utama:

\- Tabel hasil training per epoch berisi:

&nbsp; - `Epoch`, `Training Loss`, `Validation Loss`, `Accuracy`, `F1 Micro`, dan `F1 Macro`. 

&nbsp; - Contoh: akurasi validasi sekitar 0.83–0.84 dengan F1 yang sangat dekat. 

\- Ringkasan `TrainOutput` dengan `global\_step≈73632` dan `training\_loss≈0.37` serta detail throughput training. 

\- Pemetaan label `id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}` dan demo inference pada beberapa pasangan:

&nbsp; - Premise: “A man is playing a guitar.”  

&nbsp;   Hypothesis: “A person is making music.” → prediksi \*\*entailment\*\*. 

&nbsp; - Premise: “A man is playing a guitar.”  

&nbsp;   Hypothesis: “No one is playing music.” → prediksi \*\*contradiction\*\*. 

\- Visualisasi dari `trainer.state.log\_history` berupa:

&nbsp; - Kurva Training \& Validation Loss terhadap step. 

&nbsp; - Kurva Validation Accuracy terhadap step. 



\## Teknologi dan Dependensi



Notebook-notebook ini dibuat dan dijalankan terutama di lingkungan \*\*Google Colab\*\* dengan integrasi \*\*Weights \& Biases (wandb)\*\* untuk experiment tracking. 

Dependensi utama:

\- Bahasa \& environment:

&nbsp; - Python 3.x. 

&nbsp; - Jupyter Notebook / Google Colab. 

\- Library machine learning \& NLP:

&nbsp; - `transformers` (HuggingFace) untuk model, tokenizer, dan `Trainer`. 

&nbsp; - `datasets` untuk loading dataset XSUM, AG News, GoEmotions, dan MNLI dari HuggingFace Hub. 

&nbsp; - `torch` (PyTorch) sebagai backend utama untuk training model. 

\- Analisis \& visualisasi:

&nbsp; - `pandas` untuk tabulasi dan analisis log training (`log\_history`). 

&nbsp; - `matplotlib` untuk plotting kurva loss dan metrik. 

\- Experiment tracking:

&nbsp; - `wandb` untuk menyimpan dan memantau run (terlihat dari path seperti `/content/wandb/run-...`). 

