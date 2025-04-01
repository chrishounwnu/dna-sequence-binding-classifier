# 🧬 DNA Sequence Binding Classifier

This project implements a deep learning pipeline for binary classification of DNA sequences based on their binding properties. It uses convolutional neural networks (CNNs) and unsupervised clustering features for enhanced performance.

---

## 📁 Dataset Files

Place all data files in a directory named `data/`.

### Required Files:
- `Xtr.csv` : Training DNA sequences (n=2000).
- `Ytr.csv` : Corresponding binary labels (`0` = unbound, `1` = bound).
- `Xte.csv` : Test DNA sequences (n=1000).

### Optional (but recommended) Files:
- `Xtr_mat100.csv` : Cluster-based features for training sequences.
- `Xte_mat100.csv` : Cluster-based features for test sequences.

> Each DNA sequence has a fixed length of 101 nucleotides (characters from `A`, `C`, `G`, `T`).

---

## 🧠 Models Overview

Two models are trained independently and ensembled for prediction:

### 1. `CNNOnly`
A pure convolutional neural network trained on one-hot encoded sequences. It captures local patterns such as motifs and short-term dependencies between nucleotides.

### 2. `CNNKMeansFusion`
A hybrid model that combines:
- One-hot encoded sequence processed through a CNN.
- KMeans-based unsupervised feature vector (50 dimensions).

Both outputs are concatenated and passed through an MLP for classification.

---

## 🤝 Ensemble Strategy

Final predictions are made using a **weighted average** of the two models:

\[
\texttt{final\_pred} = \alpha \cdot \texttt{preds\_cnn\_only} + (1 - \alpha) \cdot \texttt{preds\_cnn\_kmeans}
\]

> Example: With `alpha = 0.715`, the ensemble achieved **75.67% accuracy**.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/chrishounwnu/dna-sequence-binding-classifier.git
cd dna-sequence-binding-classifier
