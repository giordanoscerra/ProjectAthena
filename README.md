# Project Athena

<p align="center">
  <img src="philosophy.png" alt="Project Athena Logo" width="120">
</p>

<p align="center">
  <a href="https://github.com/giordanoscerra/ProjectAthena/stargazers">
    <img src="https://img.shields.io/github/stars/giordanoscerra/ProjectAthena" alt="GitHub Stars">
  </a>
  <a href="https://github.com/giordanoscerra/ProjectAthena/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/giordanoscerra/ProjectAthena" alt="Contributors">
  </a>
  <a href="https://github.com/giordanoscerra/ProjectAthena">
    <img src="https://img.shields.io/github/repo-size/giordanoscerra/ProjectAthena" alt="Repository Size">
  </a>
  <a href="https://github.com/giordanoscerra/ProjectAthena">
    <img src="https://img.shields.io/github/last-commit/giordanoscerra/ProjectAthena" alt="Last Commit">
  </a>
</p>

---

## Overview

Project Athena is a machine learning-based NLP project developed as part of the Human Language Technologies (HLT) course, a.y. 2023-2024.  
Its goal is to classify philosophical currents from textual data, inferring the school of thought behind a given sentence.

The project explores:
- Dataset analysis and contrastive comparisons with Gutenberg, Brown, and Simple English Wikipedia corpora
- Multiple modeling approaches: Naive Bayes, RNNs, BERT, DistilBERT, and Zero-shot bart-large-mnli
- Evaluation via macro-averaged F1 score, addressing dataset imbalance
- A hypothesis on the impact of short vs. long sentences in classification accuracy  

Full report: [HLT_Project.pdf](./HLT_Project.pdf)

---

## Dataset

We used the [History of Philosophy dataset](https://www.kaggle.com/datasets/kouroshalizadeh/history-of-philosophy) curated by Kourosh Alizadeh.  
- Contains 360,808 sentences  
- Drawn from 59 texts by 36 authors  
- Covers 13 philosophical schools of thought  

Note: The dataset is highly imbalanced, with some schools being over-represented.

---

## Methods

We experimented with three families of models:
- Generative models → Naive Bayes (TF-IDF, Laplace smoothing tuning)  
- Discriminative models → Recurrent Neural Networks (LSTMs, GRUs, BiLSTMs with GloVe embeddings)  
- Transformer-based models → BERT, DistilBERT, Zero-shot bart-large-mnli  

Hypothesis: very short sentences (<15 words) may act like "semantic stopwords" and hinder classification.  
We tested this by training and evaluating on both the full dataset and a reduced dataset (sentences ≥ 84 characters).

---

## Results

- BERT and DistilBERT outperformed all other approaches  
- Naive Bayes surprisingly strong baseline with low computational cost  
- Removing short sentences hurt performance, as they actually contribute useful signals  
- Best model: BERT (fine-tuned on full dataset) → F1 ≈ 0.83 (full test set), 0.88 (long sentences only)

---

## Authors

- Davide Borghini  
- Davide Marchi  
- Giordano Scerra  
- Andrea Marino  
- Yuri Ermes Negri  
