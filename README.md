# 🧠 EEG Visual Classification Problem: Classical ML vs. End-to-End Neural Architecture Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Ready](https://img.shields.io/badge/Kaggle-Ready-20BEFF.svg)](https://www.kaggle.com/)
[![EEGCVPR40 Benchmark](https://img.shields.io/badge/Dataset-EEGCVPR40-orange.svg)](https://github.com/perceivelab/eeg-cvpr17)

This repository provides an empirical scientific benchmark on the **EEG Visual Stimulus Classification Problem** (40 ImageNet visual classes, 128-channel 1000Hz EEG, official image-disjoint benchmark split).

It benchmarks **10 diverse classical machine learning algorithms** (enhanced with multi-domain feature fusion and 8-pass signal-space Test-Time Augmentation) against **NeuroStream-SST**, a modern state-of-the-art **Spectral-Spatio-Temporal Neural Architecture** (*Scientific Reports*, 2026).

---

## 📊 Comprehensive Benchmark Results (Empirical Verification on Kaggle)

The following metrics reflect evaluations on the official **EEGCVPR40 Image-Disjoint Split** (7,959 training trials, 1,987 test trials, across 40 ImageNet classes; random chance = 2.50%):

### 1. Comparative Performance Matrix

| Model / Architecture | Model Family | Raw Top-1 (%) | TTA Top-1 (%) | TTA Δ (%) | Top-3 (%) | Top-5 (%) | Image-Pooled (%) | Mean Rank | Correct / Total | Fit Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Method #1: Riemannian Tangent Space + L2 LogReg** | Manifold / Linear | 10.17% | 10.12% | -0.05% | 22.40% | 33.17% | 20.12% | 12.46 | 201 / 1987 | 11.07s |
| **Method #2: FBCSP + Shrinkage LDA (sLDA)** | Bayesian / Discriminant | 10.42% | 10.02% | -0.40% | 22.85% | 34.17% | 20.42% | 12.59 | 199 / 1987 | 0.70s |
| **Method #3: Prototypical Memory Bank (MDM)** | Metric / Geometric | 5.54% | 5.89% | +0.35% | 15.70% | 23.65% | 10.21% | 15.74 | 117 / 1987 | 0.02s |
| **Method #4: Spatio-Temporal ERP + Linear SVM** | Max-Margin Linear | 8.25% | 8.66% | +0.40% | 19.58% | 27.98% | 20.42% | 14.00 | 172 / 1987 | 49.34s |
| **Method #5: Non-Linear Kernel SVM (RBF Kernel)** | Max-Margin Kernel | 10.77% | 10.67% | -0.10% | 27.48% | 38.05% | 31.53% | 11.23 | 212 / 1987 | 174.73s |
| **Method #6: Random Forest (150 Trees)** | Tree Ensemble (Bagging) | 5.89% | 4.93% | -0.96% | 15.45% | 23.50% | 13.21% | 15.82 | 98 / 1987 | 17.63s |
| **Method #7: Histogram Gradient Boosting** | Tree Ensemble (Boosting) | 7.05% | 7.20% | +0.15% | 18.72% | 28.33% | 13.51% | 13.93 | 143 / 1987 | 140.28s |
| **Method #8: Extra Trees Classifier** | Tree Ensemble (Random) | 7.00% | 6.74% | -0.25% | 17.87% | 27.13% | 17.42% | 14.77 | 134 / 1987 | 1.66s |
| **Method #9: K-Nearest Neighbors (Cosine k-NN)** | Non-Parametric | 7.70% | 7.60% | -0.10% | 17.56% | 26.02% | 16.22% | 17.29 | 151 / 1987 | 0.00s |
| **Method #10: Gaussian Naive Bayes** | Generative | 5.18% | 5.08% | -0.10% | 15.05% | 22.85% | 6.61% | 16.26 | 101 / 1987 | 0.03s |
| **🏆 SOTA Classical Super-Ensemble** | Multi-Family Ensemble | **11.02%** | **11.17%** | **+0.15%** | **25.57%** | **35.63%** | **26.13%** | **12.09** | **222 / 1987** | 395.46s |
| **🧠 NeuroStream-SST (End-to-End Neural Pipeline)** | Spectral-Spatio-Temporal | **57.47%** | — | — | — | **93.81%** | **92.49%** | — | **1142 / 1987** | ~18 min |

---

### 2. NeuroStream-SST Subject-Level Accuracy Breakdown

Under the strict Subject-Invariant mode (evaluating generalization across unique subject brains):

| Subject ID | Test Trials | Single-Trial Subject Accuracy |
|:---|:---:|:---:|
| **Subject 1** | 330 | **52.12%** |
| **Subject 2** | 332 | **63.25%** |
| **Subject 3** | 332 | **59.64%** |
| **Subject 4** | 333 | **68.77%** |
| **Subject 5** | 330 | **54.85%** |
| **Subject 6** | 330 | **46.06%** |
| **Overall Dataset** | **1,987** | **57.47% Single-Trial / 92.49% Image-Pooled (Top-5: 100.00%)** |

---

## 🔬 The Fundamental EEG Problem: Why Classical ML Hits an Asymptotic Ceiling

Even after extensive spatio-temporal filtering (1–30 Hz Butterworth bandpass, Common Average Referencing CAR, baseline detrending) and multi-pass Test-Time Augmentation (TTA), **classical machine learning models consistently plateau around 10%–11% Top-1 single-trial accuracy on 40 classes** (random chance: 2.50%).

This performance ceiling is rooted in **four fundamental biophysical and mathematical barriers**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 4 FUNDAMENTAL BARRIERS IN EEG DECODING                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Extreme Negative SNR (< -10 dB)   │ Skull resistivity (~80x brain) smears potentials│
│ 2. Temporal Dynamic Collapse          │ Static covariance C = (1/T)XXᵀ discards wave order│
│ 3. Inter-Subject Cortical Geometry   │ Unique gyri/sulci folding angles distort scalp dipoles│
│ 4. Axis-Aligned / Linear Boundaries  │ Curved Riemannian SPD manifolds resist hyperplanes │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Extreme Negative Signal-to-Noise Ratio ($\text{SNR} < -10\text{ dB}$)
* Scalp EEG electrodes record microvolt-level electrical activity ($1\text{--}10\ \mu\text{V}$) originating from synchronized postsynaptic potentials of pyramidal neurons in the cerebral cortex.
* The skull acts as an extreme spatial low-pass filter with electrical resistivity $\approx 80\times$ higher than cerebral tissue.
* Non-cortical artifacts are an order of magnitude larger than stimulus signals:
  * Ocular blinks: $100\text{--}200\ \mu\text{V}$ ($10\times\text{--}20\times$ signal amplitude)
  * Facial EMG muscle tremors: broadband high-amplitude noise
  * Sweating & electrode impedance drift: low-frequency DC baseline wander

### 2. Temporal Dynamic Phase Collapse in Static Features
* Classical feature extractors collapse the temporal axis:
  * **Spatial Covariance**: $C = \frac{1}{T-1} X X^T \in \mathbb{R}^{K \times K}$
  * **Bandpass Energy / FBCSP**: $\log(\text{Var}(W_b X_b))$
* By integrating across time, static features **completely discard temporal wave sequence information**. Visual categorization in the human brain is an evolving spatiotemporal cascade:
  $$\text{Early Sensory VEP (P100, 80--120ms)} \;\longrightarrow\; \text{Structural Discrimination (N170, 140--200ms)} \;\longrightarrow\; \text{Semantic Closure (P300, 250--400ms)}$$
* A classical classifier (SVM, LDA, RF) sees only a static 1D feature vector and cannot model dynamic phase-amplitude relationships across time.

### 3. High Inter-Subject & Inter-Trial Variability
* Cortical folding patterns (gyri and sulci) vary between individuals. The exact same visual stimulus generates electrical dipoles oriented in different directions relative to scalp sensors across different people.
* Single-trial visual responses exhibit $\pm 20\text{--}40\text{ ms}$ trial-to-trial latency jitter.

### 4. Non-Linear Manifolds vs. Axis-Aligned Cuts
* Spatial covariance matrices reside on a curved **Riemannian Symmetric Positive Definite (SPD) manifold**.
* Tree-based algorithms (Random Forest, Gradient Boosting) construct orthogonal, axis-aligned decision cuts that cannot capture geodesic distances along curved manifolds.
* Linear hyperplanes assume stationary separability, but trial noise causes single-trial embeddings to scatter widely.

---

## 📁 Repository Structure

```
EEG-Classification-Problem/
├── README.md                                    # Empirical benchmark documentation & analysis
├── top_classical_ml_multi_algorithms.py         # 10 Classical ML Models + 8-Pass TTA + Super-Ensemble
├── top_classical_ml_multi_algorithms.ipynb      # Kaggle Notebook version for Classical ML
├── neurostream_sst_classifier.py                # NeuroStream-SST Spectral-Spatio-Temporal Pipeline
└── neurostream_sst_classifier.ipynb             # Kaggle Notebook version for NeuroStream-SST
```

---

## 🛠️ Pipeline 1: Comprehensive Classical Machine Learning Suite

The script `top_classical_ml_multi_algorithms.py` trains and evaluates **10 diverse classical machine learning models** spanning all major model families on the exact same multi-domain feature representation:

* **Multi-Domain Feature Concatenation ($608\text{ features}$)**:
  * Riemannian Tangent Space coordinates ($300\text{ feats}$)
  * FBCSP bandpass log-variances ($20\text{ feats}$)
  * Visual Evoked Potential (VEP) windowed dynamics ($288\text{ feats}$)
* **8-Pass Signal-Space Test-Time Augmentation (TTA)**:
  * Pass 0: Pristine original signal
  * Passes 1–4: Temporal phase leads and lags ($\pm 4\text{ms}, \pm 8\text{ms}$)
  * Pass 5: Channel dropout (8% sensor masking)
  * Pass 6: Micro-Gaussian thermal jitter ($\sigma = 0.02$)
  * Pass 7: Subtle amplitude scaling ($1.02\times$) + micro-jitter
  * Dual TTA: Feature-level centroid smoothing + probability distribution pooling.

---

## ⚡ Pipeline 2: NeuroStream-SST (Spectral-Spatio-Temporal Neural Pipeline)

`neurostream_sst_classifier.py` breaks through the classical plateau by learning end-to-end differentiable representations:

```
Raw EEG Trial [128 Channels × 440 Timepoints]
     │
     ├──► Differentiable Morlet Wavelet Filterbank (High-Gamma 70-120 Hz)
     │    └── Isolates high-frequency cortical spike envelopes (immune to low-frequency session drift)
     │
     ├──► 2D Scalp Topographical Projection (11×11 Mesh)
     │    └── Preserves cortical spatial neighborhoods across the skull
     │
     ├──► 2D Spatial Convolutions (32 -> 64 filters)
     │    └── Extracts localized cortical receptive fields
     │
     ├──► Dilated 1D Temporal Convolutions
     │    └── Tracks the dynamic visual cascade across the continuous 440ms epoch
     │
     └──► Compact Latent Head (256-D, LayerNorm, Dropout 0.20)
          └── Achieves 57.47% Single-Trial and 92.49% Image-Level Pooled Accuracy
```

---

## 🚀 How to Run on Kaggle

Both scripts are completely self-contained with **zero external package installations required**.

### Step 1: Classical Machine Learning Benchmark
1. Create a new notebook on Kaggle.
2. Attach dataset: `evolutionaw/eegcvpr40` (Path: `/kaggle/input/datasets/evolutionaw/eegcvpr40/data`).
3. Copy-paste [`top_classical_ml_multi_algorithms.py`](./top_classical_ml_multi_algorithms.py) into a code cell and click **Run**.
4. Outputs saved to `/kaggle/working/eeg_denoiser_results` and synced to Google Drive folder `denoiserResult`.

### Step 2: NeuroStream-SST Pipeline
1. In Kaggle (ensure GPU accelerator: T4 or P100 is enabled).
2. Copy-paste [`neurostream_sst_classifier.py`](./neurostream_sst_classifier.py) into a code cell and click **Run**.
3. Runtime: **~18 minutes** (65 epochs with mixed-precision FP16 AMP).
4. Outputs saved: Best weights `best_neurostream_sst.pt` and scientific report synced to Drive folder `denoiserResult`.

---

## 👥 Authors & Acknowledgments

* **Repository**: [EEG-Classification-Problem](https://github.com/Youssef-Ebn-Yasser/EEG-Classification-Problem)
* **Dataset**: EEGCVPR40 Benchmark (*Spampinato et al., CVPR 2017*)
* **Scientific Reference**: *NeuroStream: spectral-spatio-temporal learning for visual stimulus classification from EEG*, Scientific Reports, 2026.
