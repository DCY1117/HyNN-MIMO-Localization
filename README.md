
# Explainable Hybrid Vision Transformer Architectures-Based MIMO Indoor Localization via Synthetic Spatial Representations

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/manwestc/Interpetrable-Hybrid-CNN-KAN/blob/master/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://pypi.python.org/pypi/)
[![Documentation Status](https://readthedocs.org/projects/morph-kgc/badge/?version=latest)](https://tintolib.readthedocs.io/en/latest/)
[![TINTOlib](https://img.shields.io/badge/library-TINTOlib-9cf)](https://github.com/oeg-upm/TINTOlib)

#### This repository supports the scientific article **"Explainable Hybrid Vision Transformer Architectures-Based MIMO Indoor Localization via Synthetic Spatial Representations"**.

It includes all experiments, hybrid model implementations, and evaluation code used to investigate the effectiveness of combining **Vision Transformers (ViT)** with **Multi-Layer Perceptrons (MLPs)** for learning from synthetic 2D image representations of tabular CSI data using various encoding methods from [TINTOlib](https://tintolib.readthedocs.io/en/latest/), including **TINTO**, **IGTD**, and **REFINED**.

[Dataset on IEEE Dataport](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)

---

## 🔎 Explore this GitHub with DeepWiki

This repository has a dedicated space on **DeepWiki**, where you can explore semantic documentation, relevant links, bibliography, and answers to frequently asked questions about its use and application.

<p align="center">
  <a href="https://deepwiki.com/DCY1117/HyNN-MIMO-Localization" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"/>
  </a>
</p>

---

## 🔬 Abstract

Channel State Information (CSI) from Massive MIMO systems offers rich spatial and frequency data for indoor localization, yet effectively learning from it remains a challenge. Recent work has applied Convolutional Neural Networks (CNNs) to this task, but these models often depend on large antenna arrays and fail to fully exploit the numerical structure of CSI. In this work, we investigate the use of synthetic image generation methods, originally developed for tabular data, to better exploit spatial relationships and enhance CSI-based indoor localization. We propose two hybrid neural network architectures, HyCNN and HyViT, that combine a CNN or Vision Transformer with a Multi-Layer Perceptron to jointly process synthetic spatial representations and raw CSI features. We evaluate these models on a large-scale MIMO indoor localization dataset spanning multiple antenna topologies. Our experiments show that HyViT consistently outperforms both classical regressors and prior vision-based models, achieving state-of-the-art accuracy even with as few as 8 or 16 antennas without any data agumentation. Beyond performance, we conduct a detailed interpretability analysis using attention attribution and entropy metrics to investigate how both vision-only and hybrid models attend to spatial regions, antenna groupings, and modality-specific features across transformer layers. Unlike prior works, our design eliminates the need for handcrafted topologies or graph construction, and explicitly incorporates interpretability as a key evaluation dimension aspect largely overlooked in CSI localization literature.

## 📓 Notebooks

| Notebook | Description |
|---------|-------------|
| `convert_csi_to_csv_by_antennas.ipynb` | Converts raw CSI samples into structured CSV format with modulus and phase features per antenna and subcarrier. |
| `prepare_MIMO_tabular_data_LAMDA_TALENT.ipynb` | Prepares data for classical model training following LAMDA-TALENT format. |
| `inference_metrics_classical_models.ipynb` | Loads trained classical models (XGBoost, CatBoost, etc.) and computes RMSE, MAE, and inference latency. |
| `Pytorch_Regression-CNN-LRFinder+OneCycleLR.ipynb`, `CNN`, `CNN+MLP`, `ViT`, `ViT+MLP`, `MLP` | Training code for deep architectures used in the paper. Logs stored under `/logs` per dataset.  |
| `Regression_ViT+MLP_XAI.ipynb` | Interpretability analysis for the hybrid ViT+MLP model using attention attribution. |
| `Regression_ViT_XAI.ipynb` | Interpretability analysis for the standalone ViT model using attention attribution. |

## 🧪 Datasets

We use the [Ultra-Dense Indoor MaMIMO CSI dataset (IEEE Dataport)](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset), which includes over 252k samples recorded across 8 and 16 antenna setups for three configurations:
- **DIS**
- **ULA**
- **URA**

Samples are split into:
- 80% Train
- 15% Validation
- 5% Test

## ⚙️ Training Configurations

- **Classical models**: Trained using Optuna for RMSE optimization.
  - XGBoost: 50 trials
  - LightGBM: 20 trials
  - CatBoost: 20 trials (`cat_policy=indices`)
  - KNN: 2 neighbors optimal (no more hyperparameter tuning) 
- **Deep models**: 200 epochs, `AdamW`, MSE loss, OneCycle learning rate schedule, batch size = 128.

## 💡 Highlights

- We adapt three state-of-the-art tabular-to-image transformation methods—TINTO, REFINED, and IGTD—to generate structured visual representations of CSI data for indoor localization, avoiding any manual graph or topology design.
- Our HyViT architecture fuses synthetic images with raw CSI features and achieves state-of-the-art localization accuracy using only 8 or 16 antennas—without any data augmentation.
- We perform in-depth attention attribution and entropy-based interpretability analysis, showing how ViT and HyViT attend to spatial regions, antenna sources, and feature types across transformer layers—a novel perspective in CSI-based modeling.
- HyViT with REFINED achieves 2.21 mm error on DIS with just 16 antennas, outperforming prior methods using 64 antennas. ViT consistently outperforms CNN across transformations. Classical KNN remains competitive (e.g., 3.27 mm on DIS) but is much slower than deep models.
- All transformation methods (TINTO, REFINED, IGTD) are implemented in [`TINTOlib`](https://tintolib.readthedocs.io), ensuring transparency, scalability, and extensibility.

## 📥 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-org/HyNN-MIMO-Localization.git
cd HyNN-MIMO-Localization

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python notebooks/convert_csi_to_csv_by_antennas.ipynb

# Train classical models
bash LAMDA-TALENT/LAMDA-TALENT/scripts/MIMO_reg.sh

# Train deep models
# Follow notebooks/CNN.ipynb, ViT+MLP.ipynb, etc.
```

## 🤝 Acknowledgements

This research was funded by:
- **European Commission H2020 project** “COGITO” (#958310)
- **Madrid Government (V PRICIT)** under the Multiannual Agreement with Universidad Politécnica de Madrid.

The classical model code is adapted from [LAMDA-TALENT](https://github.com/LAMDA-Tabular/TALENT).
