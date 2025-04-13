
# HyNN-MIMO-Localization

**Hybrid Neural Networks-based MIMO Indoor Localization via Synthetic Spatial Representations**

[Dataset on IEEE Dataport](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)

## 🔬 Abstract

Channel State Information (CSI) from Massive MIMO systems offers rich spatial and frequency data for indoor localization, yet effectively learning from it remains a challenge. Recent work has applied Convolutional Neural Networks (CNNs) to this task, but these models often depend on large antenna arrays and fail to fully exploit the numerical structure of CSI. In this work, we investigate the use of synthetic image generation techniques, originally developed for tabular data, to better exploit spatial relationships and enhance CSI-based indoor localization. We propose two hybrid neural network architectures—HyCNN and HyViT—that combine a CNN or Vision Transformer with a Multi-Layer Perceptron to jointly process synthetic spatial representations and raw CSI features. We evaluate these models on a large-scale MIMO indoor localization dataset spanning multiple antenna topologies. Our experiments show that HyViT consistently outperforms both classical regressors and prior CNN-based models, achieving state-of-the-art accuracy even with as few as 8 or 16 antennas.

## 📓 Notebooks

| Notebook | Description |
|---------|-------------|
| `convert_csi_to_csv_by_antennas.ipynb` | Converts raw CSI samples into structured CSV format with modulus and phase features per antenna and subcarrier. |
| `prepare_MIMO_tabular_data_LAMDA_TALENT.ipynb` | Prepares data for classical model training following LAMDA-TALENT format. |
| `inference_metrics_classical_models.ipynb` | Loads trained classical models (XGBoost, CatBoost, etc.) and computes RMSE, MAE, and inference latency. |
| `Pytorch_Regression-CNN-LRFinder+OneCycleLR.ipynb` `CNN`, `CNN+MLP`, `ViT`, `ViT+MLP`, `MLP` | Training code for deep architectures used in the paper. Logs stored under `/logs` per dataset.  |

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

- HyViT with REFINED achieves a 2.21 mm localization error on DIS using only 16 antennas, surpassing previous state-of-the-art methods with 64 antennas.
- **ViT outperforms CNN** in all transformation settings.
- Classical KNN performs well (e.g., 3.27 mm on DIS), but deep models are significantly faster.
- All transformation methods (TINTO, REFINED, IGTD) are implemented in [`TINTOlib`](https://tintolib.readthedocs.io).

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
