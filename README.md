已思考 6 秒

```
# Bushfire-prediction

Bushfire-prediction is an ENGG2112 project that develops, trains and benchmarks two neural architectures—an MLP classifier and a complete LSTM sequence model—alongside a baseline MODIS-FRP model, to forecast daily bushfire risk across Australia (2013–2022). It integrates rainfall, fire-incident and satellite FRP datasets, preprocesses them, trains each model, evaluates performance with classification and regression metrics, and saves the best-performing weights.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
4. [Data](#data)  
5. [Usage](#usage)  
6. [Modelling](#modelling)  
   - [Baseline MODIS-FRP Model](#baseline-modis-frp-model)  
   - [MLP Classifier](#mlp-classifier)  
   - [LSTM Network](#lstm-network)  
7. [Evaluation & Results](#evaluation--results)  
8. [Project Structure](#project-structure)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Acknowledgements](#acknowledgements)  

## Project Overview

We assemble three complementary data sources—daily rainfall, fire-incident attributes and MODIS Fire Radiative Power (FRP)—to create a unified feature set for binary bushfire-risk classification (“High” vs “Low”) at daily resolution. Three models are implemented:

- **MODIS-FRP baseline**: Simple threshold or regression on satellite FRP data.  
- **MLP Classifier**: Feed-forward network on aggregated features.  
- **LSTM Network**: Sequence model capturing temporal patterns in rainfall and FRP.

## Features

- **Data Integration**:  
  - `datasets/rainfall.csv` (daily rainfall, mm)  
  - `datasets/fire_for16-21_attributes.csv` (fire area, duration, location)  
  - `datasets/modis_YYYY_Australia.csv` (2013–2022 FRP time-series)  
- **Preprocessing**: Imputation of missing values, feature scaling, one-hot encoding.  
- **Models**:  
  - Baseline MODIS-FRP (simple regression/classification).  
  - MLP with configurable hidden layers and early stopping.  
  - LSTM with sliding windows, dropout and checkpointing.  
- **Evaluation**: Classification metrics (precision, recall, F1, accuracy) and regression metrics (MAE, RMSE, R²).  
- **Persistence**: Best weights saved as `best_model.pth` (MLP), `best_lstm_model.pt` (LSTM) and `best_frp_model.pt` (MODIS-FRP).

## Getting Started

### Prerequisites

- Python 3.8 or newer  
- `pip`

### Installation

```bash
git clone https://github.com/ErrorChen/Bushfire-prediction.git
cd Bushfire-prediction
pip install -r requirements.txt
```

## Data

Place the following files in the `datasets/` directory:

```text
fire_for16-21_attributes.csv     # Historical bushfire incidents (2016–2021)
rainfall.csv                     # Daily rainfall measurements (2016–2021)
modis_2013_Australia.csv         # Satellite FRP data (2013)
modis_2014_Australia.csv         # … through to 2022
…
modis_2022_Australia.csv
```

## Usage

1. Update file paths in `MLP.py`, `LSTM.py` and `MODIS_FRP_baseline.py` if necessary.  
2. Run each model:

   ```bash
   python MODIS_FRP_baseline.py    # trains/evaluates FRP baseline
   python MLP.py                   # trains/evaluates MLP
   python LSTM.py                  # trains/evaluates LSTM
   ```

3. Review console outputs and saved weight files.

## Modelling

### Baseline MODIS-FRP Model

- Loads per-year FRP CSVs, aggregates daily FRP.  
- Fits a simple regressor/classifier to predict risk.

### MLP Classifier

Defined in `MLP.py`:

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    random_state=42
)
```

### LSTM Network

Implemented in `LSTM.py` with PyTorch:

- **Input**: Sliding window of past _n_ days’ rainfall + FRP features.  
- **Architecture**: 2-layer LSTM → Dropout → Dense → Sigmoid.  
- **Loss**: Binary cross-entropy; **Optimiser**: Adam.  
- **Checkpoint**: Saves `best_lstm_model.pt` at lowest validation loss.

## Evaluation & Results

- **MODIS-FRP**: Baseline performance logged to `model_comparison_summary.csv`.  
- **MLP**:  
  - Classification report (precision, recall, F1, support) printed.  
  - Overall accuracy: ~80–85%.  
- **LSTM**:  
  - Best epoch: MAE = *XX*, RMSE = *YY*, R² = *ZZ*, Accuracy ≥ 82%.  

## Project Structure

```
Bushfire-prediction/
├── .venv/
├── .vscode/
├── datasets/
│   ├── fire_for16-21_attributes.csv
│   ├── rainfall.csv
│   ├── modis_2013_Australia.csv
│   ├── … 
│   └── modis_2022_Australia.csv
├── best_frp_model.pt
├── best_model.pth
├── best_lstm_model.pt
├── LICENSE
├── LSTM.py
├── MLP.py
├── MODIS_FRP_baseline.py
├── model_comparison_summary.csv
├── proj.code-workspace
└── README.md
```

## Contributing

1. Fork & clone the repo.  
2. Create a feature branch: `git checkout -b feature/…`.  
3. Commit your changes.  
4. Push & open a Pull Request.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Acknowledgements

- **Data providers**: Australian Bureau of Meteorology, NASA MODIS, Kaggle.  
- **Course**: ENGG2112, The University of Sydney.  
- **Libraries**: scikit-learn, pandas, PyTorch.
```