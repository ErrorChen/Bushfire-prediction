# Bushfire-prediction

This repository provides two PyTorch-based models to predict daily Fire Radiative Power (FRP) from MODIS satellite data over Australia:

- **DeepFRPNet (MLP)**: a deep feed-forward regression network  
- **ImprovedFireModel (LSTM + CNN + Self-Attention)**: a sequence model with 30-day sliding windows  

It also includes scripts for training, inference and summarising performance.

---

## 📁 Repository Structure

```
Bushfire-prediction/
├── datasets/                      # MODIS CSV files: modis_YYYY_Australia.csv
├── LSTM.py                        # Train ImprovedFireModel (CNN + biLSTM + attention) 
├── MLP.py                         # Train DeepFRPNet MLP model 
├── apply_lstm_modis1.py          # Inference script for LSTM model 
├── apply_mlp_modis1.py           # Inference script for MLP model 
├── apply_summary1.py             # Compute & save combined metrics 
├── best_model.pth                # Checkpoint of best LSTM model
├── best_frp_model.pt             # Checkpoint of best MLP model
├── model_comparison_summary.csv  # Summary of regression & classification metrics
├── LICENSE                        # MIT License
└── proj.code-workspace           # VS Code workspace settings
```

---

## 🛠 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/ErrorChen/Bushfire-prediction.git
   cd Bushfire-prediction
   ```

2. **(Optional) Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib
   ```

---

## 📦 Data Preparation

Place your daily MODIS data CSVs in the `datasets/` folder, named exactly as:
```
datasets/modis_YYYY_Australia.csv
```
where `YYYY` is a four-digit year.  
- **LSTM.py** aggregates these files via `glob('datasets/modis_????_Australia.csv')`   
- **MLP.py** uses `glob('datasets/modis_*.csv')`   

Each CSV must include at least:
```
acq_date, acq_time, latitude, longitude,
brightness, bright_t31, scan, track,
confidence, frp, satellite, instrument,
daynight, type, version
```

---

## ⚙️ Model Training

### 1. Train the MLP (DeepFRPNet)

```bash
python MLP.py
```

- Loads & preprocesses all `datasets/modis_*.csv`   
- Splits into train/val/test  
- Trains with mixed precision & early stopping  
- Saves best weights to `best_frp_model.pt`

### 2. Train the LSTM-based model (ImprovedFireModel)

```bash
python LSTM.py
```

- Aggregates daily summaries with a 30-day lookback window   
- Trains CNN→biLSTM→Self-Attention network  
- Saves best weights to `best_model.pth`

---

## 🚀 Inference

### MLP Inference

```bash
python apply_mlp_modis1.py
```

- Loads `best_frp_model.pt` and all `datasets/modis_*.csv`  
- Outputs `modis_mlp_frp_results.csv` with true vs predicted FRP 

### LSTM Inference

```bash
python apply_lstm_modis1.py
```

- Loads `best_model.pth` and all `datasets/modis_YYYY_Australia.csv`  
- Outputs `modis_lstm_frp_results.csv` with true vs predicted FRP 

---

## 📊 Metrics Summary

Combine regression & classification metrics for both models:

```bash
python apply_summary1.py
```

- Reads `modis_mlp_frp_results.csv` & `modis_lstm_frp_results.csv`  
- Computes MAE, RMSE, R², plus binary metrics via thresholding  
- Writes `model_comparison_summary.csv` 

Example of `model_comparison_summary.csv`:

| model            | metric   | value  |
| :--------------- | :------- | :----- |
| MLP_regression   | MAE      | 12.345 |
| MLP_regression   | RMSE     | 23.456 |
| LSTM_regression  | MAE      | 10.123 |
| LSTM_regression  | R²       | 0.789  |
| ...              | ...      | ...    |

---

## 🤝 Contributing

1. Fork this repo  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/your-feature
   ```  
3. Commit your changes:  
   ```bash
   git commit -m "Add awesome feature"
   ```  
4. Push and open a Pull Request  

Please include tests and update this README as needed.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.