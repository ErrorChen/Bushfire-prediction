#!/usr/bin/env python3
# MLP_fire_risk_regression_fixed.py
# Predicting FRP with a deep MLP, full dataset utilisation, best‐model saving,
# mixed precision, cuDNN autotune, and Windows‐safe multiprocessing.
# OneHotEncoder updated for scikit‑learn ≥1.3 (sparse_output).
# Comments in Australian English.

import os
import glob
import multiprocessing
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Load & Preprocess Data
# -----------------------------
def load_and_preprocess():
    """
    Reads all MODIS CSVs, engineers features from date/time and categoricals,
    scales numeric features, and returns train/val/test tensors.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base_dir, 'datasets', 'modis_*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Convert acquisition date/time
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['month'] = df['acq_date'].dt.month
    df['day_of_year'] = df['acq_date'].dt.dayofyear
    df['acq_time'] = df['acq_time'].astype(int)
    df['hour'] = df['acq_time'] // 100
    df['minute'] = df['acq_time'] % 100
    df['time_minutes'] = df['hour'] * 60 + df['minute']

    numeric_cols = [
        'latitude','longitude','brightness','scan','track',
        'confidence','version','bright_t31',
        'month','day_of_year','time_minutes'
    ]
    categorical_cols = ['satellite','instrument','daynight','type']

    # OneHot encode categoricals with scikit‑learn ≥1.3
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_data = enc.fit_transform(df[categorical_cols])

    # Stack numeric + categorical
    X = np.hstack([df[numeric_cols].to_numpy(dtype=np.float32),
                   cat_data.astype(np.float32)])
    y = df['frp'].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Train/Val/Test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42
    )

    # Scale numeric portion only
    scaler = StandardScaler()
    X_train[:, :len(numeric_cols)] = scaler.fit_transform(X_train[:, :len(numeric_cols)])
    X_val[:, :len(numeric_cols)]   = scaler.transform(X_val[:, :len(numeric_cols)])
    X_test[:, :len(numeric_cols)]  = scaler.transform(X_test[:, :len(numeric_cols)])

    # Convert to PyTorch tensors
    return (
        torch.from_numpy(X_train), torch.from_numpy(y_train),
        torch.from_numpy(X_val),   torch.from_numpy(y_val),
        torch.from_numpy(X_test),  torch.from_numpy(y_test)
    )

# -----------------------------
# 2. Define Deep Regression MLP
# -----------------------------
class DeepFRPNet(nn.Module):
    """
    A deeper MLP for FRP regression:
     - Input: numeric + one-hot categorical
     - Hidden layers: 128 → 64 → 32 → 16
     - Output: single continuous FRP value
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)  # regression output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# -----------------------------
# 3. Training & Evaluation
# -----------------------------
def main():
    # Device and performance flags
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess()
    input_dim = X_train.shape[1]

    # DataLoaders
    BATCH_SIZE = 256
    num_workers = min(8, os.cpu_count() or 4)
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    test_ds  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'),
                              prefetch_factor=2, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'),
                              prefetch_factor=2, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'),
                              prefetch_factor=2, persistent_workers=True)

    # Model, loss, optimiser, AMP
    model     = DeepFRPNet(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler    = GradScaler()

    # Early stopping & best-model saving
    best_val_mae = float('inf')
    best_path    = 'best_frp_model.pt'

    for epoch in range(1, 101):
        # Training
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(xb)
                loss  = criterion(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                val_preds.append(out)
                val_targets.append(yb.numpy())
        val_preds   = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_mae     = mean_absolute_error(val_targets, val_preds)

        print(f"Epoch {epoch:03d} – Train MSE: {train_loss/len(train_ds):.4f}, Val MAE: {val_mae:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_path)
            print(f"  ▶ Saved new best model (MAE {best_val_mae:.4f})")

    # Load and test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            test_preds.append(out)
            test_targets.append(yb.numpy())
    test_preds   = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    test_mae     = mean_absolute_error(test_targets, test_preds)
    print(f"\nTest MAE between predicted FRP and true FRP: {test_mae:.4f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
