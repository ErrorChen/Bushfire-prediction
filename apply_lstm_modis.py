#!/usr/bin/env python3
# apply_lstm_modis.py

import os
import glob
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from LSTM import ImprovedFireModel  # your LSTM definition

def prepare_test_data(window_size=30):
    # 1) Read & aggregate daily from all MODIS CSVs
    files = sorted(glob.glob('datasets/modis_????_Australia.csv'))
    df    = pd.concat([
                pd.read_csv(f, parse_dates=['acq_date'])
                for f in files
            ], ignore_index=True)
    df['acq_min'] = (df['acq_time'] // 100) * 60 + (df['acq_time'] % 100)
    df['date']    = df['acq_date'].dt.floor('D')

    daily = df.groupby('date').agg(
        latitude_mean    = ('latitude','mean'),
        latitude_max     = ('latitude','max'),
        longitude_mean   = ('longitude','mean'),
        longitude_max    = ('longitude','max'),
        brightness_mean  = ('brightness','mean'),
        brightness_max   = ('brightness','max'),
        bright_t31_mean  = ('bright_t31','mean'),
        bright_t31_max   = ('bright_t31','max'),
        scan_mean        = ('scan','mean'),
        scan_max         = ('scan','max'),
        track_mean       = ('track','mean'),
        track_max        = ('track','max'),
        confidence_mean  = ('confidence','mean'),
        confidence_max   = ('confidence','max'),
        acq_min_mean     = ('acq_min','mean'),
        frp_sum          = ('frp','sum'),
        satellite_count  = ('satellite','count'),
        instrument_count = ('instrument','count'),
        version_count    = ('version','count'),
        daynight_count   = ('daynight','count'),
        type_count       = ('type','count')
    )

    # back-fill any missing days
    full_idx = pd.date_range(daily.index.min(),
                             daily.index.max(), freq='D')
    daily = (
        daily.reindex(full_idx, fill_value=0)
             .reset_index().rename(columns={'index':'date'})
    )

    # 2) Prepare X and Y arrays exactly as in training
    feats = [
      'latitude_mean','latitude_max',
      'longitude_mean','longitude_max',
      'brightness_mean','brightness_max',
      'bright_t31_mean','bright_t31_max',
      'scan_mean','scan_max',
      'track_mean','track_max',
      'confidence_mean','confidence_max',
      'acq_min_mean',
      'satellite_count','instrument_count',
      'version_count','daynight_count','type_count'
    ]
    X_all = MinMaxScaler().fit_transform(
              daily[feats].values.astype(np.float32)
            )
    Y_raw = daily['frp_sum'].to_numpy(dtype=np.float32)

    # sliding windows
    Xs, Ys = [], []
    for i in range(len(X_all) - window_size):
        Xs.append(X_all[i:i+window_size])
        Ys.append(Y_raw[i+window_size])
    X = np.stack(Xs)  # shape (N, T, 20)
    Y = np.array(Ys)  # shape (N,)

    # 3) Stratify split so that test set has both FRP=0 and FRP>0
    labels = (Y > 0).astype(int)
    indices = np.arange(len(Y))
    _, test_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )
    X_test = X[test_idx]
    y_test = Y[test_idx]

    return torch.from_numpy(X_test), y_test

def main():
    # load test set
    X_test, y_test = prepare_test_data(window_size=30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model with the *same* input_dim=20
    input_dim = 20
    model     = ImprovedFireModel(input_dim).to(device)
    ckpt      = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("✅ Loaded LSTM model from", ckpt)

    # inference
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy().flatten()

    # save per‐sample CSV
    out_df = pd.DataFrame({
      'frp_true': y_test,
      'frp_pred': y_pred
    })
    out_csv = os.path.join(os.path.dirname(__file__),
                           'modis_lstm_frp_results.csv')
    out_df.to_csv(out_csv, index=False)
    print("✅ Wrote LSTM results to", out_csv)
    print(out_df.head())

if __name__ == "__main__":
    main()
