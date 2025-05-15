import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout,
    LSTM, Bidirectional, MultiHeadAttention,
    Add, LayerNormalization, Dense, GRU
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


def load_modis_timeseries(modis_dir='datasets'):
    files = glob.glob(os.path.join(modis_dir, 'modis_*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'acq_date' in df:
            df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
            dfs.append(df[['acq_date','brightness','frp']].dropna(subset=['acq_date']))
    df = pd.concat(dfs, ignore_index=True)
    df['year_month'] = df['acq_date'].dt.to_period('M').dt.to_timestamp()
    ts = df.groupby('year_month').agg(
        count=('brightness','count'),
        brightness=('brightness','mean'),
        frp=('frp','mean')
    ).reset_index()
    ts['month'] = ts['year_month'].dt.month
    ts['sin_m'] = np.sin(2*np.pi*ts['month']/12)
    ts['cos_m'] = np.cos(2*np.pi*ts['month']/12)
    return ts.sort_values('year_month').reset_index(drop=True)


def create_sequences(df_ts, seq_len=12):
    feats = ['count','brightness','frp','sin_m','cos_m']
    data = df_ts[feats].values
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y).reshape(-1,1)


def build_model(seq_len, nfeat, total_steps):
    inp = Input(shape=(seq_len, nfeat))
    # CNN block
    x = Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # Residual GRU blocks with attention
    for units in [128, 64]:
        # project residual to match bidirectional output dims
        res = Conv1D(units*2, 1, padding='same')(x)
        x = Bidirectional(GRU(units, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=units//2)(x, x)
        x = Add()([x, attn, res])
        x = LayerNormalization()(x)
    # Final sequence encoding
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    # Dense head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    # Learning rate schedule
    lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=total_steps)
    optimizer = Adam(learning_rate=lr_schedule)
    model = Model(inp, out)
    model.compile(
        optimizer=optimizer,
        loss=Huber(),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


def main():
    # load and preprocess
    df_ts = load_modis_timeseries()
    X, y = create_sequences(df_ts)
    ns, sl, nf = X.shape
    # scale
    xs = MinMaxScaler(); X = xs.fit_transform(X.reshape(-1,nf)).reshape(ns,sl,nf)
    ys = MinMaxScaler(); y = ys.fit_transform(y)
    # split
    split = int(ns*0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    # build with lr schedule
    total_steps = (split*0.8)//16 * 300
    model = build_model(sl, nf, total_steps)
    es = EarlyStopping('val_mae', patience=10, restore_best_weights=True)
    # train
    model.fit(
        X_tr, y_tr,
        validation_split=0.2,
        epochs=300,
        batch_size=16,
        callbacks=[es],
        verbose=2
    )
    # evaluate
    preds = ys.inverse_transform(model.predict(X_te)).flatten()
    true = ys.inverse_transform(y_te).flatten()
    for i in range(min(5,len(true))):
        dt = df_ts['year_month'].iloc[split+i+sl].date()
        print(f"样本 {i} ({dt}): 真实 = {true[i]:.0f}, 预测 = {preds[i]:.0f}")
    model.save('lstm_modis_final_bushfire.keras')

if __name__=='__main__':
    main()
