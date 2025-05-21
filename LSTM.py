# bushfire_lstm_modis.py

import os
# Suppress most TF logging, except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import mixed_precision

# 1. GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Enabled {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
else:
    print("No GPU detected, using CPU")

# 2. Mixed precision (optional, may improve performance on modern GPUs)
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision policy:", mixed_precision.global_policy())

import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 3. Read and concatenate MODIS CSV files (2013–2022)
data_dir = os.path.join(os.getcwd(), 'datasets')
all_files = sorted(glob.glob(os.path.join(data_dir, 'modis_*.csv')))
if not all_files:
    raise FileNotFoundError(f"No files found in {data_dir} matching 'modis_*.csv'")

df_list = [pd.read_csv(f) for f in all_files]
data = pd.concat(df_list, ignore_index=True)

# 4. Preprocessing
data['acq_date'] = pd.to_datetime(data['acq_date'])
data.sort_values('acq_date', inplace=True)
data.ffill(inplace=True)  # forward-fill missing values

# 5. One-Hot encode all non-numeric categorical columns (excluding date & target)
exclude = ['acq_date', 'brightness']
cat_cols = [c for c in data.columns if c not in exclude and data[c].dtype == object]
if cat_cols:
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# 6. Feature & target split
if 'brightness' not in data.columns:
    raise KeyError("'brightness' column not found after encoding")

feature_cols = [c for c in data.columns if c not in ['acq_date', 'brightness']]
X_raw = data[feature_cols].values.astype('float32')
y_raw = data['brightness'].values.reshape(-1, 1).astype('float32')

# 7. Normalisation
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# 8. Build time-series sequences
def create_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 30
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

# 9. Train/test split (no shuffling)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

# 10. Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    # If using mixed precision, output layer should be float32
    Dense(1, activation='linear', dtype='float32')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 11. Train with EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es],
    verbose=2
)

# 12. Evaluate on test set
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}")

# 13. Save model and scalers
model.save('bushfire_lstm_modis.h5')
joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')
