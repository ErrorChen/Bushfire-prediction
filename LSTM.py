import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_rainfall_data(path: str = 'datasets/rainfall.csv') -> pd.DataFrame:
    # CSV header row is on line 1 (header=1)
    df = pd.read_csv(path, header=1)
    # Keep year sequence with monthly rainfall and burnt area
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul']
    # Remove unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 13', 'Aug-Jan', 'Sep-Jan', 'Oct-Jan'], errors='ignore')
    # Convert to numeric values
    for m in months:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df['Ha Burnt'] = pd.to_numeric(df['Ha Burnt'], errors='coerce')
    # Drop rows containing NaN values
    df = df.dropna(subset=months + ['Ha Burnt']).reset_index(drop=True)
    return df

def create_sequences(data: np.ndarray, seq_len: int = 12) -> np.ndarray:
    # Treat each row as a sequence
    # data: (n_samples, seq_len)
    # return (n_samples, seq_len, 1)
    return data.reshape(data.shape[0], seq_len, 1)

def build_lstm_model(seq_len: int, n_features: int = 1) -> Sequential:
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(seq_len, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # 1. Load data
    df = load_rainfall_data(os.path.join('datasets', 'rainfall.csv'))
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul']
    X = df[months].values  # (n_years, 12)
    y = df['Ha Burnt'].values.reshape(-1, 1)

    # 2. Normalize data
    x_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaler = MinMaxScaler(feature_range=(0,1))
    # Flatten X for scaling then reshape into sequences
    X_flat = x_scaler.fit_transform(X)
    X_seq = create_sequences(X_flat, seq_len=12)
    y_scaled = y_scaler.fit_transform(y)

    # 3. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_scaled, test_size=0.2, random_state=42
    )

    # 4. Build and train model
    model = build_lstm_model(seq_len=12)
    cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[cb],
        verbose=2
    )

    # 5. Evaluate and predict
    y_pred_scaled = model.predict(X_test)
    # Inverse transform to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    # Print sample results
    for i in range(min(5, len(y_true))):
        print(f"Sample {i}: True Ha Burnt = {y_true[i,0]:.1f}, Predicted = {y_pred[i,0]:.1f}")

    # Optional: save the model
    model.save('lstm_bushfire_model.h5')

if __name__ == '__main__':
    main()
