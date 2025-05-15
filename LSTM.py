import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load_multivariate_data(
    rainfall_path='rainfall.csv',
    modis_dir='.'
) -> tuple:
    # 1. 加载并清理原始年度降雨与火险指数数据
    raw = pd.read_csv(rainfall_path, header=0)
    # 解析周期列和首行标题行
    period_col = raw.columns[0]
    header0 = raw.iloc[0]
    # 提取月份名称和对应的列名
    rain_cols = raw.columns[1:13]
    months = header0[1:13].tolist()
    # 构建重命名映射：旧列名 -> 月份
    rename_map = {old: new for old, new in zip(rain_cols, months)}
    # 找到烧毁面积列（首行值为 'Ha Burnt'）
    burnt_col = header0[raw.columns] .eq('Ha Burnt').idxmax()
    # 重命名列
    df = raw.rename(columns={**rename_map, burnt_col: 'HaBurnt', period_col: 'Period'})
    # 删除首行并重置索引
    df = df.drop(index=0).reset_index(drop=True)
    # 转为数值型
    for m in months + ['BushfireIndex', 'HaBurnt']:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df['BushfireIndex'] = pd.to_numeric(df['Dennison Bushfire Indices'], errors='coerce')
    df = df.drop(columns=['Dennison Bushfire Indices'], errors='ignore')
    # 丢弃含缺失值的行
    df = df.dropna(subset=months + ['BushfireIndex', 'HaBurnt']).reset_index(drop=True)

    # 2. 构建多维度特征序列
    X_rain = df[months].values
    X_index = np.repeat(df['BushfireIndex'].values.reshape(-1,1), len(months), axis=1)

    # MODIS 热点按月计数
    hotspot_counts = []
    for period in df['Period']:
        year = str(period).split('-')[0]
        modis_file = os.path.join(modis_dir, f'modis_{year}_Australia.csv')
        if os.path.exists(modis_file):
            md = pd.read_csv(modis_file)
            md['month'] = pd.to_datetime(md['acq_date']).dt.month
            counts = md.groupby('month').size().reindex(range(1,13), fill_value=0).values
        else:
            counts = np.zeros(len(months), dtype=int)
        hotspot_counts.append(counts)
    X_modis = np.array(hotspot_counts)

    # 合并特征
    X = np.stack([X_rain, X_modis, X_index], axis=2)
    y = df['HaBurnt'].values.reshape(-1,1)
    return X, y


def build_model(seq_len: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    X, y = load_multivariate_data()
    seq_len, n_features = X.shape[1], X.shape[2]

    # 归一化
    x_scaler = MinMaxScaler()
    X_flat = X.reshape(-1, n_features)
    X_scaled = x_scaler.fit_transform(X_flat).reshape(-1, seq_len, n_features)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    # 时间序列拆分
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    model = build_model(seq_len, n_features)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        validation_split=0.2,
        callbacks=[es],
        verbose=2
    )

    # 评估预测
    y_pred = y_scaler.inverse_transform(model.predict(X_test))
    y_true = y[split:]
    for i in range(min(5, len(y_true))):
        print(f"样本 {i}: 真实 Ha Burnt = {y_true[i,0]:.1f}, 预测 = {y_pred[i,0]:.1f}")

    model.save('lstm_bushfire_model.h5')

if __name__ == '__main__':
    main()