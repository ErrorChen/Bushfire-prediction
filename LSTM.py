import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import mixed_precision, layers, Model, callbacks
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ----------------------------
# 1. 混合精度训练
# ----------------------------
mixed_precision.set_global_policy('mixed_float16')

# ----------------------------
# 2. 加载与特征工程
# ----------------------------
def load_modis_timeseries(modis_dir='datasets'):
    files = glob.glob(os.path.join(modis_dir, 'modis_*.csv'))
    if not files:
        raise FileNotFoundError(f"No files matching 'modis_*.csv' found in {modis_dir!r}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'acq_date' not in df.columns:
            print(f"Warning: 'acq_date' column missing in {f!r}, skipping.")
            continue
        df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
        df = df.dropna(subset=['acq_date', 'brightness', 'frp'])
        dfs.append(df[['acq_date', 'brightness', 'frp']])

    if not dfs:
        raise ValueError("No valid MODIS dataframes were loaded; check your CSV files for required columns.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['year_month'] = df_all['acq_date'].dt.to_period('M').dt.to_timestamp()
    ts = (
        df_all
        .groupby('year_month')
        .agg(
            count=('brightness', 'count'),
            brightness=('brightness', 'mean'),
            frp=('frp', 'mean')
        )
        .reset_index()
    )
    ts['month'] = ts['year_month'].dt.month
    ts['sin_m'] = np.sin(2 * np.pi * ts['month'] / 12)
    ts['cos_m'] = np.cos(2 * np.pi * ts['month'] / 12)
    return ts.sort_values('year_month').reset_index(drop=True)

# ----------------------------
# 3. 构建 tf.data 数据流水线
# ----------------------------
def make_dataset(df, seq_len=12, batch_size=32):
    data = df[['count','brightness','frp','sin_m','cos_m']].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True).flat_map(lambda w: w.batch(seq_len + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1, 0]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    ds = ds.repeat()               # 保证无限循环，不会耗尽
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------
# 4. 改进模型结构
# ----------------------------
def build_model(seq_len, nfeat):
    inp = layers.Input(shape=(seq_len, nfeat))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    for units in [128, 64]:
        res = layers.Conv1D(units * 2, 1, padding='same')(x)
        x = layers.Bidirectional(layers.GRU(units, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=units//2)(x, x)
        x = layers.Add()([x, attn, res])
        x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, dtype='float32')(x)  # 强制输出为 float32 精度

    return Model(inputs=inp, outputs=out)

# ----------------------------
# 5. 主训练流程
# ----------------------------
def main():
    # 确保 TensorBoard 日志目录及子目录是文件夹
    if os.path.exists('logs'):
        # 如果 logs 不是目录，则删除
        if not os.path.isdir('logs'):
            os.remove('logs')
    os.makedirs('logs/train', exist_ok=True)
    os.makedirs('logs/validation', exist_ok=True)

    # 1. 加载与预处理
    df_ts = load_modis_timeseries()
    seq_len = 12
    batch_size = 16

    # 2. 构建数据集
    full_ds = make_dataset(df_ts, seq_len=seq_len, batch_size=batch_size)
    total = len(df_ts) - seq_len
    split = int(total * 0.8)
    # 计算 steps
    steps_per_epoch = split // batch_size
    validation_steps = (total - split) // batch_size

    train_ds = full_ds.take(steps_per_epoch)
    val_ds   = full_ds.skip(steps_per_epoch)

    # 3. 构建并编译模型
    model = build_model(seq_len, nfeat=5)
    total_steps = steps_per_epoch * 300
    lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=total_steps)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=Huber(),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )

    # 4. 配置回调
    cb = [
        callbacks.EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5),
        callbacks.TensorBoard(log_dir='logs'),
        callbacks.ModelCheckpoint('best_lstm_modis.h5', save_best_only=True)
    ]

    # 5. 模型训练
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=300,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=cb,
        verbose=2
    )

    # 6. 加载最佳权重并评估
    model.load_weights('best_lstm_modis.h5')

    # 构造测试集并预测
    test_ds = val_ds
    preds, trues = [], []
    for x_batch, y_batch in test_ds:
        p = model.predict(x_batch)
        preds.append(p)
        trues.append(y_batch.numpy().reshape(-1,1))
    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # 反缩放（示例用同一 scaler）
    ys_scaler = MinMaxScaler().fit(trues)
    preds_inv = ys_scaler.inverse_transform(preds)
    trues_inv = ys_scaler.inverse_transform(trues)

    # 打印前 5 个预测对比
    dates = df_ts['year_month'].iloc[split+seq_len : split+seq_len+5].dt.date
    for i, dt in enumerate(dates):
        print(f"样本 {i} ({dt}): 真实 = {trues_inv[i,0]:.0f}, 预测 = {preds_inv[i,0]:.0f}")

    # 保存最终模型
    model.save('lstm_modis_final_bushfire.keras', include_optimizer=False)
    print("Training complete. Final model saved as 'lstm_modis_final_bushfire.keras'")

if __name__ == '__main__':
    main()
