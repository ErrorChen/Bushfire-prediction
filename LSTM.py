import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 数据加载与预处理（按天累加 FRP）
data_dir = 'datasets'
csv_files = sorted(glob.glob(os.path.join(data_dir, 'modis_????_Australia.csv')))

df_list = []
for fp in csv_files:
    df = pd.read_csv(fp, parse_dates=['acq_date'])
    df_list.append(df[['acq_date', 'frp']])
all_df = pd.concat(df_list, ignore_index=True)

daily = (
    all_df
    .groupby(all_df['acq_date'].dt.date)['frp']
    .sum()
    .rename('frp_sum')
    .reset_index()
)
daily['date'] = pd.to_datetime(daily['acq_date'])
daily = daily.drop(columns=['acq_date'])

# 填补缺失日期
full_idx = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
daily = (
    daily
    .set_index('date')
    .reindex(full_idx, fill_value=0)
    .rename_axis('date')
    .reset_index()
)

# 归一化
scaler_y = MinMaxScaler()
daily['frp_scaled'] = scaler_y.fit_transform(daily[['frp_sum']])

# 构造滑动窗口
SEQ_LEN = 30
X, Y = [], []
for i in range(len(daily) - SEQ_LEN):
    X.append(daily['frp_scaled'].values[i : i + SEQ_LEN])
    Y.append(daily['frp_scaled'].values[i + SEQ_LEN])
X = np.array(X)[:, :, None]  # (N, SEQ_LEN, 1)
Y = np.array(Y)[:, None]     # (N, 1)

# 划分训练/验证
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]

# 2. Dataset & DataLoader
class FRPDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = FRPDataset(X_train, Y_train)
val_ds   = FRPDataset(X_val,   Y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)


# 3. LSTM 模型定义
class FRPLSTM(nn.Module):
    def __init__(self, in_dim=1, hid_dim=64, num_layers=1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hid_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # (batch, seq_len, hid*dirs)
        out = out[:, -1, :]        # 取最后时刻
        return self.fc(out)        # (batch, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = FRPLSTM().to(device)


# 4. 训练与验证函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch():
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_ds)

def evaluate():
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            p  = model(xb).cpu().numpy()
            preds.append(p); truths.append(yb.numpy())
    preds  = scaler_y.inverse_transform(np.vstack(preds))
    truths = scaler_y.inverse_transform(np.vstack(truths))

    mae  = mean_absolute_error(truths, preds)
    mse  = mean_squared_error(truths, preds)            # 默认返回 MSE
    rmse = np.sqrt(mse)                                 # 开根号得 RMSE
    r2   = r2_score(truths, preds)
    mape = np.mean(np.abs((truths - preds) / (truths + 1e-6)))
    conf = 1 - mape

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Confidence': conf}


# 5. 主训练循环 & 保存最佳模型
best_rmse = float('inf')
save_path = 'best_frp_lstm.pth'

for epoch in range(1, 51):
    tr_loss = train_epoch()
    metrics = evaluate()
    print(
        f"Epoch {epoch:02d} — "
        f"Train MSE: {tr_loss:.4f} | "
        f"Val RMSE: {metrics['RMSE']:.2f} | "
        f"MAE: {metrics['MAE']:.2f} | "
        f"R²: {metrics['R2']:.3f} | "
        f"Conf: {metrics['Confidence']:.3f}"
    )
    if metrics['RMSE'] < best_rmse:
        best_rmse = metrics['RMSE']
        torch.save(model.state_dict(), save_path)

print(f"Best model saved to {save_path}")


# 6. 加载模型并预测示例
def load_model(path, device=device):
    m = FRPLSTM().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

# 用验证集最后一个序列预测下一天 FRP
model_loaded = load_model(save_path)
last_seq = torch.from_numpy(X_val[-1:]).float().to(device)
with torch.no_grad():
    pred_scaled = model_loaded(last_seq).cpu().numpy()
pred_frp = scaler_y.inverse_transform(pred_scaled)[0, 0]
print(f"Predicted next-day FRP: {pred_frp:.2f} MW")
