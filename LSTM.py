import os
import glob
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
data_dir = 'datasets'
# 查找所有 MODIS CSV 文件
csv_files = sorted(glob.glob(os.path.join(data_dir, 'modis_????_Australia.csv')))

# 逐个读取并合并
df_list = []
for fp in csv_files:
    df = pd.read_csv(fp, parse_dates=['acq_date'])
    df_list.append(df[['acq_date']])
data = pd.concat(df_list, ignore_index=True)

# 按天统计火点数量
daily = data.groupby(data['acq_date'].dt.date).size().rename('fires').reset_index()
daily['acq_date'] = pd.to_datetime(daily['acq_date'])

# 完成日期索引（填补缺失天数）
full_idx = pd.date_range(daily['acq_date'].min(), daily['acq_date'].max(), freq='D')
daily = daily.set_index('acq_date').reindex(full_idx, fill_value=0).rename_axis('date').reset_index()

# 归一化
scaler = MinMaxScaler()
daily['fires_scaled'] = scaler.fit_transform(daily[['fires']])

# 构造滑动窗口序列
SEQ_LEN = 30  # 用前 30 天预测下一天
X, Y = [], []
for i in range(len(daily) - SEQ_LEN):
    X.append(daily['fires_scaled'].values[i:i+SEQ_LEN])
    Y.append(daily['fires_scaled'].values[i+SEQ_LEN])
X = np.array(X)[:, :, None]  # (N, SEQ_LEN, 1)
Y = np.array(Y)[:, None]     # (N, 1)

# 划分训练/验证
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]


# 2. 自定义 Dataset
class FireDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = FireDataset(X_train, Y_train)
val_ds   = FireDataset(X_val,   Y_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)


# 3. LSTM 模型定义
class FireLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)        # out: (batch, seq_len, hidden_dim*directions)
        out = out[:, -1, :]          # 取最后时刻输出
        return self.fc(out)          # (batch, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FireLSTM().to(device)


# 4. 训练与评估
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch():
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_ds)

def eval_epoch():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * xb.size(0)
    return total_loss / len(val_ds)

# 训练循环
EPOCHS = 50
history = {'train': [], 'val': []}
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_epoch()
    va_loss = eval_epoch()
    history['train'].append(tr_loss)
    history['val'].append(va_loss)
    print(f'Epoch {epoch:02d} — Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}')

# 绘制损失曲线
plt.plot(history['train'], label='Train')
plt.plot(history['val'],   label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
