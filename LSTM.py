import os
import glob
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score
)

import matplotlib.pyplot as plt

# --- 数据集定义 ---
class FireDataset(Dataset):
    def __init__(self, X, Y, W):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float().unsqueeze(1)
        self.W = torch.from_numpy(W).float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]

# --- 注意力模块 ---
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
    def forward(self, lstm_out):       # lstm_out: (B, T, D)
        scores = self.score(lstm_out) # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, D)
        return context

# --- 主模型 ---
class FireLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        d = hid_dim * (2 if bidirectional else 1)
        self.attn = Attention(d)
        self.norm = nn.LayerNorm(d)
        self.fc   = nn.Linear(d, 1)

    def forward(self, x):
        out, _ = self.lstm(x)    # (B, T, D)
        ctx    = self.attn(out)  # (B, D)
        ctx    = self.norm(ctx)
        return self.fc(ctx)      # (B, 1)

# --- 入口函数 ---
def main():
    mp.freeze_support()

    # 设备准备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. 读取并按天汇总
    data_dir = 'datasets'
    files = sorted(glob.glob(os.path.join(data_dir, 'modis_????_Australia.csv')))
    df = pd.concat([pd.read_csv(f, parse_dates=['acq_date']) for f in files], ignore_index=True)
    df['date'] = df['acq_date'].dt.floor('D')
    daily = df.groupby('date').agg(
        count_fires     = ('frp',       'size'),
        mean_brightness = ('brightness','mean'),
        max_brightness  = ('brightness','max'),
        mean_confidence = ('confidence','mean'),
        mean_scan       = ('scan',      'mean'),
        mean_track      = ('track',     'mean'),
        day_count       = ('daynight', lambda x: (x=='D').sum()),
        night_count     = ('daynight', lambda x: (x=='N').sum()),
        frp_sum         = ('frp',       'sum')
    ).reset_index()
    full_idx = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = (daily.set_index('date')
                  .reindex(full_idx)
                  .fillna(0)
                  .rename_axis('date')
                  .reset_index())

    # 对数化目标：log(FRP+1)
    daily['frp_log'] = np.log1p(daily['frp_sum'])

    # 特征列与目标
    feature_cols = [
        'count_fires','mean_brightness','max_brightness',
        'mean_confidence','mean_scan','mean_track',
        'day_count','night_count'
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_all = scaler_X.fit_transform(daily[feature_cols])
    y_all = scaler_y.fit_transform(daily[['frp_log']]).flatten()

    # 构造滑动窗口
    SEQ_LEN = 30
    X, Y = [], []
    for i in range(len(X_all) - SEQ_LEN):
        X.append(X_all[i:i+SEQ_LEN])
        Y.append(y_all[i+SEQ_LEN])
    X = np.stack(X)  # (N, T, F)
    Y = np.array(Y)  # (N,)

    # 样本权重：90th 百分位以上加权
    thr90 = np.percentile(daily['frp_sum'], 90)
    W = np.array([
        2.0 if daily['frp_sum'].iloc[i+SEQ_LEN] > thr90 else 1.0
        for i in range(len(X))
    ])

    # 划分训练/验证
    n_train = int(0.8 * len(X))
    X_tr, X_va = X[:n_train], X[n_train:]
    Y_tr, Y_va = Y[:n_train], Y[n_train:]
    W_tr, W_va = W[:n_train], W[n_train:]

    train_ds = FireDataset(X_tr, Y_tr, W_tr)
    val_ds   = FireDataset(X_va, Y_va, W_va)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=32,
        num_workers=0, pin_memory=True
    )

    # 模型、损失、优化、调度
    model = FireLSTM(input_dim=len(feature_cols)).to(device)
    criterion = nn.SmoothL1Loss(reduction='none')  # HuberLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练参数
    EPOCHS   = 500
    PATIENCE = 60
    fire_thr  = 10.0   # moderate fire 阈值（MW）:contentReference[oaicite:3]{index=3}
    large_thr = 100.0  # large fire 阈值（MW）:contentReference[oaicite:4]{index=4}

    history = {'train_loss': [], 'val_loss': []}
    best_mae, best_ep = float('inf'), 0

    # 逐 Epoch 训练
    for ep in range(1, EPOCHS+1):
        # —— Train —— #
        model.train()
        total_t = 0.0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb).mul(wb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_t += loss.item() * xb.size(0)
        tr_loss = total_t / len(train_ds)
        history['train_loss'].append(tr_loss)

        # —— Eval —— #
        model.eval()
        total_v, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                total_v += criterion(p, yb).mean().item() * xb.size(0)
                preds.append(p.cpu().numpy())
                truths.append(yb.cpu().numpy())
        val_loss = total_v / len(val_ds)
        history['val_loss'].append(val_loss)

        # 调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'LR reduced: {old_lr:.5f} → {new_lr:.5f}')

        # 反变换至实际 FRP（MW）
        preds_act  = np.expm1(scaler_y.inverse_transform(np.vstack(preds)))
        truths_act = np.expm1(scaler_y.inverse_transform(np.vstack(truths)))

        # 回归指标
        mae  = mean_absolute_error(truths_act, preds_act)
        rmse = np.sqrt(mean_squared_error(truths_act, preds_act))
        r2   = r2_score(truths_act, preds_act)
        mape = np.mean(np.abs((truths_act - preds_act) / (truths_act + 1e-6)))
        conf = 1 - mape

        # 二分类指标：中等火 & 大火
        true_mod = (truths_act > fire_thr).astype(int)
        pred_mod = (preds_act   > fire_thr).astype(int)
        acc_mod  = accuracy_score(true_mod, pred_mod)
        prec_mod = precision_score(true_mod, pred_mod, zero_division=0)

        true_large = (truths_act > large_thr).astype(int)
        pred_large = (preds_act   > large_thr).astype(int)
        acc_l = accuracy_score(true_large, pred_large)
        prec_l= precision_score(true_large, pred_large, zero_division=0)

        print(
            f"Epoch {ep:03d}/{EPOCHS} — "
            f"TrHuber: {tr_loss:.5f} | ValHuber: {val_loss:.5f} | "
            f"MAE: {mae:.1f} | RMSE: {rmse:.1f} | R²: {r2:.3f} | "
            f"Acc_mod: {acc_mod:.3f} | Prec_mod: {prec_mod:.3f} | "
            f"Acc_large: {acc_l:.3f} | Prec_large: {prec_l:.3f}"
        )

        # EarlyStopping (监控 MAE)
        if mae < best_mae:
            best_mae, best_ep = mae, ep
            torch.save(model.state_dict(), 'best_fire_lstm.pth')
        elif ep - best_ep >= PATIENCE:
            print(f"Early stopping at epoch {ep}, best MAE {best_mae:.1f} at epoch {best_ep}")
            break

    # 绘制损失曲线
    plt.figure()
    plt.plot(history['train_loss'], label='Train Huber')
    plt.plot(history['val_loss'],   label='Val Huber')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()
