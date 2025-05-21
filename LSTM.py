import os, glob
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score
)

import matplotlib.pyplot as plt

# ─── Dataset ────────────────────────────────────────────────────────────────
class FireDataset(Dataset):
    def __init__(self, X, Y_log, Y_raw, W):
        self.X      = torch.from_numpy(X.astype(np.float32))
        self.Y_log  = torch.from_numpy(Y_log.astype(np.float32))
        self.Y_raw  = torch.from_numpy(Y_raw.astype(np.float32))
        self.W      = torch.from_numpy(W.astype(np.float32))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y_log[i], self.Y_raw[i], self.W[i]

# ─── Multi-Head Self-Attention ───────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads, self.dim = heads, dim
        assert dim % heads == 0
        self.dk = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.unify  = nn.Linear(dim, dim)
    def forward(self, x):
        B, T, D = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qs, ks, vs = [t.view(B, T, self.heads, self.dk)
                      .transpose(1,2) for t in qkv]
        scores = torch.matmul(qs, ks.transpose(-2,-1)) / np.sqrt(self.dk)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(attn, vs)\
                    .transpose(1,2)\
                    .contiguous()\
                    .view(B, T, D)
        return self.unify(out)

# ─── Model: CNN–LSTM + Attention + FFN ────────────────────────────────────────
class ImprovedFireModel(nn.Module):
    def __init__(self, feat_dim, hid_dim=128, lstm_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            hid_dim, hid_dim, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attn = MultiHeadSelfAttention(hid_dim * 2, heads=heads)
        self.norm = nn.LayerNorm(hid_dim * 2)
        self.ffn  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        c        = self.cnn(x.transpose(1,2)).transpose(1,2)  # (B, T, 2H)
        out, _   = self.lstm(c)                              # (B, T, 2H)
        attn_out = self.attn(out)                            # (B, T, 2H)
        last_lstm = out[:, -1, :]                            # (B, 2H)
        last_attn = attn_out[:, -1, :]                       # (B, 2H)
        h         = self.norm(last_lstm + last_attn)
        return self.ffn(h).squeeze(-1)                       # (B,)

# ─── Adaptive Extreme Loss ───────────────────────────────────────────────────
class ExtremeLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(reduction='none'), thr=0.05):
        super().__init__()
        self.base, self.thr = base_loss, thr
    def forward(self, pred, target, weight):
        err  = self.base(pred, target)
        mask = (err > self.thr).float()
        return ((1 + mask) * err * weight).mean()

# ─── Main Training Loop ───────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. 数据加载与聚合
    files = sorted(glob.glob('datasets/modis_????_Australia.csv'))
    df    = pd.concat([pd.read_csv(f, parse_dates=['acq_date']) for f in files],
                      ignore_index=True)
    df['date'] = df['acq_date'].dt.floor('D')
    daily = df.groupby('date').agg(
        count=('frp','size'),
        mb=('brightness','mean'),
        Nb=('brightness','max'),
        mc=('confidence','mean'),
        ms=('scan','mean'),
        mt=('track','mean'),
        dcount=('daynight',lambda x:(x=='D').sum()),
        ncount=('daynight',lambda x:(x=='N').sum()),
        frp_raw=('frp','sum')
    ).reset_index()
    idx   = pd.date_range(daily.date.min(), daily.date.max(), freq='D')
    daily = (daily.set_index('date')
                 .reindex(idx, fill_value=0)
                 .rename_axis('date')
                 .reset_index())

    # 2. 归一化与滑窗（T=30, 步长=1）
    daily['frp_log'] = np.log1p(daily['frp_raw'])
    feats    = ['count','mb','Nb','mc','ms','mt','dcount','ncount']
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_all    = scaler_X.fit_transform(daily[feats]).astype(np.float32)
    y_log    = scaler_y.fit_transform(daily[['frp_log']]).flatten().astype(np.float32)
    y_raw    = daily['frp_raw'].to_numpy(dtype=np.float32)

    T, F = 30, len(feats)
    X, Yl, Yr = [], [], []
    for i in range(len(X_all)-T):
        X.append(X_all[i:i+T])
        Yl.append(y_log[i+T])
        Yr.append(y_raw[i+T])
    X, Yl, Yr = map(np.stack, (X, Yl, Yr))

    # 3. 样本权重（90th 百分位加权）
    p90 = np.percentile(y_raw, 90)
    W   = ((y_raw > p90) * 1.5 + 1.0).astype(np.float32)

    # 4. 划分训练/验证集（80/20）
    n   = len(X); nt = int(0.8 * n)
    slices = lambda arr: (arr[:nt], arr[nt:])
    X_tr, X_va = slices(X)
    Yl_tr, Yl_va = slices(Yl)
    Yr_tr, Yr_va = slices(Yr)
    W_tr, W_va   = slices(W)

    tr_ds = FireDataset(X_tr, Yl_tr, Yr_tr, W_tr)
    va_ds = FireDataset(X_va, Yl_va, Yr_va, W_va)
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # 5. 模型、损失、优化器与调度器
    model     = ImprovedFireModel(F).to(device)
    criterion = ExtremeLoss(thr=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = 500 * len(tr_ld)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=total_steps
    )

    # 6. 训练参数 & ε 设置
    EPOCHS, PATIENCE, MIN_DELTA = 500, 20, 1e-3
    eps_abs = 500.0        # 绝对容差（原参考）
    eps_rel = 0.266        # 相对容差（测量不确定度 26.6%）【:contentReference[oaicite:3]{index=3}】

    best_score, wait = float('inf'), 0
    history = {'train':[], 'val':[]}

    for ep in range(1, EPOCHS+1):
        # — 训练 —
        model.train()
        tr_loss = 0.0
        for xb, yb_log, _, wb in tr_ld:
            xb, yb_log, wb = xb.to(device), yb_log.to(device), wb.to(device)
            pred_log = model(xb)
            loss     = criterion(pred_log, yb_log, wb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_ds)
        history['train'].append(tr_loss)

        # — 验证 —
        model.eval()
        va_loss = 0.0
        preds_log, truths = [], []
        with torch.no_grad():
            for xb, yb_log, yb_raw, _ in va_ld:
                xb, yb_log = xb.to(device), yb_log.to(device)
                pred_log = model(xb)
                va_loss  += criterion(pred_log, yb_log, torch.ones_like(yb_log)).item() * xb.size(0)
                preds_log.append(pred_log.cpu().numpy())
                truths.append(yb_raw.numpy())
        va_loss /= len(va_ds)
        history['val'].append(va_loss)

        # 反归一化
        preds_log = np.concatenate(preds_log)
        truths    = np.concatenate(truths)
        preds_frp = np.expm1(scaler_y.inverse_transform(preds_log[:,None])).flatten()

        # — 回归指标 —
        mae  = mean_absolute_error(truths, preds_frp)
        rmse = np.sqrt(mean_squared_error(truths, preds_frp))
        r2   = r2_score(truths, preds_frp)

        # — 精度评估 —  
        # 绝对容差
        corr_abs = np.abs(preds_frp - truths) <= eps_abs
        acc_abs  = accuracy_score(corr_abs, np.ones_like(corr_abs))
        prec_abs = precision_score(corr_abs, np.ones_like(corr_abs), zero_division=0)
        # 相对容差
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_err = np.abs(preds_frp - truths) / np.where(truths>0, truths, 1.0)
        corr_rel = frac_err <= eps_rel
        acc_rel  = accuracy_score(corr_rel, np.ones_like(corr_rel))
        prec_rel = precision_score(corr_rel, np.ones_like(corr_rel), zero_division=0)

        print(
            f"Epoch {ep}/{EPOCHS} | Tr: {tr_loss:.4f} Va: {va_loss:.4f} | "
            f"MAE: {mae:.1f} RMSE: {rmse:.1f} R²: {r2:.3f} | "
            f"Acc@±{int(eps_abs)}: {acc_abs:.3f} Prec@±{int(eps_abs)}: {prec_abs:.3f} | "
            f"Acc_rel@{eps_rel:.3f}: {acc_rel:.3f} Prec_rel@{eps_rel:.3f}: {prec_rel:.3f}"
        )

        # — 改进 EarlyStopping（综合 MAE + 0.1*RMSE）—
        score = mae + 0.1 * rmse
        if best_score - score > MIN_DELTA:
            best_score, wait = score, 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep} (no improvement ≥{MIN_DELTA} for {PATIENCE} epochs)")
                break

    # — 绘制损失曲线 —
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'],   label='Val   Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()
