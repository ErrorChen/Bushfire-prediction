import os, glob
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
        qs, ks, vs = [
            t.view(B, T, self.heads, self.dk).transpose(1,2)
            for t in qkv
        ]
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
        # 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # LSTM
        self.lstm = nn.LSTM(
            hid_dim, hid_dim, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        # Attention & FFN
        self.attn = MultiHeadSelfAttention(hid_dim * 2, heads=heads)
        self.norm = nn.LayerNorm(hid_dim * 2)
        self.ffn  = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        c        = self.cnn(x.transpose(1,2))  # (B, H, T)
        c        = c.transpose(1,2)            # (B, T, H)
        out, _   = self.lstm(c)                # (B, T, 2H)
        attn_out = self.attn(out)              # (B, T, 2H)
        # 取最后 time-step
        last_lstm = out[:, -1, :]              # (B, 2H)
        last_attn = attn_out[:, -1, :]         # (B, 2H)
        h         = self.norm(last_lstm + last_attn)
        return self.ffn(h).squeeze(-1)         # (B,)

# ─── Adaptive Extreme Loss ───────────────────────────────────────────────────
class ExtremeLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(reduction='none'), thr=0.05):
        super().__init__()
        self.base, self.thr = base_loss, thr
    def forward(self, pred, target, weight):
        # pred & target: (B,), weight: (B,)
        err  = self.base(pred, target)
        mask = (err > self.thr).float()
        return ((1 + mask) * err * weight).mean()

# ─── Main Training Loop ───────────────────────────────────────────────────────
def main():
    mp.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Load & aggregate daily
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
    daily = daily.set_index('date').reindex(idx, fill_value=0)\
                 .rename_axis('date').reset_index()

    # 2. Log-transform & scale
    daily['frp_log'] = np.log1p(daily['frp_raw'])
    feats    = ['count','mb','Nb','mc','ms','mt','dcount','ncount']
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_all    = scaler_X.fit_transform(daily[feats]).astype(np.float32)
    y_log    = scaler_y.fit_transform(daily[['frp_log']]).flatten().astype(np.float32)
    y_raw    = daily['frp_raw'].to_numpy(dtype=np.float32)

    # 3. Sliding window
    T, F = 30, len(feats)
    X, Yl, Yr = [], [], []
    for i in range(len(X_all)-T):
        X.append(X_all[i:i+T])
        Yl.append(y_log[i+T])
        Yr.append(y_raw[i+T])
    X   = np.stack(X)
    Yl  = np.array(Yl)
    Yr  = np.array(Yr)

    # 4. Weight by 90th percentile
    p90 = np.percentile(y_raw, 90)
    W   = ((y_raw > p90) * 1.5 + 1.0).astype(np.float32)

    # 5. Split train/val
    n   = len(X); nt = int(0.8 * n)
    by = lambda a: (a[:nt], a[nt:])
    X_tr, X_va = by(X)
    Yl_tr, Yl_va = by(Yl)
    Yr_tr, Yr_va = by(Yr)
    W_tr, W_va   = by(W)

    tr_ds = FireDataset(X_tr, Yl_tr, Yr_tr, W_tr)
    va_ds = FireDataset(X_va, Yl_va, Yr_va, W_va)
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True,  pin_memory=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

    # 6. Model, loss, optim, scheduler
    model     = ImprovedFireModel(F).to(device)
    criterion = ExtremeLoss(thr=0.05)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10
                )

    # 7. Training settings
    EPOCHS, PATIENCE, MIN_DELTA = 500, 20, 1e-3
    eps = 500.0  # error tolerance for accuracy/precision
    best_score, wait = float('inf'), 0
    history = {'train':[], 'val':[]}

    for ep in range(1, EPOCHS+1):
        # — Train —
        model.train()
        train_loss = 0.0
        for xb, yb_log, _, wb in tr_ld:
            xb, yb_log, wb = xb.to(device), yb_log.to(device), wb.to(device)
            pred_log = model(xb)
            loss     = criterion(pred_log, yb_log, wb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(tr_ds)
        history['train'].append(train_loss)

        # — Validate —
        model.eval()
        val_loss = 0.0
        preds_log = []
        truths    = []
        with torch.no_grad():
            for xb, yb_log, yb_raw, _ in va_ld:
                xb, yb_log = xb.to(device), yb_log.to(device)
                pred_log = model(xb)
                val_loss += criterion(pred_log, yb_log, torch.ones_like(yb_log)).item() * xb.size(0)
                preds_log.append(pred_log.cpu().numpy())
                truths.append(yb_raw.numpy())
        val_loss /= len(va_ds)
        history['val'].append(val_loss)
        scheduler.step(val_loss)

        # — Metrics —
        preds_log = np.concatenate(preds_log)
        preds_frp = np.expm1(scaler_y.inverse_transform(preds_log[:,None])).flatten()
        truths    = np.concatenate(truths)

        mae   = mean_absolute_error(truths, preds_frp)
        rmse  = np.sqrt(mean_squared_error(truths, preds_frp))
        r2    = r2_score(truths, preds_frp)

        # Accuracy/Precision within eps
        correct = np.abs(preds_frp - truths) <= eps
        acc_eps = accuracy_score(correct, np.ones_like(correct))
        prec_eps= precision_score(correct, np.ones_like(correct), zero_division=0)

        print(
            f"Epoch {ep}/{EPOCHS} | "
            f"TrH: {train_loss:.4f} | VaH: {val_loss:.4f} | "
            f"MAE: {mae:.1f} | RMSE: {rmse:.1f} | R²: {r2:.3f} | "
            f"Acc@{int(eps)}: {acc_eps:.3f} | Prec@{int(eps)}: {prec_eps:.3f}"
        )

        # — Improved EarlyStopping —
        score = mae + 0.1 * rmse
        if best_score - score > MIN_DELTA:
            best_score, wait = score, 0
            torch.save(model.state_dict(), 'best_improved.pth')
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep} (no improvement ≥{MIN_DELTA} for {PATIENCE} epochs)")
                break

    # 8. Plot Loss curves
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()
