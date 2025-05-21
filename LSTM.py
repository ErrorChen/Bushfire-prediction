import glob
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


class FireDataset(Dataset):
    def __init__(self, X, Y_log, Y_raw, W):
        self.X     = torch.from_numpy(X.astype(np.float32))
        self.Y_log = torch.from_numpy(Y_log.astype(np.float32))
        self.Y_raw = torch.from_numpy(Y_raw.astype(np.float32))
        self.W     = torch.from_numpy(W.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y_log[i], self.Y_raw[i], self.W[i]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, "Embedding dim must divide by heads"
        self.dk = dim // heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.unify  = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qs, ks, vs = [
            t.view(B, T, self.heads, self.dk).transpose(1, 2)
            for t in qkv
        ]
        scores = torch.matmul(qs, ks.transpose(-2, -1)) / np.sqrt(self.dk)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(attn, vs) \
                    .transpose(1, 2) \
                    .contiguous() \
                    .view(B, T, D)
        return self.unify(out)


class ImprovedFireModel(nn.Module):
    def __init__(self, feat_dim, hid_dim=128, lstm_layers=3, heads=4, dropout=0.3):
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
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)
        )

    def forward(self, x):
        c, (B, T, _) = self.cnn(x.transpose(1, 2)).transpose(1, 2), x.size()
        lstm_out, _  = self.lstm(c)
        attn_out     = self.attn(lstm_out)
        last         = lstm_out[:, -1, :] + attn_out[:, -1, :]
        h            = self.norm(last)
        return self.ffn(h).squeeze(-1)


class ExtremeLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(reduction='none'), thr=0.05):
        super().__init__()
        self.base = base_loss
        self.thr  = thr

    def forward(self, pred, target, weight):
        err  = self.base(pred, target)
        mask = (err > self.thr).float()
        return ((1 + mask) * err * weight).mean()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Load & preprocess
    files = sorted(glob.glob('datasets/modis_????_Australia.csv'))
    df    = pd.concat([pd.read_csv(f, parse_dates=['acq_date']) for f in files],
                      ignore_index=True)
    df['acq_min'] = (df['acq_time']//100)*60 + (df['acq_time']%100)
    df['date']    = df['acq_date'].dt.floor('D')

    daily = df.groupby('date').agg(
        latitude_mean    = pd.NamedAgg('latitude',   'mean'),
        latitude_max     = pd.NamedAgg('latitude',   'max'),
        longitude_mean   = pd.NamedAgg('longitude',  'mean'),
        longitude_max    = pd.NamedAgg('longitude',  'max'),
        brightness_mean  = pd.NamedAgg('brightness', 'mean'),
        brightness_max   = pd.NamedAgg('brightness', 'max'),
        bright_t31_mean  = pd.NamedAgg('bright_t31', 'mean'),
        bright_t31_max   = pd.NamedAgg('bright_t31', 'max'),
        scan_mean        = pd.NamedAgg('scan',       'mean'),
        scan_max         = pd.NamedAgg('scan',       'max'),
        track_mean       = pd.NamedAgg('track',      'mean'),
        track_max        = pd.NamedAgg('track',      'max'),
        confidence_mean  = pd.NamedAgg('confidence', 'mean'),
        confidence_max   = pd.NamedAgg('confidence', 'max'),
        acq_min_mean     = pd.NamedAgg('acq_min',    'mean'),
        frp_sum          = pd.NamedAgg('frp',        'sum'),
        satellite_count  = pd.NamedAgg('satellite',  'count'),
        instrument_count = pd.NamedAgg('instrument', 'count'),
        version_count    = pd.NamedAgg('version',    'count'),
        daynight_count   = pd.NamedAgg('daynight',   'count'),
        type_count       = pd.NamedAgg('type',       'count')
    )
    idx   = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(idx, fill_value=0).rename_axis('date').reset_index()

    # Features & targets
    daily['frp_log'] = np.log1p(daily['frp_sum'])
    feat_cols       = [c for c in daily.columns if c not in ('date','frp_sum','frp_log')]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_all    = scaler_X.fit_transform(daily[feat_cols]).astype(np.float32)
    y_log    = scaler_y.fit_transform(daily[['frp_log']]).flatten().astype(np.float32)
    y_raw    = daily['frp_sum'].to_numpy(dtype=np.float32)

    # Sliding window
    T, X, Yl, Yr = 30, [], [], []
    for i in range(len(X_all)-T):
        X.append(X_all[i:i+T])
        Yl.append(y_log[i+T])
        Yr.append(y_raw[i+T])
    X, Yl, Yr = map(np.stack, (X, Yl, Yr))

    # Sample weights
    p90 = np.percentile(y_raw, 90)
    W   = ((y_raw > p90)*1.5 + 1.0).astype(np.float32)

    # Split
    n, nt  = len(X), int(0.8*len(X))
    split  = lambda arr: (arr[:nt], arr[nt:])
    X_tr, X_va  = split(X)
    Yl_tr,Yl_va = split(Yl)
    Yr_tr,Yr_va = split(Yr)
    W_tr, W_va  = split(W)

    tr_ds = FireDataset(X_tr, Yl_tr, Yr_tr, W_tr)
    va_ds = FireDataset(X_va, Yl_va, Yr_va, W_va)
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True,  pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, pin_memory=True)

    # Model setup
    model     = ImprovedFireModel(len(feat_cols)).to(device)
    criterion = ExtremeLoss(thr=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=500*len(tr_ld)
    )

    # Training with best-epoch tracking
    EPOCHS, PATIENCE, MIN_DELTA = 500, 50, 10.0
    eps_rel   = 0.266
    best_score, wait = float('inf'), 0
    best_epoch      = 0
    history = {'train':[], 'val':[], 'comb':[]}

    for ep in range(1, EPOCHS+1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb_log, _, wb in tr_ld:
            xb, yb_log, wb = xb.to(device), yb_log.to(device), wb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb_log, wb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss /= len(tr_ds)
        history['train'].append(tr_loss)

        # Validate
        model.eval()
        va_loss, all_p, all_t = 0.0, [], []
        with torch.no_grad():
            for xb, yb_log, yb_raw, _ in va_ld:
                xb, yb_log = xb.to(device), yb_log.to(device)
                p = model(xb)
                va_loss += criterion(p, yb_log, torch.ones_like(yb_log)).item()*xb.size(0)
                all_p.append(p.cpu().numpy()); all_t.append(yb_raw.numpy())
        va_loss /= len(va_ds)
        history['val'].append(va_loss)

        # Metrics
        preds  = np.expm1(scaler_y.inverse_transform(np.concatenate(all_p)[:,None])).flatten()
        truths = np.concatenate(all_t)
        mae    = mean_absolute_error(truths, preds)
        rmse   = np.sqrt(mean_squared_error(truths, preds))
        r2     = r2_score(truths, preds)
        frac_err = np.abs(preds-truths)/np.where(truths>0, truths,1.0)
        corr_rel = frac_err <= eps_rel
        acc_rel  = accuracy_score(corr_rel, np.ones_like(corr_rel))
        prec_rel = precision_score(corr_rel, np.ones_like(corr_rel), zero_division=0)

        # Combined & smoothed metric
        comb_val = mae + 0.1*rmse
        history['comb'].append(comb_val)
        sm = np.mean(history['comb'][-5:]) if len(history['comb'])>=5 else comb_val

        print(
            f"Epoch {ep}/{EPOCHS} | Tr {tr_loss:.4f} Va {va_loss:.4f} | "
            f"MAE {mae:.1f} RMSE {rmse:.1f} R² {r2:.3f} | "
            f"Acc_rel@{eps_rel:.3f} {acc_rel:.3f} Prec_rel {prec_rel:.3f} | "
            f"Comb_sm {sm:.2f}"
        )

        # Early stopping and best‐epoch tracking
        if sm < best_score - MIN_DELTA:
            best_score, wait = sm, 0
            best_epoch       = ep
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep} (no real improvement)")
                break

    print(f"\nBest model saved at epoch {best_epoch} with smoothed metric {best_score:.2f}")

    # Plot losses
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()


if __name__ == "__main__":
    main()
