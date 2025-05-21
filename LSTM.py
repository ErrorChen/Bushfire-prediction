import glob
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score
)

import matplotlib.pyplot as plt


# ─── FireDataset: wraps sliding-window arrays with per-sample weights ───
class FireDataset(Dataset):
    def __init__(self, X, Y_log, Y_raw, W):
        """
        X     : np.ndarray (N, T, F) – feature windows
        Y_log : np.ndarray (N,)      – log-scaled FRP targets
        Y_raw : np.ndarray (N,)      – raw FRP targets
        W     : np.ndarray (N,)      – sample weights
        """
        self.X     = torch.from_numpy(X.astype(np.float32))
        self.Y_log = torch.from_numpy(Y_log.astype(np.float32))
        self.Y_raw = torch.from_numpy(Y_raw.astype(np.float32))
        self.W     = torch.from_numpy(W.astype(np.float32))

    def __len__(self):
        # Subset implements this, so Pylance is happy
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_log[idx], self.Y_raw[idx], self.W[idx]


# ─── Multi-Head Self-Attention ───────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, "Embedding dim must be divisible by heads"
        self.heads = heads
        self.dk    = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.unify  = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch, seq_len, dim]
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


# ─── Improved Fire Prediction Model ──────────────────────────────────────
class ImprovedFireModel(nn.Module):
    def __init__(self, feat_dim, hid_dim=128, lstm_layers=3, heads=4, dropout=0.3):
        super().__init__()
        # 1D CNN for short-term patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Multi-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            hid_dim, hid_dim, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        # Self-attention + LayerNorm
        self.attn = MultiHeadSelfAttention(hid_dim * 2, heads=heads)
        self.norm = nn.LayerNorm(hid_dim * 2)
        # Deep FFN for final regression
        self.ffn  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2),  nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        c         = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # → [B, T, 2H]
        lstm_out, _ = self.lstm(c)                              # → [B, T, 2H]
        attn_out   = self.attn(lstm_out)                        # → [B, T, 2H]
        last       = lstm_out[:, -1, :] + attn_out[:, -1, :]    # residual sum
        h          = self.norm(last)                            # → [B, 2H]
        return self.ffn(h).squeeze(-1)                          # → [B]


# ─── ExtremeLoss: penalise large errors more heavily ─────────────────────
class ExtremeLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(reduction='none'), thr=0.05):
        super().__init__()
        self.base = base_loss
        self.thr  = thr

    def forward(self, pred, target, weight):
        err  = self.base(pred, target)
        mask = (err > self.thr).float()
        return ((1 + mask) * err * weight).mean()


# ─── Utility to split into train & val and wrap in DataLoaders ──────────
def make_loaders(X, Yl, Yr, W, split=0.8, batch_size=32):
    ds      = FireDataset(X, Yl, Yr, W)
    lengths = [int(len(ds) * split), len(ds) - int(len(ds) * split)]
    tr_ds, va_ds = random_split(ds, lengths)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return tr_ld, va_ld, tr_ds, va_ds


# ─── Main training loop with multi-criteria early stopping ──────────────
def main(dynamic_split=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Load & aggregate raw MODIS CSVs
    files = sorted(glob.glob('datasets/modis_????_Australia.csv'))
    df    = pd.concat([pd.read_csv(f, parse_dates=['acq_date']) for f in files],
                      ignore_index=True)
    df['acq_min'] = (df['acq_time'] // 100) * 60 + (df['acq_time'] % 100)
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

    # Ensure continuous dates
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily    = daily.reindex(full_idx, fill_value=0).rename_axis('date').reset_index()

    # 2. Prepare features & sliding windows
    daily['frp_log'] = np.log1p(daily['frp_sum'])
    feats           = [c for c in daily.columns if c not in ('date','frp_sum','frp_log')]
    scaler_X        = MinMaxScaler()
    scaler_y        = MinMaxScaler()
    X_all           = scaler_X.fit_transform(daily[feats]).astype(np.float32)
    y_log           = scaler_y.fit_transform(daily[['frp_log']]).flatten().astype(np.float32)
    y_raw           = daily['frp_sum'].to_numpy(dtype=np.float32)

    T, X, Yl, Yr = 30, [], [], []
    for i in range(len(X_all) - T):
        X.append(X_all[i:i+T])
        Yl.append(y_log[i+T])
        Yr.append(y_raw[i+T])
    X, Yl, Yr = map(np.stack, (X, Yl, Yr))

    # Sample weights (initially uniform)
    W = np.ones_like(Yr, dtype=np.float32)

    # 3. Initialise model, loss, optimiser, scheduler
    model     = ImprovedFireModel(len(feats)).to(device)
    criterion = ExtremeLoss(thr=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps= 500 * ((len(X) // 32) + 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=total_steps
    )

    # Prepare initial DataLoaders
    tr_ld, va_ld, tr_ds, va_ds = make_loaders(X, Yl, Yr, W)

    # Early-stop & history settings
    history     = {'train':[], 'val':[], 'comb':[]}
    best_comb   = float('inf')
    best_r2     = -float('inf')
    best_accrel = -float('inf')
    best_ep     = 0
    wait        = 0
    EPOCHS      = 500
    PATIENCE    = 50
    MIN_DELTA   = 10.0
    eps_rel     = 0.266

    for ep in range(1, EPOCHS+1):
        if dynamic_split:
            # Optionally reshuffle train/val each epoch
            tr_ld, va_ld, tr_ds, va_ds = make_loaders(X, Yl, Yr, W)

        # — Train —
        model.train()
        tr_loss = 0.0
        for xb, yb_log, _, wb in tr_ld:
            xb, yb_log, wb = xb.to(device), yb_log.to(device), wb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb_log, wb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * xb.size(0)
        # <<< FIX HERE: use len(tr_ds), not len(tr_ld.dataset)
        tr_loss /= len(tr_ds)
        history['train'].append(tr_loss)

        # — Validate —
        model.eval()
        va_loss, all_p, all_t = 0.0, [], []
        with torch.no_grad():
            for xb, yb_log, yb_raw, _ in va_ld:
                xb, yb_log = xb.to(device), yb_log.to(device)
                p = model(xb)
                va_loss += criterion(p, yb_log, torch.ones_like(yb_log)).item() * xb.size(0)
                all_p.append(p.cpu().numpy())
                all_t.append(yb_raw.numpy())
        # <<< FIX HERE: use len(va_ds), not len(va_ld.dataset)
        va_loss /= len(va_ds)
        history['val'].append(va_loss)

        # Invert log-scale & clip negatives only
        preds   = np.expm1(scaler_y.inverse_transform(
                      np.concatenate(all_p).reshape(-1,1)
                  )).flatten()
        preds   = np.clip(preds, 0, None)
        truths  = np.concatenate(all_t)

        # Metrics computation
        mae     = mean_absolute_error(truths, preds)
        rmse    = np.sqrt(mean_squared_error(truths, preds))
        r2m     = r2_score(truths, preds)
        frac_err= np.abs(preds - truths) / np.where(truths>0, truths, 1.0)
        corr_rel= frac_err <= eps_rel
        acc_rel = accuracy_score(corr_rel, np.ones_like(corr_rel))
        prec_rel= precision_score(corr_rel, np.ones_like(corr_rel), zero_division=0)

        comb_val= mae + 0.1 * rmse
        history['comb'].append(comb_val)
        comb_sm = np.mean(history['comb'][-5:]) if len(history['comb'])>=5 else comb_val

        # Epoch summary
        print(
            f"Epoch {ep}/{EPOCHS} | Tr {tr_loss:.4f} Va {va_loss:.4f} | "
            f"MAE {mae:.1f} RMSE {rmse:.1f} R² {r2m:.3f} | "
            f"Acc_rel@{eps_rel:.3f} {acc_rel:.3f} Prec_rel@{prec_rel:.3f} | "
            f"Comb_sm {comb_sm:.2f}"
        )

        # Multi-criteria early stopping
        improved = False
        if comb_sm < best_comb - MIN_DELTA:
            best_comb = comb_sm; improved = True
        if r2m > best_r2 + 1e-3:
            best_r2 = r2m; improved = True
        if acc_rel > best_accrel + 1e-3:
            best_accrel = acc_rel; improved = True

        if improved:
            best_ep = ep; wait = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= PATIENCE:
                print(
                    f"Early stopping at epoch {ep} – no real improvement "
                    f"in last {PATIENCE} epochs"
                )
                break

    # Final report
    print(
        f"\nBest model at epoch {best_ep}  "
        f"Comb_sm {best_comb:.2f}  R² {best_r2:.3f}  Acc_rel {best_accrel:.3f}"
    )

    # Plot training vs validation loss
    epochs = range(1, len(history['train'])+1)
    plt.figure(figsize=(8,4))
    plt.plot(epochs, history['train'], label='Train Loss')
    plt.plot(epochs, history['val'],   label='Val   Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend(); plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set dynamic_split=True to reshuffle train/val each epoch
    main(dynamic_split=False)
