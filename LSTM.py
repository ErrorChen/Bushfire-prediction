import glob
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
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
    def __init__(self, feat_dim, hid_dim=128, lstm_layers=3,
                 heads=4, dropout=0.4):
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
    tr_ld = DataLoader(tr_ds, batch_size=batch_size,
                       shuffle=True,  pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size,
                       shuffle=False, pin_memory=True)
    return tr_ld, va_ld, tr_ds, va_ds


# ─── Main training loop with adaptive LR and always-dynamic split ────────
def main(dynamic_split=True):
    """
    dynamic_split: always reshuffle train/val split before each epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Load & aggregate raw MODIS CSVs
    files = sorted(glob.glob('datasets/modis_????_Australia.csv'))
    df    = pd.concat([pd.read_csv(f, parse_dates=['acq_date'])
                       for f in files], ignore_index=True)
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
    full_idx = pd.date_range(daily.index.min(),
                             daily.index.max(), freq='D')
    daily    = daily.reindex(full_idx, fill_value=0) \
                    .rename_axis('date').reset_index()

    # 2. Prepare features & sliding windows
    daily['frp_log'] = np.log1p(daily['frp_sum'])
    feats           = [c for c in daily.columns
                       if c not in ('date','frp_sum','frp_log')]
    scaler_X        = MinMaxScaler()
    scaler_y        = MinMaxScaler()
    X_all           = scaler_X.fit_transform(
                        daily[feats]).astype(np.float32)
    y_log           = scaler_y.fit_transform(
                        daily[['frp_log']]).flatten().astype(np.float32)
    y_raw           = daily['frp_sum'].to_numpy(dtype=np.float32)

    T, X, Yl, Yr = 30, [], [], []
    for i in range(len(X_all) - T):
        X.append(X_all[i:i+T])
        Yl.append(y_log[i+T])
        Yr.append(y_raw[i+T])
    X, Yl, Yr = map(np.stack, (X, Yl, Yr))
    W = np.ones_like(Yr, dtype=np.float32)

    # 3. Initialise model, loss, optimiser
    model      = ImprovedFireModel(len(feats)).to(device)
    criterion  = ExtremeLoss(thr=0.05)
    optimiser  = optim.AdamW(model.parameters(),
                             lr=1e-3, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min',
        factor=0.5, patience=10, min_lr=1e-6
    )

    # Prepare initial DataLoaders
    tr_ld, va_ld, tr_ds, va_ds = make_loaders(X, Yl, Yr, W)

    # Early-stop & history settings
    MIN_DELTA    = 10.0
    best_val_loss = float('inf')
    best_comb     = float('inf')
    best_ep_comb  = 0
    wait          = 0
    PATIENCE      = 30
    EPOCHS        = 500
    history       = {'train': [], 'val': [], 'comb': []}
    eps_rel       = 0.266

    for ep in range(1, EPOCHS+1):
        # always reshuffle train/val if dynamic_split=True
        if dynamic_split:
            tr_ld, va_ld, tr_ds, va_ds = make_loaders(X, Yl, Yr, W)

        # — Train —
        model.train()
        tr_loss = 0.0
        for xb, yb_log, _, wb in tr_ld:
            xb, yb_log, wb = xb.to(device), yb_log.to(device), wb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb_log, wb)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_ds)
        history['train'].append(tr_loss)

        # — Validate —
        model.eval()
        va_loss = 0.0
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb_log, yb_raw, _ in va_ld:
                xb, yb_log = xb.to(device), yb_log.to(device)
                p = model(xb)
                va_loss += criterion(p, yb_log,
                                     torch.ones_like(yb_log)).item() * xb.size(0)
                all_p.append(p.cpu().numpy())
                all_t.append(yb_raw.numpy())
        va_loss /= len(va_ds)
        history['val'].append(va_loss)

        # Invert log-scale & clip negatives only
        preds  = np.expm1(scaler_y.inverse_transform(
                     np.concatenate(all_p).reshape(-1,1)
                  )).flatten()
        preds  = np.clip(preds, 0, None)
        truths = np.concatenate(all_t)

        # Metrics computation
        mae     = mean_absolute_error(truths, preds)
        rmse    = np.sqrt(mean_squared_error(truths, preds))
        r2m     = r2_score(truths, preds)
        frac_err= np.abs(preds - truths) / np.where(truths>0, truths, 1.0)
        corr_rel= frac_err <= eps_rel
        acc_rel = float(corr_rel.mean())         # relative accuracy

        # Combined metric (smoothed over last 5)
        comb_val= mae + 0.1 * rmse
        history['comb'].append(comb_val)
        comb_sm = (np.mean(history['comb'][-5:])
                   if len(history['comb'])>=5 else comb_val)

        current_lr = optimiser.param_groups[0]['lr']
        print(
            f"Epoch {ep}/{EPOCHS} | Tr {tr_loss:.4f} | Va {va_loss:.4f} | "
            f"MAE {mae:.1f} RMSE {rmse:.1f} R² {r2m:.3f} | "
            f"Acc_rel@{eps_rel:.3f} {acc_rel:.3f} | "
            f"Comb_sm {comb_sm:.2f} | LR {current_lr:.5f}"
        )

        # Adjust LR on plateau
        scheduler.step(va_loss)

        # Early stopping on validation loss
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            wait          = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep} – no improvement for {PATIENCE} epochs")
                break

        # Save best model by combined metric
        if comb_sm < best_comb - MIN_DELTA:
            best_comb    = comb_sm
            best_ep_comb = ep
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest combined metric: {best_comb:.2f} at epoch {best_ep_comb}")

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
    # dynamic_split defaults to True: reshuffle per epoch
    main()
