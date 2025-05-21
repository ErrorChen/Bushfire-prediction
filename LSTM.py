import glob
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ─── FireDataset: wraps our sliding‐window arrays in a PyTorch Dataset ─────
class FireDataset(Dataset):
    def __init__(self, X, Y_log, Y_raw, W):
        """
        X     : np.ndarray of shape (N, T, F)  – input feature windows
        Y_log : np.ndarray of shape (N,)      – log-transformed targets
        Y_raw : np.ndarray of shape (N,)      – raw FRP targets
        W     : np.ndarray of shape (N,)      – per-sample weights
        """
        self.X     = torch.from_numpy(X.astype(np.float32))
        self.Y_log = torch.from_numpy(Y_log.astype(np.float32))
        self.Y_raw = torch.from_numpy(Y_raw.astype(np.float32))
        self.W     = torch.from_numpy(W.astype(np.float32))

    def __len__(self):
        # Return number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Return the tuple (features, log-target, raw-target, weight)
        return self.X[idx], self.Y_log[idx], self.Y_raw[idx], self.W[idx]


# ─── MultiHeadSelfAttention: standard scaled dot‐product self‐attention ────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        """
        dim   : int – embedding dimension (must be divisible by heads)
        heads : int – number of attention heads
        """
        super().__init__()
        assert dim % heads == 0, "Embedding dim must divide evenly by number of heads"
        self.heads = heads
        self.dk    = dim // heads

        # Single linear projects into Q, K, V concatenated
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        # Linear to combine multi-head output
        self.unify  = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x : Tensor of shape (batch, time, dim)
        returns: Tensor of same shape after self‐attention
        """
        B, T, D = x.size()

        # Project to queries, keys, values and split heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qs, ks, vs = [
            t.view(B, T, self.heads, self.dk).transpose(1, 2)
            for t in qkv
        ]  # each has shape (batch, heads, time, dk)

        # Scaled dot-product attention scores
        scores = torch.matmul(qs, ks.transpose(-2, -1)) / np.sqrt(self.dk)
        attn   = torch.softmax(scores, dim=-1)

        # Weighted sum of values, then merge heads
        out = torch.matmul(attn, vs)               # (batch, heads, time, dk)
        out = out.transpose(1, 2).contiguous()     # (batch, time, heads, dk)
        out = out.view(B, T, D)                    # (batch, time, dim)

        return self.unify(out)


# ─── ImprovedFireModel: CNN → BiLSTM → Attention → Deep FFN ─────────────
class ImprovedFireModel(nn.Module):
    def __init__(self, feat_dim, hid_dim=128, lstm_layers=3, heads=4, dropout=0.3):
        """
        feat_dim    : int – number of input features per timestep
        hid_dim     : int – hidden size for CNN & LSTM
        lstm_layers : int – number of BiLSTM layers
        heads       : int – number of attention heads
        dropout     : float – dropout probability
        """
        super().__init__()

        # 1D CNN extracts local temporal patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Bidirectional LSTM for sequential dependencies
        self.lstm = nn.LSTM(
            hid_dim, hid_dim, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Multi-head self-attention over LSTM outputs
        self.attn = MultiHeadSelfAttention(hid_dim * 2, heads=heads)

        # Layer normalisation before feed‐forward
        self.norm = nn.LayerNorm(hid_dim * 2)

        # Deep feed‐forward network with extra hidden layer
        self.ffn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)
        )

    def forward(self, x):
        """
        x : Tensor of shape (batch, time, features)
        returns: Tensor of shape (batch,) – predicted log-FRP
        """
        # Apply CNN (requires (batch, channels, time))
        c = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # → (batch, time, 2*hid_dim)

        # Pass through BiLSTM
        lstm_out, _ = self.lstm(c)                       # → (batch, time, 2*hid_dim)

        # Self-attention
        attn_out = self.attn(lstm_out)                   # → (batch, time, 2*hid_dim)

        # Combine last timestep from LSTM and attention, then normalise
        last = lstm_out[:, -1, :] + attn_out[:, -1, :]
        h    = self.norm(last)

        # Final FFN to scalar
        return self.ffn(h).squeeze(-1)


# ─── ExtremeLoss: upweights “extreme” errors above threshold ─────────────
class ExtremeLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(reduction='none'), thr=0.05):
        """
        base_loss : nn.Module – base per-sample loss (no reduction)
        thr       : float     – error threshold to consider “extreme”
        """
        super().__init__()
        self.base = base_loss
        self.thr  = thr

    def forward(self, pred, target, weight):
        """
        pred   : Tensor of predictions
        target : Tensor of true log-targets
        weight : Tensor of per-sample weights
        returns: mean weighted loss with double penalty on extreme errors
        """
        err  = self.base(pred, target)
        mask = (err > self.thr).float()
        return ((1 + mask) * err * weight).mean()


# ─── Utility: build train/validation loaders ───────────────────────────────
def make_loaders(X, Yl, Yr, W, split=0.8, batch_size=32):
    """
    Randomly splits the full dataset into train/val.
    Returns DataLoaders and the underlying Datasets for size queries.
    """
    ds      = FireDataset(X, Yl, Yr, W)
    n_train = int(len(ds) * split)
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    tr_ld   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    va_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    return tr_ld, va_ld, train_ds, val_ds


# ─── Main training & evaluation routine ────────────────────────────────────
def main(dynamic_split=False):
    # Choose GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ─── 1. Load & aggregate raw MODIS CSVs ─────────────────────────────────
    files = sorted(glob.glob('datasets/modis_????_Australia.csv'))
    df    = pd.concat([pd.read_csv(f, parse_dates=['acq_date']) for f in files],
                      ignore_index=True)

    # Convert HHMM to minutes since midnight, extract date
    df['acq_min'] = (df['acq_time'] // 100) * 60 + (df['acq_time'] % 100)
    df['date']    = df['acq_date'].dt.floor('D')

    # Named‐aggregation of all 15 original fields
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
        type_count       = pd.NamedAgg('type',       'count'),
    )

    # Ensure no missing dates, fill zeros
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily    = daily.reindex(full_idx, fill_value=0).rename_axis('date').reset_index()

    # ─── 2. Prepare features & sliding windows ────────────────────────────────
    daily['frp_log'] = np.log1p(daily['frp_sum'])
    feat_cols       = [c for c in daily.columns if c not in ('date', 'frp_sum', 'frp_log')]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_all    = scaler_X.fit_transform(daily[feat_cols]).astype(np.float32)
    y_log    = scaler_y.fit_transform(daily[['frp_log']]).flatten().astype(np.float32)
    y_raw    = daily['frp_sum'].to_numpy(dtype=np.float32)

    # Build sliding windows of length T=30
    T, X, Yl, Yr = 30, [], [], []
    for i in range(len(X_all) - T):
        X.append(X_all[i:i+T])
        Yl.append(y_log[i+T])
        Yr.append(y_raw[i+T])
    X, Yl, Yr = map(np.stack, (X, Yl, Yr))

    # Compute sample weights: upweight top‐10% days by 1.5×
    p90 = np.percentile(y_raw, 90)
    W   = ((y_raw > p90) * 1.5 + 1.0).astype(np.float32)

    # ─── 3. Initialise model, loss, optimiser & scheduler ────────────────────
    model     = ImprovedFireModel(len(feat_cols)).to(device)
    criterion = ExtremeLoss(thr=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = 500 * (len(X) // 32 + 1)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, total_steps=total_steps
    )

    # Initial train/val split
    tr_ld, va_ld, tr_ds, va_ds = make_loaders(X, Yl, Yr, W)

    # ─── 4. Training loop with smoothed early‐stopping ───────────────────────
    history      = {'train': [], 'val': [], 'comb': []}
    best_score   = float('inf')
    best_epoch   = 0
    wait         = 0
    EPOCHS       = 500
    PATIENCE     = 50
    MIN_DELTA    = 10.0
    eps_rel      = 0.266

    for ep in range(1, EPOCHS+1):
        # Optionally reshuffle train/val each epoch
        if dynamic_split:
            tr_ld, va_ld, tr_ds, va_ds = make_loaders(X, Yl, Yr, W)

        # — Training phase —
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
        tr_loss /= len(tr_ds)
        history['train'].append(tr_loss)

        # — Validation phase —
        model.eval()
        va_loss, all_preds, all_truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb_log, yb_raw, _ in va_ld:
                xb, yb_log = xb.to(device), yb_log.to(device)
                pred = model(xb)
                va_loss += criterion(pred, yb_log, torch.ones_like(yb_log)).item() * xb.size(0)
                all_preds.append(pred.cpu().numpy())
                all_truths.append(yb_raw.numpy())

        va_loss /= len(va_ds)
        history['val'].append(va_loss)

        # Concatenate varying‐size batches
        preds_concat = np.concatenate(all_preds)           # shape (N_val,)
        preds_log    = preds_concat.reshape(-1, 1)         # for scaler inverse
        frp_preds    = np.expm1(scaler_y.inverse_transform(preds_log)).flatten()
        # Clip to 99th percentile to curb extreme blowups
        frp_preds    = np.clip(frp_preds, 0, np.percentile(y_raw, 99))
        truths       = np.concatenate(all_truths)

        # Compute regression metrics
        mae   = mean_absolute_error(truths, frp_preds)
        rmse  = np.sqrt(mean_squared_error(truths, frp_preds))
        r2    = r2_score(truths, frp_preds)

        # Combined metric and smoothing over last 5 epochs
        comb_val = mae + 0.1 * rmse
        history['comb'].append(comb_val)
        if len(history['comb']) >= 5:
            comb_sm = np.mean(history['comb'][-5:])
        else:
            comb_sm = comb_val

        # Logging in Aussie style
        print(
            f"Epoch {ep}/{EPOCHS} | Tr {tr_loss:.4f} Va {va_loss:.4f} | "
            f"MAE {mae:.1f} RMSE {rmse:.1f} R² {r2:.3f} | "
            f"Comb_sm {comb_sm:.2f}"
        )

        # Early‐stopping check
        if comb_sm < best_score - MIN_DELTA:
            best_score = comb_sm
            best_epoch = ep
            wait        = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep} – no real improvement in last {PATIENCE} epochs")
                break

    # Final best‐model report
    print(f"\nBest model saved at epoch {best_epoch} with Comb_sm {best_score:.2f}")

    # ─── 5. Plot training & validation loss curves ─────────────────────────
    ep_range = range(1, len(history['train'])+1)
    plt.figure(figsize=(8,4))
    plt.plot(ep_range, history['train'], label='Train Loss')
    plt.plot(ep_range, history['val'],   label='Val   Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend(); plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pass dynamic_split=True to reshuffle train/val each epoch if desired
    main(dynamic_split=False)
