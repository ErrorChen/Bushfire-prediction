import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from MLP import DeepFRPNet   # your MLP definition

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_frp_model.pt')
OUT_CSV    = os.path.join(BASE_DIR, 'modis_mlp_frp_results.csv')

# 1) load & concat all MODIS files
files = glob.glob(os.path.join(BASE_DIR, 'datasets', 'modis_*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.rename(columns={'frp': 'frp_true'})  # rename for clarity

# 2) rebuild exactly the same features you used in training...
#    numeric features
num_cols = ['latitude','longitude','brightness','scan','track',
            'confidence','version','bright_t31']
df['acq_date']     = pd.to_datetime(df['acq_date'])
df['month']        = df['acq_date'].dt.month
df['day_of_year']  = df['acq_date'].dt.dayofyear
df['acq_time']     = df['acq_time'].astype(int)
df['hour']         = df['acq_time'] // 100
df['minute']       = df['acq_time'] % 100
df['time_minutes'] = df['hour']*60 + df['minute']
num_cols += ['month','day_of_year','time_minutes']

#    categorical features → one‐hot
cat_cols = ['satellite','instrument','daynight','type']
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_mat = enc.fit_transform(df[cat_cols])

# 3) assemble feature matrix and standardize numeric part
X_num = df[num_cols].to_numpy(dtype=np.float32)
scaler = StandardScaler().fit(X_num)
X_num = scaler.transform(X_num)
X     = np.hstack([X_num, cat_mat.astype(np.float32)])

# 4) batch it and predict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds     = TensorDataset(torch.from_numpy(X))
loader = DataLoader(ds, batch_size=512, shuffle=False)

model = DeepFRPNet(X.shape[1]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for (xb,) in loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().ravel()
        preds.append(out)
preds = np.concatenate(preds, axis=0)

# 5) attach predictions & compute error columns
df['frp_pred']  = preds
df['abs_error'] = (df['frp_pred'] - df['frp_true']).abs()
df['sq_error']  = (df['frp_pred'] - df['frp_true'])**2
df['pct_error'] = (df['frp_pred'] - df['frp_true']) / df['frp_true'] * 100

# 6) compute global metrics
mae  = mean_absolute_error(df['frp_true'], df['frp_pred'])
mse  = mean_squared_error(df['frp_true'], df['frp_pred'])
rmse = np.sqrt(mse)
r2   = r2_score(df['frp_true'], df['frp_pred'])

print("=== Global regression metrics ===")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 7) tack them into the DataFrame as constant columns
df['MAE']  = mae
df['MSE']  = mse
df['RMSE'] = rmse
df['R2']   = r2

# 8) write out
out_cols = [
    'frp_true','frp_pred',
    'abs_error','sq_error','pct_error',
    'MAE','MSE','RMSE','R2'
]
df[out_cols].to_csv(OUT_CSV, index=False)
print(f"✅ Wrote per-sample & global metrics to {OUT_CSV}")
