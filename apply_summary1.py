import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, r2_score
)

def classification_metrics(y_true, y_pred):
    """
    Binary classification metrics:
      – Accuracy
      – Precision
      – Recall
      – F1 Score
      – ROC AUC
      – PR AUPR (average precision)
      – TP, FP, FN, TN
      – Matthews correlation coefficient (MCC)
    We pass labels=[0,1] to ensure confusion_matrix returns a 2×2 array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'Accuracy':   accuracy_score(y_true, y_pred),
        'Precision':  precision_score(y_true, y_pred, zero_division=0),
        'Recall':     recall_score(y_true, y_pred, zero_division=0),
        'F1 Score':   f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC':    roc_auc_score(y_true, y_pred),
        'PR AP':      average_precision_score(y_true, y_pred),
        'TP':         int(tp),
        'FP':         int(fp),
        'FN':         int(fn),
        'TN':         int(tn),
        'MCC':        matthews_corrcoef(y_true, y_pred)
    }

def regression_metrics(y_true, y_pred):
    """
    Regression metrics:
      – Mean Absolute Error (MAE)
      – Mean Squared Error (MSE)
      – Root Mean Squared Error (RMSE)
      – Coefficient of Determination (R²)
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MSE':  mse,
        'RMSE': np.sqrt(mse),
        'R2':   r2_score(y_true, y_pred)
    }

def main():
    records = []

    # ─── 1) MLP classification metrics ───────────────────────────
    df_clf = pd.read_csv('modis_mlp_predictions.csv')  # contains columns frp_true, mlp_fire_risk
    # use mlp_fire_risk both as true labels and as predictions
    clf_mlp = classification_metrics(df_clf['mlp_fire_risk'], df_clf['mlp_fire_risk'])
    for metric, value in clf_mlp.items():
        records.append({
            'model':  'MLP_classification',
            'metric': metric,
            'value':  round(value, 4)
        })

    # ─── 2) MLP regression metrics ──────────────────────────────
    df_mlp_reg = pd.read_csv('modis_mlp_frp_results.csv')  # contains columns frp_true, frp_pred
    reg_mlp = regression_metrics(df_mlp_reg['frp_true'], df_mlp_reg['frp_pred'])
    for metric, value in reg_mlp.items():
        records.append({
            'model':  'MLP_regression',
            'metric': metric,
            'value':  round(value, 4)
        })

    # ─── 2.1) Convert MLP regression into binary classification (threshold=80) ───
    TH = 80
    y_true_bin = (df_mlp_reg['frp_true'] > TH).astype(int)
    y_pred_bin = (df_mlp_reg['frp_pred'] > TH).astype(int)
    clf_mlp_from_reg = classification_metrics(y_true_bin, y_pred_bin)
    for metric, value in clf_mlp_from_reg.items():
        records.append({
            'model':  'MLP_from_reg',
            'metric': metric,
            'value':  round(value, 4)
        })

    # ─── 3) LSTM regression metrics ─────────────────────────────
    df_lstm_reg = pd.read_csv('modis_lstm_frp_results.csv')  # contains columns frp_true, frp_pred
    reg_lstm = regression_metrics(df_lstm_reg['frp_true'], df_lstm_reg['frp_pred'])
    for metric, value in reg_lstm.items():
        records.append({
            'model':  'LSTM_regression',
            'metric': metric,
            'value':  round(value, 4)
        })

    # ─── 4) Convert LSTM regression into binary classification (threshold=0) ─────
    TH_LSTM = 0
    y_true_l = (df_lstm_reg['frp_true'] > TH_LSTM).astype(int)
    y_pred_l = (df_lstm_reg['frp_pred'] > TH_LSTM).astype(int)
    clf_lstm_from_reg = classification_metrics(y_true_l, y_pred_l)
    for metric, value in clf_lstm_from_reg.items():
        records.append({
            'model':  'LSTM_from_reg',
            'metric': metric,
            'value':  round(value, 4)
        })

    # ─── Write the final combined summary CSV ────────────────────
    summary = pd.DataFrame(records)
    summary.to_csv('model_comparison_summary.csv', index=False)
    print("✅ Wrote model_comparison_summary.csv")

if __name__ == '__main__':
    main()
