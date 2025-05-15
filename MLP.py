# MLP.py
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


def train_rainfall_model():
    print("== Rainfall → Burnt Area Regression ==")
    df = pd.read_csv('datasets/rainfall.csv', header=0)
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul']
    rename = {
        'Unnamed: 0': 'Period',
        'Dennison Bushfire Indices': 'BushfireIndex',
        'Unnamed: 17': 'Ha_Burnt'
    }
    for i, m in enumerate(months, start=1):
        rename[f'Unnamed: {i}'] = m
    df = df.rename(columns=rename)

    df[months + ['BushfireIndex', 'Ha_Burnt']] = df[months + ['BushfireIndex', 'Ha_Burnt']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=months + ['BushfireIndex', 'Ha_Burnt'])

    X = df[months + ['BushfireIndex']]
    y = df['Ha_Burnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f" MSE: {mse:.2f}")
    print(f" R²: {r2:.3f}\n")
    return {'task': 'rainfall_regression', 'MSE': mse, 'R2': r2}


def train_fire_attribute_model():
    print("== Fire Attribute → Burn Type Classification ==")
    df = pd.read_csv('datasets/fire_for16-21_attributes.csv', header=0)
    numeric_feats = ['COUNT', 'VALUE', 'FOREST']
    cat_feats = ['FOR_CATEGO', 'FOR_TEN', 'STATE']

    X = df[numeric_feats].copy()
    for cat in cat_feats:
        X = pd.concat([X, pd.get_dummies(df[cat], prefix=cat)], axis=1)
    y = df['FOR_BURN_T'].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return {'task': 'fire_attribute_classification', 'accuracy': acc, 'report': report_dict}


def train_modis_model():
    print("== MODIS Hotspot High-Confidence Classification ==")
    files = glob.glob('datasets/modis_*.csv')
    if not files:
        raise FileNotFoundError("No MODIS CSV files found in datasets/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df['acq_datetime'] = pd.to_datetime(
        df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
        format='%Y-%m-%d %H%M'
    )
    df['month'] = df['acq_datetime'].dt.month
    df['dayofyear'] = df['acq_datetime'].dt.dayofyear
    df['hour'] = df['acq_datetime'].dt.hour
    df['is_day'] = (df['daynight'] == 'D').astype(int)
    df['sat_code'] = df['satellite'].astype('category').cat.codes

    features = [
        'brightness', 'bright_t31', 'frp', 'scan', 'track',
        'month', 'dayofyear', 'hour', 'is_day', 'sat_code'
    ]
    X = df[features]
    y = (df['confidence'] > 80).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return {'task': 'modis_classification', 'accuracy': acc, 'report': report_dict}


def generate_bushfire_report():
    results = [
        train_rainfall_model(),
        train_fire_attribute_model(),
        train_modis_model()
    ]
    df = pd.json_normalize(results, sep='_')
    df = df.rename(columns={
        'task': 'Task',
        'MSE': 'Regression_MSE',
        'R2': 'Regression_R2',
        'accuracy': 'Classification_Accuracy'
    })
    df.columns = df.columns.str.replace('report_', '', regex=False)
    df.columns = df.columns.str.replace('macro avg_', 'macro_', regex=False)
    df.columns = df.columns.str.replace('weighted avg_', 'weighted_', regex=False)
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    keep = [
        'Task', 'Regression_MSE', 'Regression_R2',
        'Classification_Accuracy',
        'macro_precision', 'macro_recall', 'macro_f1-score',
        'weighted_precision', 'weighted_recall', 'weighted_f1-score'
    ]
    cols = [c for c in keep if c in df.columns]
    print("\n== Bushfire Prediction Summary ==")
    print(df[cols].to_string(index=False))

if __name__ == '__main__':
    train_rainfall_model()
    train_fire_attribute_model()
    train_modis_model()
    generate_bushfire_report()
