# MLP.py
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, classification_report

def train_rainfall_model():
    # -- 1. Annual rainfall & Bushfire Index regression --
    df = pd.read_csv('datasets/rainfall.csv', header=0)
    # Rename columns
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul']
    rename_map = {'Unnamed: 0':'Period',
                  'Dennison Bushfire Indices':'BushfireIndex',
                  'Unnamed: 17':'Ha_Burnt'}
    for i, m in enumerate(months, start=1):
        rename_map[f'Unnamed: {i}'] = m
    df = df.rename(columns=rename_map)

    # Convert to numeric and drop rows with missing values
    to_num = months + ['BushfireIndex', 'Ha_Burnt']
    for col in to_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=to_num)

    X = df[months + ['BushfireIndex']]
    y = df['Ha_Burnt']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(100,50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print('== Rainfall → Burnt Area Regression ==')
    print(f' MSE: {mean_squared_error(y_test,y_pred):.2f}')
    print(f'  R²: {r2_score(y_test,y_pred):.3f}\n')

def train_fire_attribute_model():
    # -- 2. Plot attribute classification --
    df = pd.read_csv('datasets/fire_for16-21_attributes.csv', header=0)

    # Basic numeric features
    X = df[['COUNT','VALUE','FOREST']].copy()

    # Categorical features one-hot encoding
    for cat in ['FOR_CATEGO','FOR_TEN','STATE']:
        X = pd.concat([X, pd.get_dummies(df[cat], prefix=cat)], axis=1)

    y = df['FOR_BURN_T'].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(100,50),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print('== Fire Attribute → Burn Type Classification ==')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

def train_modis_model():
    # -- 3. MODIS high-confidence hotspot classification --
    files = glob.glob('datasets/modis_*.csv')
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # Parse datetime
    df['acq_datetime'] = pd.to_datetime(
        df['acq_date'] + ' ' +
        df['acq_time'].astype(str).str.zfill(4),
        format='%Y-%m-%d %H%M'
    )
    df['month']     = df['acq_datetime'].dt.month
    df['dayofyear'] = df['acq_datetime'].dt.dayofyear
    df['hour']      = df['acq_datetime'].dt.hour

    # Encode features
    df['is_day']   = (df['daynight']=='D').astype(int)
    df['sat_code'] = df['satellite'].astype('category').cat.codes

    features = [
        'brightness','bright_t31','frp','confidence',
        'scan','track','month','dayofyear','hour',
        'is_day','sat_code'
    ]
    X = df[features]

    # Target: confidence > 80% as high-confidence hotspot
    y = (df['confidence'] > 80).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(100,50),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print('== MODIS Hotspot High-Confidence Classification ==')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_rainfall_model()
    train_fire_attribute_model()
    train_modis_model()
