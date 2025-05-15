# MLP.py
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, classification_report

def train_rainfall_model():
    # —— 1. 年度降雨 & Bushfire Index 回归 —— 
    df = pd.read_csv('datasets/rainfall.csv', header=0)
    # 重命名列
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul']
    rename_map = {'Unnamed: 0':'Period',
                  'Dennison Bushfire Indices':'BushfireIndex',
                  'Unnamed: 17':'Ha_Burnt'}
    for i, m in enumerate(months, start=1):
        rename_map[f'Unnamed: {i}'] = m
    df = df.rename(columns=rename_map)

    # 转为数值型并丢弃含缺失值的行
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
    # —— 2. 地块属性分类 —— 
    df = pd.read_csv('datasets/fire_for16-21_attributes.csv', header=0)

    # 基本数值特征
    X = df[['COUNT','VALUE','FOREST']].copy()

    # 类别特征 one-hot
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
    # —— 3. MODIS 热点高置信度分类 —— 
    files = glob.glob('datasets/modis_*.csv')
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # 时间解析
    df['acq_datetime'] = pd.to_datetime(
        df['acq_date'] + ' ' +
        df['acq_time'].astype(str).str.zfill(4),
        format='%Y-%m-%d %H%M'
    )
    df['month']     = df['acq_datetime'].dt.month
    df['dayofyear'] = df['acq_datetime'].dt.dayofyear
    df['hour']      = df['acq_datetime'].dt.hour

    # 编码
    df['is_day']   = (df['daynight']=='D').astype(int)
    df['sat_code'] = df['satellite'].astype('category').cat.codes

    features = [
        'brightness','bright_t31','frp','confidence',
        'scan','track','month','dayofyear','hour',
        'is_day','sat_code'
    ]
    X = df[features]

    # 目标：置信度 > 80% 视作“高置信火点”
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
