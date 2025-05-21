# MODIS_only_fire_risk.py
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 固定随机种子
SEED = 42
# 缓存文件名
CACHE_FILE = 'modis_fire_risk_cache.json'
# 特征缓存文件
FEAT_CACHE = 'modis_features_cache.pkl'

# 模型加速配置：使用 lbfgs 加速小型网络
CLF_PARAMS = dict(
    hidden_layer_sizes=(50, 20),
    activation='relu',
    solver='lbfgs',
    tol=1e-3,
    max_iter=200,
    random_state=SEED
)


def train_modis_fire_risk():
    print("== MODIS Fire Risk Classification ==")
    # 预处理和特征缓存
    if os.path.exists(FEAT_CACHE):
        X, y = pd.read_pickle(FEAT_CACHE)
    else:
        files = glob.glob('datasets/modis_*.csv')
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        dt = pd.to_datetime(
            df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M'
        )
        df['month'] = dt.dt.month
        df['dayofyear'] = dt.dt.dayofyear
        df['hour'] = dt.dt.hour
        df['is_day'] = (df['daynight'] == 'D').astype(int)
        df['sat_code'] = df['satellite'].astype('category').cat.codes

        features = [
            'brightness', 'bright_t31', 'frp', 'scan', 'track',
            'month', 'dayofyear', 'hour', 'is_day', 'sat_code'
        ]
        X = df[features]
        y = (df['confidence'] > 80).astype(int)
        # 缓存特征以加速后续运行
        pd.to_pickle((X, y), FEAT_CACHE)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Pipeline + MLP
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**CLF_PARAMS))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 评估
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    return {
        'Task': 'modis_fire_risk',
        'Accuracy': acc,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1-score': report['1']['f1-score']
    }


def summary_table():
    # 支持缓存，加速重复调用
    if os.path.exists(CACHE_FILE):
        df = pd.read_json(CACHE_FILE)
        print("\n== Fire Risk Summary (cached) ==")
        print(df.to_string(index=False))
        return

    # 训练并获取结果
    res = train_modis_fire_risk()
    df = pd.DataFrame([res])
    df.to_json(CACHE_FILE, orient='records')

    # 打印 Summary 表格
    print("\n== Fire Risk Summary ==")
    print(df.to_string(index=False))


if __name__ == '__main__':
    summary_table()