#!/usr/bin/env python3
# MODIS_frp_confidence_fire_risk.py
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 固定随机种子
SEED = 42

# 模型加速配置：使用 lbfgs 加速小型网络
CLF_PARAMS = dict(
    hidden_layer_sizes=(50, 20),
    activation='relu',
    solver='lbfgs',
    tol=1e-3,
    max_iter=200,
    random_state=SEED
)


def train_and_evaluate():
    print("== MODIS Fire Risk Classification (frp+confidence) ==")
    # 1. 读取所有 MODIS CSV 文件
    files = glob.glob(os.path.join(BASE_DIR, 'datasets', 'modis_*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 2. 提取特征和标签（仅 frp 和 confidence）
    X = df[['frp', 'confidence']]
    y = (df['confidence'] > 80).astype(int)

    # 3. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 4. 构建 Pipeline 并训练
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**CLF_PARAMS))
    ])
    pipeline.fit(X_train, y_train)

    # 5. 在测试集上预测并评估
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # 6. 返回结果
    return {
        'Task': 'modis_fire_risk',
        'Accuracy': round(acc, 4),
        'Precision': round(report['1']['precision'], 4),
        'Recall': round(report['1']['recall'], 4),
        'F1-score': round(report['1']['f1-score'], 4)
    }


def summary_table():
    # 训练并获取结果
    res = train_and_evaluate()
    df = pd.DataFrame([res])

    # 打印 Summary 表格
    print("\n== Fire Risk Summary ==")
    print(df.to_string(index=False))


if __name__ == '__main__':
    summary_table()