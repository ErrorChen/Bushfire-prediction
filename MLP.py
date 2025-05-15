# MLP.py —— 已修改版
import os
import pandas as pd                           # 用于数据加载 :contentReference[oaicite:0]{index=0}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 标准化 :contentReference[oaicite:1]{index=1}
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib  # 用于模型持久化

def load_data(dataset_name):
    """
    从仓库中的datasets目录加载CSV文件
    参数:
        dataset_name (str): CSV文件名（含扩展名），例如 'AUSWeatherData.csv'
    返回:
        X (DataFrame): 特征矩阵
        y (Series): 目标标签
    """
    base_dir = os.path.join(os.path.dirname(__file__), 'datasets')  # 仓库根目录下 datasets :contentReference[oaicite:2]{index=2}
    path = os.path.join(base_dir, dataset_name)
    df = pd.read_csv(path)  # 加载数据
    # 假设最后一列为标签，其余列为特征
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def main():
    # 1. 读取数据
    X, y = load_data('YourDataset.csv')  # TODO: 替换为实际数据集文件名
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 构建 MLP 模型
    # 输入维度自动对应特征数量
    input_dim = X_train.shape[1]
    mlp = MLPClassifier(
        hidden_layer_sizes=(input_dim*2, input_dim),  # 示例：两层隐藏层，节点数为2*输入和1*输入
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=1
    )
    
    # 5. 模型训练
    mlp.fit(X_train_scaled, y_train)
    
    # 6. 性能评估
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {acc:.4f}')
    
    # 7. 保存模型与预处理器
    joblib.dump({'model': mlp, 'scaler': scaler}, 'mlp_model.joblib')
    print('Model and scaler saved to mlp_model.joblib')

if __name__ == '__main__':
    main()
