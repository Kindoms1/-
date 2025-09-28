import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 定义一个函数来绘制分界线和样本点
def plot_decision_boundary(model, X, y, title):
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 预测网格中的每个点的标签
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制等高线图和训练样本
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)

# 加载乳腺癌数据集
def plot_logistic():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用PCA降至二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 训练逻辑回归模型
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='lbfgs', max_iter=200)
    model.fit(X_train, y_train)

    # 绘制训练集的分界线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, X_train, y_train, 'logistic - Train Set (PCA 2D)')

    # 绘制测试集的分界线
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model, X_test, y_test, 'logistic - Test Set (PCA 2D)')

# 加载鸢尾花数据集
def plot_softmax():
    data = load_iris()
    X = data.data
    y = data.target

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用PCA降至二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 训练逻辑回归模型
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='lbfgs', max_iter=200)
    model.fit(X_train, y_train)

    # 绘制训练集的分界线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, X_train, y_train, 'Iris - Train Set (PCA 2D)')

    # 绘制测试集的分界线
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model, X_test, y_test, 'Iris - Test Set (PCA 2D)')

# 加载乳腺癌数据集_P
def plot_P():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用PCA降至二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 训练逻辑回归模型
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(), activation='relu', solver='adam', max_iter=1000, random_state=42)

    mlp.fit(X_train, y_train)

    # 绘制训练集的分界线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(mlp, X_train, y_train, 'MLP - Train Set (PCA 2D)')

    # 绘制测试集的分界线
    plt.subplot(1, 2, 2)
    plot_decision_boundary(mlp, X_test, y_test, 'MLP - Test Set (PCA 2D)')

if __name__=="__main__":
    plot_logistic()
    plt.show()
    plot_softmax()
    plt.show()
    plot_P()
    plt.show()