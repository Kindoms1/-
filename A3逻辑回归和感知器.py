import warnings
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA


# 忽略RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def logistic_regression_breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X) # 会计算标准差和方差

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear') #liblinear适合小规模数据集，默认l2正则化

    scores = cross_val_score(model, X_train, y_train, cv=5)  # 5折交叉验证
    print(f"logistic调用: {scores}")
    print(f"平均每折准确率: {np.mean(scores):.2f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) # np.mean()将bool数组转化成01, (1+1+1+0+1)/5 = 0.8
    print(f"在测试集上准确率: {accuracy:.2f}\n")

logistic_regression_breast_cancer()

def A_logistic_regression_breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.c_[np.ones(shape=X.shape[0]), X]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    theta = np.random.uniform(-1, 1, X_train.shape[1])
    alpha = 0.01
    iterations = 100

    def sigmoid(x):
        return 1 / (1 + np.exp(-x)) 

    # 梯度下降
    def gradient_descent(X, y, theta, alpha, iterations):
        m = len(y)
        for _ in range(iterations):
            h = sigmoid(X.dot(theta))
            gradient = X.T.dot(h - y) / m
            theta -= alpha * gradient
        return theta

    def predict(X, theta):
        probabilities = sigmoid(X.dot(theta))
        return (probabilities >= 0.5).astype(int) #概率大于0.5为正类
    
    scores = []
    for train_index, test_index in sklearn.model_selection.KFold(n_splits=5).split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        theta = np.random.uniform(-1, 1, x_train.shape[1])
        # theta = np.zeros(x_train.shape[1]) # 初始化为1和0差很大
        theta = gradient_descent(x_train, Y_train, theta, alpha, iterations)
        scores.append(np.mean(predict(x_test, theta) == Y_test))
    print("logistic手动:{}".format(scores))
    print(f"平均每折准确率: {np.mean(scores):.2f}")

    theta = gradient_descent(X_train, y_train, theta, alpha, iterations)
    y_pred = predict(X_test, theta)
    accuracy = np.mean(y_pred == y_test)
    print(f"在测试集上准确率: {accuracy:.2f}\n")

A_logistic_regression_breast_cancer()

def softmax_iris():
    data = load_iris()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='lbfgs', max_iter=200) #lbfgs多分类

    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("softmax调用:{}".format(scores))
    print(f"平均每折准确率: {np.mean(scores):.2f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"在测试集上准确率: {accuracy:.2f}\n")

softmax_iris()

def A_softmax_iris():
    data = load_iris()
    X = data.data
    y = data.target # y=1,2,3

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 添加偏置项
    X = np.c_[np.ones(X.shape[0]), X]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将标签转换为one-hot编码
    K = len(np.unique(y_train))
    y_train_one_hot = np.eye(K)[y_train] # y = [1,0,0] || [0,1,0] || [0,0,1]与梯度下降中的预测对其

    theta = np.random.uniform(-1, 1, (X_train.shape[1], K))
    alpha = 0.01
    iterations = 100

    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def gradient_descent(X, y, theta, alpha, iterations):
        m = len(y)
        for _ in range(iterations):
            h = softmax(X.dot(theta))
            gradient = X.T.dot(h - y) / m
            theta -= alpha * gradient
        return theta

    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        Y_train, Y_test = y_train_one_hot[train_index], y_train_one_hot[test_index]
        theta = np.zeros((X_train.shape[1], K))  # 重新初始化theta
        theta = gradient_descent(x_train, Y_train, theta, alpha, iterations)
        predictions = np.argmax(softmax(x_test.dot(theta)), axis=1)
        scores.append(np.mean(predictions == y_train[test_index]))

    print("softmax手动:{}".format(scores))
    print(f"平均每折准确率: {np.mean(scores):.2f}")

    theta = gradient_descent(X_train, y_train_one_hot, theta, alpha, iterations)

    def predict(X, theta):
        probabilities = softmax(X.dot(theta))
        return np.argmax(probabilities, axis=1)

    y_pred = predict(X_test, theta)
    accuracy = np.mean(y_pred == y_test)
    print(f"在测试集上准确率: {accuracy:.2f}\n")

A_softmax_iris()

# 感知器
def P():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    p = MLPClassifier(hidden_layer_sizes=(), activation='relu', solver='adam', max_iter=1000, random_state=42) # 输入层直连输出层

    scores = cross_val_score(p, X_train, y_train, cv=5)
    print("单层感知器调用:{}".format(scores))
    print(f"平均每折准确率: {np.mean(scores):.2f}")

    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print(f"在测试集上准确率: {accuracy:.2f}\n")

P()

def A_P():
    data = load_breast_cancer()
    X, y = data.data, data.target
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1

    # 先分训练和测试，再分别标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = np.array(X_train, dtype=np.float64)
    X_test = scaler.transform(X_test)
    X_test = np.array(X_test, dtype=np.float64)

    class Perceptron:
        def __init__(self, input_dim):
            self.weights = np.random.randn(input_dim)
            self.bias = 0

        def predict(self, X):
            linear_output = np.dot(X, self.weights) + self.bias
            return np.where(linear_output > 0, 1, -1)
        
        def train(self, X, y, epochs):
            for epoch in range(epochs):
                # 对训练样本随机排序
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                
                for i in indices:
                    x_i = X[i]
                    y_i = y[i]
                    prediction = np.dot(x_i, self.weights) + self.bias
                    if y_i * prediction <= 0: # 误分类时更新
                        self.weights += y_i * x_i
                        self.bias += y_i

                # if epoch % 100 == 0:
                #     loss = np.mean((y - self.predict(X)) ** 2)
                #     print(f'Epoch {epoch}, Loss: {loss:.4f}')

        def evaluate(self, X, y):
            predictions = self.predict(X).flatten()
            accuracy = np.mean(predictions == y)
            return accuracy

    scores = []
    for train_index, test_index in sklearn.model_selection.KFold(n_splits=5).split(X_train):
        input_dim = X_train[train_index].shape[1]
        perceptron = Perceptron(input_dim=input_dim)
        perceptron.train(X_train[train_index], y_train[train_index], epochs=1000)
        accuracy = perceptron.evaluate(X_train[test_index], y_train[test_index])
        scores.append(accuracy)
    print("单层感知器手动:{}".format(scores))
    print(f"平均每折准确率: {np.mean(scores):.2f}")

    input_dim = X_train.shape[1]
    perceptron = Perceptron(input_dim=input_dim)
    perceptron.train(X_train, y_train, epochs=200)
    accuracy = perceptron.evaluate(X_test, y_test)
    print(f'在测试集上准确率: {accuracy:.4f}')

A_P()

from A3逻辑回归和感知器PCA画图 import *
plot_logistic()
plt.show()
plot_softmax()
plt.show()
plot_P()
plt.show()