###输入X：[569, 30]
###y[569, 1] 0或1

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time


def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
        return x * (1 - x)


# 自定义实现
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # self.epsilon = 1e-15
        # 输入到隐藏层的权重和偏置
        self.W1 = np.random.rand(input_size, hidden_size) # [0,1]均匀分布的随机数
        self.b1 = np.random.rand(hidden_size)
        # 隐藏层到输出的权重和偏置
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.random.rand(output_size)
        self.losses = []

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate=0.01):
        # 除以m
        m = y.shape[0]
        output_loss = self.a2 - y # 交叉熵损失对输出的梯度
        dW2 = np.dot(self.a1.T, output_loss) / m
        db2 = np.sum(output_loss, axis=0) / m
        hidden_loss = np.dot(output_loss, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, hidden_loss) / m
        db1 = np.sum(hidden_loss, axis=0) / m

        # 更新
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.loss_function(y)
            self.losses.append(loss)
            # if epoch % 100 == 0:  # 每100次迭代输出损失
            #     print(f'Epoch {epoch}, Loss: {loss}')
            self.backward(X, y)

    # def loss_function(self, y):
    #     return np.mean((self.a2 - y) ** 2)
    
    def loss_function(self, y):
        predictions = sigmoid(self.a2)
        
        # 二元交叉熵损失计算
        # 对于y=1的情况，损失为 -log(predictions)
        # 对于y=0的情况，损失为 -log(1-predictions)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def accuracy(self, X, y):
        predictions = self.forward(X)
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes == y)

# 使用库实现
class SimpleNNLib(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNNLib, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.losses = []

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def train(self, X, y, epochs):
        optimizer = optim.Adam(self.parameters(), lr=0.01)  # 获取神经网络模型的所有可训练参数:权重和偏置
        criterion = nn.BCELoss() #交叉熵损失

        for epoch in range(epochs):
            optimizer.zero_grad() #清除参数附属的梯度
            outputs = self.forward(X)
            loss = criterion(outputs, y) # 计算损失
            self.losses.append(loss.item())
            # if epoch % 100 == 0:  # 每100次迭代输出损失
            #     print(f'Epoch {epoch}, Loss: {loss.item()}')
            loss.backward() # 得到梯度
            optimizer.step() # 参数更新

    def accuracy(self, X, y):
        with torch.no_grad(): # 计算准确率时禁止计算梯度，以提高效率和节省内存
            predictions = self(X)
            predicted_classes = (predictions > 0.5).float()
            return (predicted_classes == y).float().mean().item()

class SklearnNN:
    def __init__(self, hidden_size):
        # 优化器默认是adma，损失函数是交叉熵损失
        self.model = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=1000, warm_start=True)
        self.losses = []

    def accuracy(self, X, y):
        return self.model.score(X, y)
    
    # def calculate_mse(self, y_true, y_pred):
    #     return np.mean((y_true - y_pred) ** 2)

    def calculate_bce(self, y_true, y_pred):
        epsilon = 1e-15  # 防止对数计算中的数值问题
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # 交叉熵损失
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.model.fit(X, y.ravel())
            predictions = self.model.predict(X)
            loss = self.calculate_bce(y, predictions.reshape(-1, 1))
            self.losses.append(loss)
            # if epoch % 100 == 0:  # 每100次迭代输出均方损失
            #     print(f'Epoch {epoch}, Loss: {loss}')  # 输出均方损失



# 加载数据集
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# 训练测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

# 训练和比较不同神经元个数下的模型
neurons = [4, 8, 16]  # 可调神经元个数
epochs = 1000
loss_curves_custom = {}
loss_curves_lib = {}
loss_curves_sklearn = {}
all_time = []

for hidden_size in neurons:
    # 自定义实现
    start_time = time.time()
    custom_nn = SimpleNN(input_size=X.shape[1], hidden_size=hidden_size, output_size=1)
    custom_nn.train(X_train, y_train, epochs)
    end_time1 = time.time() - start_time
    loss_curves_custom[hidden_size] = custom_nn.losses
    print(f'Custom NN with {hidden_size} neurons Accuracy: {custom_nn.accuracy(X_test, y_test)}')

    # torch实现
    start_time = time.time()
    lib_nn = SimpleNNLib(input_size=X.shape[1], hidden_size=hidden_size, output_size=1)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    lib_nn.train(X_train_tensor, y_train_tensor, epochs)
    end_time2 = time.time() - start_time
    loss_curves_lib[hidden_size] = lib_nn.losses
    print(f'torch NN with {hidden_size} neurons Accuracy: {lib_nn.accuracy(torch.FloatTensor(X_test), torch.FloatTensor(y_test))}')

    # sklearn实现
    start_time = time.time()
    sklearn_nn = SklearnNN(hidden_size)
    sklearn_nn.train(X_train, y_train, epochs)
    end_time3 = time.time() - start_time
    loss_curves_sklearn[hidden_size] = sklearn_nn.losses
    print(f'Sklearn NN with {hidden_size} neurons Accuracy: {sklearn_nn.accuracy(X_test, y_test)}\n')

    all_time.append([end_time1, end_time2, end_time3])
    
#打印v...
print('Loss\n\t\tcustom\ttorch\tsklearn')
for x in neurons:
    print(f"neurons={x}")
    for i in range(0, 1001, 199):
        print("\t\t{:<.3f}\t{:<.3f}\t{:<.3f}".format(
            loss_curves_custom.get(x, [0])[i], 
            loss_curves_lib.get(x, [0])[i], 
            loss_curves_sklearn.get(x, [0])[i]
        ))

#打印
print('Time\n\t\tcustom\ttorch\tsklearn')
for i in range(len(neurons)):
    print(f"neurons={neurons[i]}")
    print("\t\t{:<.3f}\t{:<.3f}\t{:<.3f}".format(
        all_time[i][0], 
        all_time[i][1], 
        all_time[i][2]
    ))

# 绘制损失曲线
plt.figure(figsize=(12, 8))
for hidden_size in neurons:
    plt.plot(loss_curves_custom[hidden_size], label=f'Custom NN - {hidden_size} neurons')
    plt.plot(loss_curves_lib[hidden_size], label=f'torch NN - {hidden_size} neurons', linestyle='dashed')
    plt.plot(loss_curves_sklearn[hidden_size], label=f'Sklearn NN - {hidden_size} neurons', linestyle='dotted')

plt.title('Loss Curves Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
