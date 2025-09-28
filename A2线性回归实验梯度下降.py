import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path = 'sine_points_with_bias.csv'
data = pd.read_csv(file_path)

# 提取x和y数据
x_data = data['x'].values
y_data = data['y'].values

# 定义多项式函数，返回x的degree阶多项式的值：y = w0+w1x+w2x^2+...+wmx^m
def poly_features(x, degree):
    return np.array([x**i for i in range(degree + 1)]).T

# 定义梯度下降算法
def gradient_descent(X, y, learning_rate=0.001, epochs=10000):
    m, n = X.shape
    w = np.ones(n)
    for epoch in range(epochs):
        y_pred = np.dot(X, w)
        error = y_pred - y
        gradient = (2/m) * np.dot(X.T, error)
        w -= learning_rate * gradient
    return w

# 定义要使用的M值
M_values = [1, 3, 5, 9]
learning_rate = [0.001, 0.0001, 0.000001, 0.00000000000001]  #不同项数学习率不同
epochs = [1000, 10000, 10000, 10000]

# 设置图形
plt.figure(figsize=(10, 8))
x_range = np.linspace(min(x_data), max(x_data), 100)

# 对于每一个M值，进行拟合并绘制图像
for i in range(len(M_values)):
    M = M_values[i]
    A = learning_rate[i]
    B = epochs[i]
    # 获取M阶的特征矩阵
    X_poly = poly_features(x_data, M)
    
    # 使用梯度下降进行训练
    w = gradient_descent(X_poly, y_data, learning_rate=A, epochs=B)
    
    # 计算拟合值
    X_range_poly = poly_features(x_range, M)
    y_range_pred = np.dot(X_range_poly, w)
    
    # 绘制原始点和拟合曲线
    plt.subplot(2, 2, i+1)
    plt.scatter(x_data, y_data, color='red', label='Data Points')
    plt.plot(x_range, y_range_pred, label=f'Polynomial Degree {M}')
    plt.title(f'Polynomial Degree {M}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

# 显示图像
plt.tight_layout()
plt.show()
