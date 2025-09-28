import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv("sine_points_with_bias.csv") # sin曲线

x = df['x'].values
y = df['y'].values

degrees = [1, 3, 4, 5, 30]
colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.figure(figsize=(10, 6))

for degree, color in zip(degrees, colors):
    # numpy的polyfit进行多项式拟合
    '''
    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error in the order `deg`, `deg-1`, ... `0`.
    '''
    coefficients = np.polyfit(x, y, degree)
    # 打印系数
    print(f"项数 {degree} 系数: {coefficients}")
    # 创建一个多项式函数
    p = np.poly1d(coefficients)
    # 生成用于绘图的x值
    x_plot = np.linspace(min(x), max(x), 400)
    # 计算多项式在这些x值上的y值
    y_plot = p(x_plot)
    # 绘制拟合的多项式曲线
    plt.plot(x_plot, y_plot, label=f'Degree {degree}', color=color)

# 绘制原始数据点
plt.scatter(x, y, label='Original Data', color='black')

plt.ylim(-2, 2)
plt.title('Polynomial Fits of Different Degrees')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


# 加载数据
file_path = './line.csv' #直线：y=2x+5
data = pd.read_csv(file_path)

x_data = data['x'].values
y_data = data['y'].values

# 添加偏置项
X = np.vstack([np.ones(len(x_data)), x_data]).T

# 1) 解析解求解
def analytical_solution(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# 2) 岭回归求解
def ridge_solution(X, y, lambda_):
    n = X.shape[1]
    I = np.eye(n)
    return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y

# 解析解
w_analytical = analytical_solution(X, y_data)
print("解析解(截距，斜率):", w_analytical)

# 岭回归
lambda_ = 1
w_ridge = ridge_solution(X, y_data, lambda_)
print(f"岭回归(lambda={lambda_})(截距，斜率):", w_ridge)



# 生成拟合曲线
x_range = np.linspace(min(x_data), max(x_data), 100)
X_range = np.vstack([np.ones(len(x_range)), x_range]).T
y_analytical_pred = X_range @ w_analytical
y_ridge_pred = X_range @ w_ridge

# 绘制解析解图像
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_range, y_analytical_pred, label='Analytical Solution', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Analytical Solution')
plt.show()

# 绘制岭回归图像
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_range, y_ridge_pred, label=f'Ridge Regression (lambda={lambda_})', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Ridge Regression (lambda={lambda_})')
plt.show()

# # 两张图在一张画布上
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # 解析解图像
# ax[0].scatter(x_data, y_data, color='red', label='Data Points')
# ax[0].plot(x_range, y_analytical_pred, label='Analytical Solution', color='blue')
# ax[0].set_title('Analytical Solution')
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# ax[0].legend()

# # 岭回归图像
# ax[1].scatter(x_data, y_data, color='red', label='Data Points')
# ax[1].plot(x_range, y_ridge_pred, label=f'Ridge Regression (lambda={lambda_})', color='green')
# ax[1].set_title(f'Ridge Regression (lambda={lambda_})')
# ax[1].set_xlabel('x')
# ax[1].set_ylabel('y')
# ax[1].legend()

# # 调整布局以防止重叠
# plt.tight_layout()
# plt.show()
