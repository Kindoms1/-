import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 生成N个随机点
N=30
x = np.random.uniform(0, 2 * np.pi, N)
y = np.sin(x)

# 添加随机偏差
bias = np.random.uniform(-0.1, 0.1, N)
y = y + bias

# 将数据点保存到CSV文件
data = np.column_stack((x, y))
np.savetxt("sine_points_with_bias.csv", data, delimiter=',', fmt='%.6f', header='x,y', comments='')

# 绘制
plt.scatter(x, y, color='blue')
plt.title(f'{N} Random Points of the Sine Function with Bias')
plt.xlabel('x')
plt.ylabel('sin(x) with Bias')
plt.grid(True)
plt.show()



# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 生成N个x值
N = 30
x_values = np.random.uniform(-10, 10, N)

# 计算对应的y值
y_values = 2 * x_values + 5

# 为y值添加微小浮动
noise = np.random.normal(0, 0.5, N)  # 浮动的标准差为0.5
y_values_noisy = y_values + noise

# 创建DataFrame
data = pd.DataFrame({
    'x': x_values,
    'y': y_values_noisy
})

# 保存到CSV文件
data.to_csv('line.csv', index=False)

# 绘制散点图
plt.scatter(x_values, y_values_noisy, color='blue', label='Noisy Data')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Scatter Plot of Generated Data with Fitted Line')
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.show()