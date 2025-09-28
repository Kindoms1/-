import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def function(x):
    return x**2

def gradient(x):
    return 2 * x

def gradient_descent(learning_rate, n_iterations, initial_x):
    x = initial_x
    x_history = [x] # 保存x
    for _ in range(n_iterations):
        x = x - learning_rate * gradient(x)
        x_history.append(x)
    return x_history

learning_rate = 0.1
n_iterations = 20
initial_x = 10

x_history = gradient_descent(learning_rate, n_iterations, initial_x)

#画图
x_values = np.linspace(-11, 11, 400)
y_values = function(x_values)

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], label='Function: $f(w) = w^2$', color='black')
points, = ax.plot([], [], 'ro', zorder=5, label='w_i')

ax.scatter(0, 0, color='green', label='best', zorder=5)
ax.text(0, 2, 'best', fontsize=12, color='green', horizontalalignment='center')

ax.set_xlim(-11, 11)
ax.set_ylim(0, 121)
ax.set_title(f"Gradient Descent lr={learning_rate}")
ax.set_xlabel('w')
ax.set_ylabel('f(w)')
ax.legend()
ax.grid(True)

text_display = ax.text(0, 110, '', fontsize=12, color='purple')

#动画
def init():
    line.set_data([], [])
    points.set_data([], [])
    text_display.set_text('')
    return line, points, text_display

def update(frame):
    line.set_data(x_values, y_values)
    points.set_data(x_history[:frame+1], [function(x) for x in x_history[:frame+1]])

    for arrow in ax.patches:
        arrow.remove()

    for i in range(frame):
        ax.annotate("", xy=(x_history[i+1], function(x_history[i+1])),
                    xytext=(x_history[i], function(x_history[i])),
                    arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))

    current_x = x_history[frame]
    current_y = function(current_x)
    text_display.set_text(f'(w, f(w)): ({current_x:.2f}, {current_y:.2f})')

    return line, points, text_display

ani = animation.FuncAnimation(fig, update, frames=len(x_history), repeat=False, init_func=init, blit=True, interval=100)
ani.save(f"ex1 lr={learning_rate}.gif")
plt.show()


# 二
def f(x):
    return x[0]**2 + x[1]**2

def gradient2(x):
    return np.array([2 * x[0], 2 * x[1]])

def gradient_descent2(learning_rate, n_iterations, initial_point):
    x = initial_point
    x_history = [x] # 保存x
    for _ in range(n_iterations):
        grad = gradient2(x)
        x = x - learning_rate * grad
        x_history.append(x)
    return x_history

# 初始化参数
initial_point = np.array([1, 3])
learning_rate = 0.1
num_iterations = 20

x_history = gradient_descent2(learning_rate, num_iterations, initial_point)
points = np.array(x_history)

# 准备绘图
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# 添加显示文本
text_display = ax.text2D(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='purple')

# 动画更新函数
def update(frame):
    ax.cla()  # 清除之前的内容
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.scatter(points[:frame+1, 0], points[:frame+1, 1], f(points[:frame+1].T), color='r')  # 画出路径

    # 显示每一帧的w1, w2, f(w)
    current_w1 = points[frame][0]
    current_w2 = points[frame][1]
    current_f = f(points[frame])
    text_display.set_text(f'(w1, w2, f(w)): ({current_w1:.2f}, {current_w2:.2f}, {current_f:.2f})')
    
    ax.set_title(f"Gradient Descent lr={learning_rate}")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("f(w)")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, f([4, 4])])

    # 显示每一帧的文本
    ax.text2D(0.05, 0.9, f'(w1, w2, f(w)): ({current_w1:.2f}, {current_w2:.2f}, {current_f:.2f})', transform=ax.transAxes)

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(points), repeat=False)
ani.save(f"ex2 lr={learning_rate}.gif")

plt.show()