import numpy as np
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, w, grad):
        raise NotImplementedError

class Adagrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-6):
        super().__init__(lr)
        self.epsilon = epsilon  # 防止分母为零的小值
        self.G = np.zeros(2) # 累积梯度平方初始化为空

    def update(self, w, grad):
        self.G += grad**2
        return w - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.E_g2 = np.zeros(2)

    def update(self, w, grad):
        self.E_g2 = self.gamma * self.E_g2 + (1 - self.gamma) * grad**2  #grad指数衰减移动平均
        return w - self.lr * grad / (np.sqrt(self.E_g2) + self.epsilon)

class AdaDelta(Optimizer):
    def __init__(self, lr=0.01, rho=0.95, epsilon=1e-5):
        super().__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.E_g2 = np.zeros(2)
        self.E_dx2 = np.zeros(2)

    def update(self, w, grad):  # 参数更新差值的平方的指数衰减移动平均
        self.E_g2 = self.rho * self.E_g2 + (1 - self.rho) * grad**2
        delta_w = -np.sqrt(self.E_dx2 + self.epsilon) / np.sqrt(self.E_g2 + self.epsilon) * grad
        self.E_dx2 = self.rho * self.E_dx2 + (1 - self.rho) * delta_w**2
        return w + delta_w

class SGDWithMomentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = np.zeros(2)

    def update(self, w, grad):
        self.v = self.momentum * self.v - self.lr * grad
        return w + self.v

class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(2) # 一阶动量
        self.v = np.zeros(2) # 二阶动量
        self.t = 0

    def update(self, w, grad):
        self.t += 1
        # 偏差修正
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def objective(w):
    return 0.1 * w[0]**2 + 2 * w[1]**2

def gradient(w):
    return np.array([0.2 * w[0], 4 * w[1]])

def run_optimizer(optimizer, w_init, num_steps=1000):
    w = w_init.copy()
    path = [w.copy()]
    losses = []  # Loss recording
    for _ in range(num_steps):
        loss = objective(w)
        losses.append(loss)
        grad = gradient(w)
        w = optimizer.update(w, grad)
        path.append(w.copy())
    return np.array(path), losses

w_init = np.array([5.0, 5.0])
optimizers = {
    "Adagrad": Adagrad(lr=1),
    "RMSProp": RMSProp(lr=0.01),
    "AdaDelta": AdaDelta(),
    "SGD with Momentum lr=0.01": SGDWithMomentum(lr=0.01),
    "SGD with Momentum lr=0.1": SGDWithMomentum(lr=0.1),
    "Adam": Adam(lr=0.1)
}
paths = {}
loss_results = {}

for name, optimizer in optimizers.items():
    paths[name], loss_results[name] = run_optimizer(optimizer, w_init, num_steps=1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 创建2行3列的子图布局
axes = axes.flatten()  # 将子图数组平铺成一维

for i, (name, path) in enumerate(paths.items()):
    ax = axes[i]
    ax.plot(path[:, 0], path[:, 1], label=name, linewidth=1.5)
    ax.scatter(path[-1, 0], path[-1, 1], marker='x', color='red', label="End")
    ax.set_title(f"{name} Path")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.axhline(0, color='black', linewidth=0.5, linestyle="--")
    ax.axvline(0, color='black', linewidth=0.5, linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

# 删除多余的空白子图
for j in range(len(paths), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
for name, losses in loss_results.items():
    plt.plot(losses, label=name, linewidth=1.5)

plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


print("Final parameter values and corresponding losses:\n")
print(f"{'Optimizer':<20}{'Final Parameters':<40}{'Final Loss':<20}")
print("-" * 80)
for name, optimizer in optimizers.items():
    final_params = paths[name][-1]
    final_loss = loss_results[name][-1]
    print(f"{name:<20}{str(final_params):<40}{final_loss:<20.5e}")
