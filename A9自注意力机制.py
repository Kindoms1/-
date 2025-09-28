import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np


class CustomQKVAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CustomQKVAttention, self).__init__()
        self.query_projection = nn.Linear(hidden_size, hidden_size)  # 用于生成 Q
        self.key_projection = nn.Linear(hidden_size, hidden_size)    # 用于生成 K
        self.value_projection = nn.Linear(hidden_size, hidden_size)  # 用于生成 V
        self.scale = 1.0 / (hidden_size ** 0.5)  # 缩放因子

    def forward(self, query_input, key_input, value_input, mask=None):
        # 投影得到 Q, K, V
        Q = self.query_projection(query_input)  # (batch_size, seq_len, hidden_size)
        K = self.key_projection(key_input)      # (batch_size, seq_len, hidden_size)
        V = self.value_projection(value_input)  # (batch_size, seq_len, hidden_size)

        # 计算注意力分数 (batch_size, seq_len, seq_len)
        # Q(seq_len, hidden_size) K.transpose(1, 2)(hidden_size, seq_len) -> (seq_len, seq_len) -> (batch_size, seq_len, seq_len)
        # attention_scores[b, i, j] 第b个批次中第i个时间步的Query和第j个时间步的Key的相似度
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale # 批量矩阵乘法 点积

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 计算上下文向量 (batch_size, seq_len, hidden_size)
        context = torch.bmm(attention_weights, V)

        return context, attention_weights

class RNNWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNWithAttention, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.attention = CustomQKVAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # RNN 输出
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_size)

        # 使用自定义 Q-K-V 注意力机制
        query = rnn_out  # RNN 输出作为 Query
        key = rnn_out    # RNN 输出作为 Key
        value = rnn_out  # RNN 输出作为 Value

        context, attn_weights = self.attention(query, key, value) # 调用attention的forward

        # 取最后一个时间步的上下文向量
        out = self.fc(context[:, -1, :])  # context: (batch_size, seq_len, hidden_size)

        return out, attn_weights
    
# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out, None  # 返回 None 作为注意力权重，保持接口一致
    
def visualize_attention_scores(attention_scores, input_sequence):
    """
    可视化注意力得分的热图。
    """
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(attention_scores, annot=False, fmt='.2f', cmap='viridis')
    plt.title("Attention Scores Heatmap")
    plt.xlabel("Key Sequence")
    plt.ylabel("Query Sequence")
    plt.show()


# 数据加载和预处理
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader, device, epochs, learning_rate, dataset='MNIST'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    loss_values = []
    accuracy_values = []  # 添加准确率记录列表

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs, attn_weights = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss_values.append(loss.item())

        # 计算并记录每个epoch的训练准确率
        epoch_accuracy = 100 * correct / total
        accuracy_values.append(epoch_accuracy)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%')

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Training completed in {total_training_time:.4f} seconds on {device}.")

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    model.eval()
    for images, labels in test_loader:
        images = images.squeeze(1).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs, attn_weights = model(images[:1])  # 仅使用一个样本进行展示
            if attn_weights is not None:
                attn_weights_np = attn_weights.cpu().squeeze(0).numpy()  # 提取注意力权重
                visualize_attention_scores(attn_weights_np, images[:1])
                break

    return loss_values, accuracy_values, total_training_time, test_accuracy

    

# 主程序
def run_rnn_model(models, hidden_size, epochs, batch_size, learning_rate, device):
    train_loader, test_loader = get_dataloaders(batch_size)

    input_size = 28
    num_classes = 10
    results = {}

    # 设置绘图样式
    plt.style.use('default')  # 使用默认样式
    colors = ['#2ecc71', '#e74c3c']  # 绿色和红色
    
    # 设置全局字体样式
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    for i, model_name in enumerate(models):
        if model_name == 'RNNWithAttention':
            model = RNNWithAttention(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == 'RNN':
            model = RNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        else:
            raise ValueError("Invalid model type. Choose 'RNNWithAttention' or 'RNN'.")
        
        print(f"\nTraining {model_name} model...")
        loss_values, accuracy_values, training_time, test_accuracy = train_and_evaluate(
            model, train_loader, test_loader, device, epochs, learning_rate
        )
        results[model_name] = {
            'loss_values': loss_values,
            'accuracy_values': accuracy_values,
            'training_time': training_time,
            'test_accuracy': test_accuracy,
        }

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    for i, (model_name, result) in enumerate(results.items()):
        losses = result['loss_values']
        n = len(losses)
        steps_per_epoch = n // epochs
        epoch_losses = [np.mean(losses[i:i+steps_per_epoch]) for i in range(0, n, steps_per_epoch)]
        plt.plot(epoch_losses, label=model_name, color=colors[i], linewidth=2, marker='o')
    plt.title('Loss Curves', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 绘制训练准确率曲线
    plt.figure(figsize=(10, 6))
    for i, (model_name, result) in enumerate(results.items()):
        plt.plot(result['accuracy_values'], label=model_name, 
                color=colors[i], linewidth=2, marker='o')
    plt.title('Training Accuracy Curves', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 绘制测试准确率对比
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    test_accuracies = [results[name]['test_accuracy'] for name in model_names]
    bars = plt.bar(model_names, test_accuracies, color=colors)
    plt.title('Test Accuracy Comparison', fontsize=12, pad=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.grid(True, alpha=0.3)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    # 绘制训练时间对比
    plt.figure(figsize=(10, 6))
    training_times = [results[name]['training_time'] for name in model_names]
    bars = plt.bar(model_names, training_times, color=colors)
    plt.title('Training Time Comparison', fontsize=12, pad=10)
    plt.ylabel('Time (seconds)', fontsize=10)
    plt.grid(True, alpha=0.3)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


batch_size = 128
epochs = 5
hidden_size = 128
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 运行模型
run_rnn_model(['RNN', 'RNNWithAttention'], hidden_size, epochs, batch_size, learning_rate, device)
