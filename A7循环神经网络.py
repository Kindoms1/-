import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义 GRU 模型
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out
    

class LSTMWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.3):
        super(LSTMWithDropout, self).__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)  # 添加dropout层
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # 在连接全连接层之前应用dropout
        out = self.fc(out)
        return out
    

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

    # 记录总训练时间
    start_time = time.time()

    # 记录损失值
    loss_values = []

    # 训练模型
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.squeeze(1).to(device)  # 将图像从 (batch_size, 1, 28, 28) 转换为 (batch_size, 28, 28)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # 钩子机制
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 记录每个 batch 的损失值
            loss_values.append(loss.item())

            # 计算当前batch的准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # 计算总训练时间
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
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    # 返回结果
    return loss_values, total_training_time, accuracy

# 主程序
def run_rnn_model(models, hidden_size, epochs, batch_size, learning_rate, device):
    train_loader, test_loader = get_dataloaders(batch_size)

    input_size = 28  # 每行28个像素
    num_classes = 10  # 数字0-9

    results = {}

    for model_name in models:
        if model_name == 'RNN':
            model = RNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == 'LSTM':
            model = LSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == 'GRU':
            model = GRU(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == 'LSTMWithDropout':
            model = LSTMWithDropout(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        else:
            raise ValueError("Invalid model type. Choose from 'RNN', 'LSTM', or 'GRU'.")

        print(f"\nTraining {model_name} model...")
        loss_values, training_time, accuracy = train_and_evaluate(
            model, train_loader, test_loader, device, epochs, learning_rate
        )
        results[model_name] = {
            'loss_values': loss_values,
            'training_time': training_time,
            'accuracy': accuracy,
        }

    # 绘制损失曲线对比
    plt.figure(figsize=(12, 6))
    for model_name, result in results.items():
        plt.plot(result['loss_values'], label=model_name)
    plt.title('Loss Curve Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制训练时间对比图
    plt.figure(figsize=(8, 5))
    model_names = list(results.keys())
    training_times = [results[name]['training_time'] for name in model_names]
    plt.bar(model_names, training_times, color=['blue', 'orange', 'green'])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.show()

    # 绘制准确率对比图
    plt.figure(figsize=(8, 5))
    accuracies = [results[name]['accuracy'] for name in model_names]
    plt.bar(model_names, accuracies, color=['blue', 'orange', 'green'])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.show()


batch_size = 128
epochs = 10
hidden_size = 128
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 运行模型
run_rnn_model(['LSTM','LSTMWithDropout'], hidden_size, epochs, batch_size, learning_rate, device)
