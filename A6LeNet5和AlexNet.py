import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self, input_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (1,32,32) -> (6,28,28)
        x = F.max_pool2d(x, 2) # (6,28,28) -> (6,14,14)
        x = F.relu(self.conv2(x)) # (6,14,14) -> (16,10,10)
        x = F.max_pool2d(x, 2) # (16,10,10) -> (16,5,5)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义 AlexNet 模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2), # 224×224×3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 64×55×55 → 64×27×27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 192×27×27 → 192×13×13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 384×13×13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 256×13×13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256×13×13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 256×6×6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        feature = self.features(x)
        feature = feature.view(feature.size(0), 256 * 6 * 6)
        output = self.classifier(feature)
        return output


# # 训练和评估函数
# def train_and_evaluate(model, train_loader, test_loader, device, dataset):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     epochs = 10

#     # 记录总训练时间
#     start_time = time.time()

#     # 训练模型
#     model.train()
#     for epoch in range(epochs):
#         epoch_start_time = time.time()  # 记录每个epoch的开始时间
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#             # 计算当前batch的准确率
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         epoch_end_time = time.time()  # 记录每个epoch的结束时间
#         epoch_time = epoch_end_time - epoch_start_time
#         epoch_accuracy = 100 * correct / total
#         print(f'Epoch [{epoch+1}/{epochs}], '
#               f'Loss: {running_loss/len(train_loader):.4f}, '
#               f'Accuracy: {epoch_accuracy:.2f}%, '
#               f'Epoch Time: {epoch_time:.4f} seconds')

#     # 计算总训练时间
#     end_time = time.time()
#     total_training_time = end_time - start_time
#     print(f"Training completed in {total_training_time:.4f} seconds on {device}.")

#     # 保存模型
#     model_name = type(model).__name__
#     model_path = f"{model_name}_{dataset}_.pth"
#     torch.save(model.state_dict(), model_path)
#     print(f"Model saved as {model_path}")

#     # 评估模型
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f'Accuracy: {accuracy:.2f}%')

# 用于记录结果的全局变量
results = {
    'models': [],
    'dataset': [],
    'train_time': [],
    'accuracy': [],
    'loss_curves': []
}

# 修改 train_and_evaluate 函数
def train_and_evaluate(model, train_loader, test_loader, device, dataset):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    # 记录总训练时间
    start_time = time.time()

    # 存储 loss 曲线
    loss_curve = []

    # 训练模型
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())  # 记录每个 batch 的 loss

    # 计算总训练时间
    end_time = time.time()
    total_training_time = end_time - start_time

    # 保存模型
    model_name = type(model).__name__
    model_path = f"{model_name}_{dataset}_.pth"
    torch.save(model.state_dict(), model_path)

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # 存储结果
    results['models'].append(model_name)
    results['dataset'].append(dataset)
    results['train_time'].append(total_training_time)
    results['accuracy'].append(accuracy)
    results['loss_curves'].append(loss_curve)

    print(f'{model_name} on {dataset}: '
          f'Training Time: {total_training_time:.2f}s, '
          f'Accuracy: {accuracy:.2f}%')

def run(net, dataset, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 设置输入通道数和图像变换
    if dataset == 'CIFAR10':
        input_channels = 3
        if net == 'AlexNet':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整尺寸为 224x224
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        elif net == 'LeNet5':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),  # 调整尺寸为 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
    elif dataset == 'MNIST':
        input_channels = 1
        if net == 'AlexNet':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整尺寸为 224x224
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif net == 'LeNet5':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),  # 调整尺寸为 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    else:
        raise ValueError("Unsupported dataset. Please use 'CIFAR10' or 'MNIST'.")

    train_dataset = datasets.__dict__[dataset](root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.__dict__[dataset](root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 初始化模型
    if net == 'LeNet5':
        model = LeNet5(input_channels=input_channels)
    elif net == 'AlexNet':
        model = AlexNet(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError("Unsupported network. Please use 'LeNet5' or 'AlexNet'.")

    train_and_evaluate(model, train_loader, test_loader, device,  dataset)

run('LeNet5', 'MNIST', 10)
run('AlexNet', 'MNIST', 10)
run('LeNet5', 'CIFAR10', 10)
run('AlexNet', 'CIFAR10', 10)



import matplotlib.pyplot as plt

# 绘制训练时间对比图
plt.figure(figsize=(8, 5))
plt.bar(results['models'], results['train_time'], color=['blue', 'green', 'red', 'purple'])
plt.title("Training Time Comparison")
plt.ylabel("Time (seconds)")
plt.xlabel("Model")
plt.show()

# 绘制最终准确率对比图
plt.figure(figsize=(8, 5))
plt.bar(results['models'], results['accuracy'], color=['blue', 'green', 'red', 'purple'])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.show()

# 绘制 Loss 随 Iterations 变化图
plt.figure(figsize=(10, 6))
for i in range(len(results['loss_curves'])):
    plt.plot(results['loss_curves'][i], label=f"{results['models'][i]} ({results['dataset'][i]})")
plt.title("Loss Curve Comparison")
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.legend()
plt.show()
