import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load MNIST dataset and get one sample image
transform_original = transforms.Compose([
    transforms.ToTensor()
])

transform_resized = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load a single image from the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_original)
image, label = mnist_dataset[0]  # Get the first image

# Load the resized version of the same image
mnist_dataset_resized = datasets.MNIST(root='./data', train=True, download=True, transform=transform_resized)
image_resized, label_resized = mnist_dataset_resized[0]  # Get the first image (resized)

# Plot the original and resized images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display original image
axs[0].imshow(image.squeeze(), cmap='gray')
axs[0].set_title('Original MNIST Image (28x28)')
axs[0].axis('off')

# Display resized image
axs[1].imshow(image_resized.squeeze(), cmap='gray')
axs[1].set_title('Resized MNIST Image (224x224)')
axs[1].axis('off')

plt.show()


import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 加载 CIFAR-10 数据集中的一张图片
transform_original = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_original)
original_image, _ = dataset[0]  # 获取第一张图片
print(original_image.size())

# 调整为 32x32 的变换（仅作示意，图像本身就是 32x32）
transform_resized = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset_resized = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_resized)
resized_image, _ = dataset_resized[0]

# 将图像转换为可绘制的格式 (C, H, W -> H, W, C)
original_image = original_image.permute(1, 2, 0).numpy()
resized_image = resized_image.permute(1, 2, 0).numpy()

# 显示图片对比
plt.figure(figsize=(12, 12))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Resized Image (224x224)")
plt.imshow(resized_image)
plt.axis('off')

plt.show()
