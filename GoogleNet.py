# -*- coding:utf-8 -*-
# @Author  : t0ugh
# @Date    : 2025/5/13 19:22
# @Description: 
# @Version : v1
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["SimHei"]

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # 增加随机缩放
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集并分割
dataset = datasets.ImageFolder(root=r'D:\data\cnn_project\VGG\flower_photos', transform=transform_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_val

# 打印类别分布
class_counts = [len([img for img in dataset.imgs if img[1] == i]) for i in range(5)]
print("每个类别的样本数:", class_counts)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Inception 模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 1x1 卷积分支
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 3x3 卷积分支（先 1x1 降维）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        # 5x5 卷积分支（先 1x1 降维）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        # 3x3 池化分支（后接 1x1 卷积）
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


# GoogLeNet 模型
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 模块
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        # 辅助分类器
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(528, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = self.pool2(torch.relu(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x)  # 第一个辅助分类器
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)  # 第二个辅助分类器
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.global_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.training:
            return x, aux1, aux2
        return x
# 初始化模型
if os.path.exists(r'best_GoogleNet.pth'):
    model = GoogLeNet(num_classes=5).to(device)
    model.load_state_dict(torch.load(r'best_GoogleNet.pth'))
else:
    model = GoogLeNet(num_classes=5).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练参数
epochs = 30
train_losses, val_losses, val_accuracies = [], [], []


# 训练和验证
best_accuracy = 0
for epoch in range(epochs):
    # 训练
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, aux1, aux2 = model(data)
        loss1 = criterion(output, target)
        loss2 = criterion(aux1, target)
        loss3 = criterion(aux2, target)
        loss = loss1 + 0.3 * loss2 + 0.3 * loss3  # 辅助损失权重为 0.3
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # 验证
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_losses.append(val_loss / len(val_loader))
    accuracy = 100.0 * correct / total
    val_accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_GoogleNet.pth')


    print(
        f'Epoch {epoch + 1}/{epochs}, 训练损失: {train_losses[-1]:.4f}, 验证损失: {val_losses[-1]:.4f}, 验证准确率: {accuracy:.2f}%')

# 保存最终模型
torch.save(model.state_dict(), 'final_GoogleNet.pth')

# 绘制曲线
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
plt.title('训练损失变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, len(val_losses) + 1), val_losses, 'g-', label='验证损失')
plt.title('验证损失变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='验证准确率')
plt.title('验证准确率变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('准确率 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()
