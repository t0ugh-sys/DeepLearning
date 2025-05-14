import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm库用于显示进度条
import os

plt.rcParams["font.family"] = ["SimHei"]

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集并分割
dataset = datasets.ImageFolder(root='flower_photos', transform=transform_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_val  # 验证集使用不同变换

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 定义 VGG 模型
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 5)

    def forward(self, x):
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.pool(x)
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = torch.relu(self.conv3_3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4_1(x))
        x = torch.relu(self.conv4_2(x))
        x = torch.relu(self.conv4_3(x))
        x = self.pool(x)
        x = torch.relu(self.conv5_1(x))
        x = torch.relu(self.conv5_2(x))
        x = torch.relu(self.conv5_3(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if os.path.exists('final_VGG-16.pth'):
    model = VGG().to(device)
    model.load_state_dict(torch.load('final_VGG-16.pth'))
else:
# 初始化模型
    model = VGG().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练参数
epochs = 30
train_losses, val_losses, val_accuracies = [], [], []

# 训练和验证
best_accuracy = 0
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # 训练阶段使用tqdm显示进度条
    model.train()
    running_loss = 0.0
    train_iter = tqdm(train_loader, desc="训练中", unit="batch")
    for data, target in train_iter:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 更新进度条显示
        train_iter.set_postfix({"损失": loss.item()})

    train_losses.append(running_loss / len(train_loader))

    # 验证阶段使用tqdm显示进度条
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    val_iter = tqdm(val_loader, desc="验证中", unit="batch")
    with torch.no_grad():
        for data, target in val_iter:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 更新进度条显示
            val_iter.set_postfix({"损失": loss.item(), "准确率": f"{100 * correct / total:.2f}%"})

    val_losses.append(val_loss / len(val_loader))
    accuracy = 100.0 * correct / total
    val_accuracies.append(accuracy)

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_VGG-16.pth')
        print(f"已保存最佳模型 (准确率: {accuracy:.2f}%)")

    # 打印本轮训练结果
    print(f"训练损失: {train_losses[-1]:.4f}, 验证损失: {val_losses[-1]:.4f}, 验证准确率: {accuracy:.2f}%")

# 保存最终模型
torch.save(model.state_dict(), 'final_VGG-16.pth')
print(f"训练完成，最终模型已保存，最佳验证准确率: {best_accuracy:.2f}%")

# 绘制训练损失、验证损失和验证准确率曲线
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs + 1), train_losses, 'b-', label='训练损失')
plt.title('训练损失变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, epochs + 1), val_losses, 'g-', label='验证损失')
plt.title('验证损失变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, epochs + 1), val_accuracies, 'r-', label='验证准确率')
plt.title('验证准确率变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('准确率 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()