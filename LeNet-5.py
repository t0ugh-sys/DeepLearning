import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# 定义模型
class BackBone(nn.Module):
    """LeNet-5的卷积网络"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return x


class Head(nn.Module):
    """LeNet-5的全连接网络"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    """完整的LeNet-5模型"""

    def __init__(self):
        super().__init__()
        self.backbone = BackBone()
        self.head = Head()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# 初始化模型
model = LeNet5().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
epochs = 10

# 训练模型
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 计算平均训练损失
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch {epoch+1}/{epochs}, 损失Loss:{avg_loss:.4f}, 测试准确率: {accuracy:.2f}%')

torch.save(model.state_dict(), 'LeNet-5.pth')

# 绘制训练损失和测试准确率曲线
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=1)
plt.title('训练损失变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, 'r-', linewidth=1)
plt.title('测试准确率变化', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('准确率 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(90, 100)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()