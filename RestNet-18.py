# -*- coding:utf-8 -*-
# @Author  : t0ugh
# @Date    : 2025/5/15 14:09
# @Description: 
# @Version : v1
import torch
import torch.nn as nn

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # shortcut 路径（调整维度如果需要）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out

# ResNet 网络定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始通道数

        # 1. 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 2. 残差层（4个阶段，对应ResNet-18的[2,2,2,2]结构）
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 3. 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(512, num_classes)  # 全连接层

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        # 第一个残差块：可能包含维度调整（stride和通道数变化）
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels  # 更新当前通道数

        # 后续残差块：stride=1，通道数不变
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        out = self.relu(self.bn1(self.conv1(x)))  # 3x3卷积+BN+ReLU

        # 残差层
        out = self.layer1(out)  # 输出维度：64通道，尺寸不变（stride=1）
        out = self.layer2(out)  # 输出维度：128通道，尺寸减半（stride=2）
        out = self.layer3(out)  # 输出维度：256通道，尺寸再减半（stride=2）
        out = self.layer4(out)  # 输出维度：512通道，尺寸再减半（stride=2）

        # 全局平均池化：将特征图压缩为 [batch_size, 512, 1, 1]
        out = self.avg_pool(out)

        # 展平并分类
        out = out.view(out.size(0), -1)  # 展平为 [batch_size, 512]
        out = self.fc(out)  # 全连接层输出 [batch_size, num_classes]
        return out

# 创建 ResNet-18 模型
def ResNet18(num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

# 示例：训练和测试
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型
    model = ResNet18(num_classes=10).to(device)
    # 随机输入（模拟 CIFAR-10：batch_size=4, channels=3, 32x32）
    x = torch.randn(4, 3, 32, 32).to(device)
    # 前向传播
    output = model(x)
    print(f"Output shape: {output.shape}")  # 应为 [4, 10]

    # 简单训练示例（以 CIFAR-10 为例）
    import torchvision
    import torchvision.transforms as transforms

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练一个 epoch
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
