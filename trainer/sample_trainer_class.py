import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, trainloader, epochs=10):
        self.model.train()  # 设置模型为训练模式
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

    def test(self, testloader):
        self.model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total}%")

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为 224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 下载并加载 MNIST 数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 定义 ResNet 模型
resnet = torchvision.models.resnet101(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # 修改最后的全连接层为 10 个输出（MNIST 有 10 个类别）

# 将模型移动到设备
resnet = resnet.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# 创建 Trainer 实例
trainer = Trainer(resnet, criterion, optimizer, device)

# 训练和测试
trainer.train(trainloader, epochs=10)
trainer.test(testloader)
