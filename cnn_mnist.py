import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SIZE = 64  # 神经网络每次处理的数据量
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 选择设备，显然，本机只能支持CPU，不支持CUDA加速
EPOCHS = 5  # 训练轮数，即一共对整个数据集遍历多少轮
# INPUT_SIZE是图像的像素大小，也为输入层中神经元的个数，每个神经元接收一个像素
INPUT_SIZE = 784
# HIDDEN_SIZE是隐藏层中神经元的个数
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
LEARNING_RATE = 0.01

# 定义图像变换对象，来对图像进行预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 将图像对象转化为张量
        # 对图像进行标准化，这里均值取0.1307，方差取0.3081
        transforms.Normalize(mean=0.1307, std=0.3081)
    ]
)

# 下载数据集
data_train = datasets.MNIST(root='data_mnist/', transform=transform, train=True, download=True)
data_test = datasets.MNIST(root='data_mnist/', transform=transform, train=False)
# 加载数据集
train_loader = DataLoader(dataset=data_train,
                          shuffle=True,
                          batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=data_test,
                         shuffle=True,
                         batch_size=1)


# 构建全连接网络模型
class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        # 引入ReLU函数
        self.relu = nn.ReLU()
        # 这里有1个隐藏层，因此需要分别设置两组参数，一组是由输出层到隐藏层，另一组是由隐藏层到输出层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    # 进行前项传播
    def forward(self, x):
        # 将输入值传入神经网络，经过fc1进行计算后，传入隐藏层
        out = self.fc1(x)
        # 将传入隐藏层的数据通过ReLU函数重新映射到[0,+无穷)
        out = self.relu(out)
        # 将经过ReLU函数映射过的数据经过fc2计算后，传入输出层
        out = self.fc2(out)
        # 返回输出结果
        return out


# 初始化神经网络模型
model = FCNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
# 定义Adam优化器，并设置学习率
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 总步数为训练集中数据的个数
total_step = len(train_loader)


# 定义模型训练函数
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    # 枚举数据集中的每一个数据
    for batch_index, (data, label) in enumerate(train_loader):
        # 读入数据与数据标签
        data, label = data.reshape(-1, 28 * 28).to(device), label.to(device)
        # 将优化器梯度置零
        optimizer.zero_grad()
        # 代入模型，通过前向传播计算预测值
        output = model(data)
        # 计算交叉熵损失
        loss = F.cross_entropy(output, label)
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 每100个数据输出一个状态结果
        if batch_index % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, EPOCHS, batch_index + 1, total_step, loss))


# 定义模型测试函数
def test_model(model, device, test_loader):
    model.eval()
    # 定义变量
    correct = 0.0
    test_loss = 0.0
    # 在测试过程中，不需要计算梯度，只需要验证即可，因此设置no_grad
    with torch.no_grad():
        for data, label in test_loader:
            # 读取数据集与对应的标签
            data, label = data.reshape(-1, 28 * 28).to(device), label.to(device)
            # 通过前向传播进行预测
            output = model(data)
            # 计算交叉熵损失，并进行累计求和
            test_loss += F.cross_entropy(output, label).item()
            # 输出结果的矩阵中，值最大的就是预测的结果
            _, predict = torch.max(output, dim=1)
            correct += predict.eq(label.view_as(predict)).sum().item()

        # 计算平均损失
        test_loss /= len(test_loader.dataset)
        print(
            "Test -- Average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100.0 * correct / len(test_loader)))


# 执行10轮训练
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)

import matplotlib.pyplot as plt
figure = plt.figure()
num_of_images = 10
for imgs, targets in train_loader:
    break
for index in range(num_of_images):
    plt.subplot(6, 10, index+1)
    plt.axis('off')
    img = imgs[index, ...]
    output = model(img.reshape(-1, 28*28).to(DEVICE))
    _, predict = torch.max(output, dim=1)
    print(predict)
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
plt.show()