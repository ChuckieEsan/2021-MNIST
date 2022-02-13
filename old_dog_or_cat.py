import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# 设置超参数
# 每次向神经网络传入4张图片
BATCH_SIZE = 4
# 总的训练轮次
EPOCH_NUM = 5
# 选择部署设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置学习率
LEARNING_RATE = 1e-3

# 设置图像预处理方法
pipeline = transforms.Compose([
    transforms.ToTensor(),
    # 进行标准化，由于传入的是一个RGB三通道图片，所以要分三个维度，这里方差选择0.2，均值选取0.4
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    # 将图像归一化并裁剪，并取中间部分。此时图片大小为3 * 224 * 224
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
# 设置图像路径
image_path = "./data_dog_or_cat/train"


# 定义数据处理类，这里数据处理的是train目录下的图片，test目录下的图片没有标签，不能作为训练集或测试集
class DataProcess(Dataset):
    def __init__(self, image_path, transform):
        super(DataProcess, self).__init__()
        self.image_path = image_path
        self.images = os.listdir(image_path)
        self.transform = transform

    # 重载__len__方法
    def __len__(self):
        return len(self.images)

    # 重载__getitem__方法
    def __getitem__(self, index):
        # 文件名为 dog.0.jpg，因此要进行文件名的分割
        img_name = self.images[index]
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path)  # 打开图像
        img = self.transform(img)  # 将图像按照设定的变换方式变换

        # 设置分类标签
        if str.split(img_name, '.')[0] == "cat":
            label = 0
        else:
            label = 1

        return img, label


# 对数据进行初始化，并对数据集进行划分，训练集占比0.8，测试集占比0.2
data = DataProcess(image_path=image_path, transform=pipeline)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
# 进行数据集的随机划分，设置随机数种子，以确保结果的可重复性
random_state = 1
torch.manual_seed(random_state)
train_data, test_data = random_split(data, [train_size, test_size])

# 加载训练数据与测试数据
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


# 定义卷积神经网络模型
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 定义第一个卷积层，输入3层，提取6个特征，使用5*5的卷积核，此时输出的是6 * 220 * 220
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义池化层，将图像的长和宽变为原来的1/2，此时输出的是6 * 110 * 110，经过第二次池化，输出的是16 * 53 * 53
        self.maxpool = nn.MaxPool2d(2, 2)
        # 定义第二个卷积层，输入6层，提取16个特征，使用5*5的卷积核，此时输出的是16 * 106 * 106
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 设置第一个全连接层，输入为16 * 53 * 53，输出为1025
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        # 设置第二个全连接层，输入为1025，输出512
        self.fc2 = nn.Linear(1024, 512)
        # 设置第三个全连接层，输入512，输出2 (结果要么0，要么1,即要么猫要么狗)
        self.fc3 = nn.Linear(512, 2)

    # 定义前项遍历函数
    def forward(self, x):
        # 先卷积，再ReLU，然后池化，重复两次，即有2个卷积层-ReLU-池化层
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        # 进行全连接，这里执行了3次全连接，最后一次全连接操作即输出结果
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNNNet().to(DEVICE)  # 部署模型到cpu上
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  # 设置SGD优化器
loss_func = torch.nn.CrossEntropyLoss()  # 定义损失函数为交叉熵函数


# 定义训练函数
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for iteration, (im, label) in enumerate(dataloader):
        im, label = im.to(DEVICE), label.to(DEVICE)

        # 前向传播
        pre = model(im)
        loss = loss_func(pre, label)

        # 后向传播，即先让梯度置0，然后让损失后向传播，随后更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            loss, current = loss.item(), iteration * len(im)
            print("loss: %.4f, current:%5d/size:%5d" % (loss, current, size))

    return loss


# 定义测试函数
def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    # 测试过程是没有梯度计算的，只是单纯的检验
    with torch.no_grad():
        for im, label in dataloader:
            im, label = im.to(DEVICE), label.to(DEVICE)
            pre = model(im)
            test_loss += loss_func(pre, label).item()
            pre_class = pre.argmax(dim=1)  # 求得输出结果中较大值对应的label，该label值就对应了一个分类结果
            correct += (pre_class == label).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print("Test\n  Accuracy: %.1f,  Average loss:%.8f \n" % (100 * correct, test_loss))
        return test_loss, correct


train_loss_all = np.zeros(EPOCH_NUM)
test_loss_all = np.zeros(EPOCH_NUM)
correct_all = np.zeros(EPOCH_NUM)
for t in range(EPOCH_NUM):
    print(f"Epoch {t}\n-------------------------------")
    train_loss_all[t] = train(train_dataloader, model, loss_func, optimizer)
    test_loss_all[t], correct_all[t] = test(test_dataloader, model, loss_func)
print("Done!")
