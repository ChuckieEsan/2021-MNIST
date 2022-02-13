import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# 设置超参数
# 每次向神经网络传入4张图片
BATCH_SIZE = 4
# 总的训练轮次
EPOCH_NUM = 5
# 选择部署设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置学习率
LEARNING_RATE = 1e-3

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置图像预处理方法
pipeline = transforms.Compose([
    transforms.ToTensor(),
    # 进行标准化，由于传入的是一个RGB三通道图片，所以要分三个维度，这里方差选择0.2，均值选取0.4
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    # 将图像归一化并裁剪，并取中间部分。此时图片大小为3 * 224 * 224
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

transform = transforms.Compose([
    transforms.ToTensor(),
    # 将图像归一化并裁剪，并取中间部分。此时图片大小为3 * 224 * 224
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
# 设置图像路径
image_path = "./data_dog_or_cat/train__"


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
data_ = DataProcess(image_path=image_path, transform=transform)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
# 进行数据集的随机划分，设置随机数种子，以确保结果的可重复性
random_state = 1
torch.manual_seed(random_state)
train_data, test_data = random_split(data, [train_size, test_size])

# 加载训练数据与测试数据
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
_dataloader = DataLoader(data_, batch_size=BATCH_SIZE)


# 定义卷积神经网络模型，这里用的是VGG16模型
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        # out = F.log_softmax(out, dim=1)

        return out


model = VGG16().to(DEVICE)  # 部署模型到cpu上
# 设置分类器
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
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
            out = model(im)
            test_loss += loss_func(out, label).item()
            _, pre_class = torch.max(out, 1)
            correct += (pre_class == label).sum().item()

        test_loss /= size
        correct /= size
        print("Test\n  Accuracy: %.1f,  Average loss:%.8f \n" % (100 * correct, test_loss))
        return test_loss, correct


train_loss_all = np.zeros(EPOCH_NUM)
test_loss_all = np.zeros(EPOCH_NUM)
correct_all = np.zeros(EPOCH_NUM)


def train_test():
    for t in range(EPOCH_NUM):
        print(f"Epoch {t}\n-------------------------------")
        train_loss_all[t] = train(train_dataloader, model, loss_func, optimizer)
        test_loss_all[t], correct_all[t] = test(test_dataloader, model, loss_func)
    print("Done!")


def only_test():
    model = models.vgg16()
    model.load_state_dict(torch.load('./models/vgg16_model.pth'))
    figure = plt.figure()
    num_of_images = 10
    for imgs, targets in _dataloader:
        break
    im = imgs[0, ...]
    out = model(im)
    _, pre_class = torch.max(out, 1)
    plt.show()


