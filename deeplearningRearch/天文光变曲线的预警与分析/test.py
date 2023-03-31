import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader,dataset,Dataset
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import copy
import pandas as pd


# 遍历文件夹读为矩阵
def readFolderAsMatrix(foldername):
    filenames = os.listdir(foldername)
    martixList = []
    count = 0
    for filename in filenames:
        filePath = foldername+'/'+filename
        folder_data_read = pd.read_csv(filePath, sep=' ', header=None,skiprows=1, names=['JD', 'Magnorm', 'Mage'])
        martixList.append(folder_data_read)
        count += 1
    martixList_concat = pd.concat(martixList,axis=0,ignore_index=True)
    print(count)
    # print(martixList_concat)
    return martixList_concat

# 将数据切片为n个点构成的小片段
def imgSpilt(data_read,col_name1,col_name2,n):
    x = []
    y = []
    index = 0;
    while (index + n) < (data_read[col_name1].shape[0]):
        # print(data_read[col_name][index:index+50].tolist())
        x.append(data_read[col_name1][index:index + n].tolist())
        y.append(data_read[col_name2][index:index + n].tolist())
        index += n
        # print(index)
    return x,y

# 将文件夹目录的所有数据切成小片段
normalList = readFolderAsMatrix('testData/normalData')
targetList = readFolderAsMatrix('testData/targetData')
normal_x,normal_y = imgSpilt(normalList,'JD','Magnorm',50)
target_x,target_y = imgSpilt(targetList,'JD','Magnorm',50)


# 批量保存target图片
for i in range(len(target_x)-1):
    img_savePath = 'testData/targetImg'
    plt.ylim(target_y[i][0]-2, target_y[i][0]+2)
    plt.scatter(target_x[i], target_y[i], label="targetstar", color="black",s=50)
    plt.axis('off')
    # print(np.var(target_x[i]))
    if not os.path.exists(img_savePath):
        os.makedirs(img_savePath)
    # plt.figure(figsize=(5,5))
    # 把两条不相连的线剔除，并把波动较小的线剔除
    plt.savefig(os.path.join(img_savePath,'{num}'.format(num=i)),dpi = 10,bbox_inches='tight')
    plt.cla()


# 批量保存normal图片
for i in range(len(normal_x)-1):
    img_savePath = 'testData/normalImg'
    plt.ylim(normal_y[i][0]-2, normal_y[i][0]+2)
    plt.scatter(normal_x[i], normal_y[i], label="normalstar", color="black",s=50)
    plt.axis('off')
    if not os.path.exists(img_savePath):
        os.makedirs(img_savePath)
    # plt.figure(figsize=(5,5))
    # plt.imshow(plt,cmap='gray')
    # 把不相连的线剔除

    plt.savefig(os.path.join(img_savePath,'{num}'.format(num=i)),dpi = 10,bbox_inches='tight')
    plt.cla()

def init_data(path,lens):
    data = []
    min = lens[0]
    max = lens[1]
    count = 0;
    filenames = os.listdir(path)
    for filename in filenames:
        if (count >= min) and (count <= max):
            filePath = path + '/' + filename
            data.append([filePath,label(filePath)])
        count += 1
    # print(data)
    return data


# 根据标签进行判断
def label(filename):
    if 'target' in filename:
        return 1
    if 'normal' in filename:
        return 0

def Myloader(filepath):
    return Image.open(filepath).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

def loadData():
    transform = transforms.Compose([
        # transforms.CenterCrop(38),
        # transforms.Resize((38, 38)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    test_normal_data = init_data('testData/normalImg', [0, 120])
    test_target_data = init_data('testData/targetImg', [0, 135])
    data = test_target_data + test_normal_data
    np.random.shuffle(data)
    data_size = len(data)
    test_data = MyDataset(data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=0)
    # print(Dtr, Val, Dte)
    return Dte

# 定义模型
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
            ),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=1,
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(46*33*64,50)
        # self.fc2 = nn.Linear(64, 10)
        # self.out = nn.Linear(10, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        # print(x.shape[0])
        x = x.view(-1,46*33*64)
        x = self.fc1(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.out(x)
        x = F.log_softmax(x, dim=1)
        return x

def test():
    Dte = loadData()
    # 打印loadData里的图片
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load("model/cnn.pkl"), False)
    model.eval()
    total = 0
    current = 0
    for (data, target) in Dte:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = torch.max(outputs.data, 1)[1].data
        total += target.size(0)
        current += (predicted == target).sum()
        # print(predicted)
        # print(predicted==target)

    print('Accuracy:%d%%' % (100 * current / total))
test()