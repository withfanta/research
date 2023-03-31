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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 完成图片路径处理
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
# 处理图片
def Myloader(filepath):
    return Image.open(filepath).convert('RGB')
# 重写Dataset类
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
    train_normal_data = init_data('img/normalImg', [0, 811])
    test_normal_data = init_data('img/normalImg', [811, 1217])
    train_target_data = init_data('img/targetImg', [0, 136])
    test_target_data = init_data('img/targetImg', [136, 205])
    data = train_normal_data+test_normal_data+train_target_data+test_target_data
    np.random.shuffle(data)
    data_size = len(data)
    train_data,val_data,test_data = data[:1200],data[1200:1201],data[1201:1422]
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=50, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=0)
    # print(Dtr, Val, Dte)
    return Dtr, Val, Dte

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

def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []
    for (data, target) in Val:
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)

def train():
    Dtr, Val, Dte = loadData()
    print('train...')
    epoch_num = 100
    best_model = None
    min_epochs = 5
    min_val_loss = 5
    model = cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCELoss().to(device)
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = []
        for batch_idx, (data, target) in enumerate(Dtr, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            # target = target.view(target.shape[0], -1)
            # print(target)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
        # validation
        val_loss = get_val_loss(model, Val)
        model.train()
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))
    torch.save(copy.deepcopy(model).state_dict(), "model/cnn.pkl")

def test():
    Dtr, Val, Dte = loadData()
    print(Dte)
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

train()
test()




