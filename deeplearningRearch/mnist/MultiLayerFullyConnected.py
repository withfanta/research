import torch
import torch.nn as nn
from torch. autograd import Variable
import torch.optim as optim
import torch.nn. functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt     # %matplotlib inline 可以让Jupyter Notebook直接输出图像
import pylab
from torch.utils.data import DataLoader
from torchvision.datasets import mnist

train_batch_size = 64  #指定DataLoader在训练集中每批加载的样本数量
test_batch_size = 128  #指定DataLoader在测试集中每批加载的样本数量
num_epoches = 20 # 模型训练轮数
lr = 0.01  #设置SGD中的初始学习率
momentum = 0.5 #设置SGD中的冲量

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.1307], [0.3081])])
#4下载和分批加载数据集

#将训练和测试数据集下载到同目录下的data文件夹下
train_dataset = mnist.MNIST('.\data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('.\data', train=False, transform=transform,download=True)

#dataloader是一个可迭代对象，可以使用迭代器一样使用。
#其中shuffle参数为是否打乱原有数据顺序
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


# 5定义一个神经网络模型

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Linear(n_hidden_2, out_dim)  # 最后一层接Softmax所以不需要ReLU激活

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# Sequential() 即相当于把多个模块按顺序封装成一个模块

# 实例化网络模型

# 检测是否有可用的GPU，否则使用cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 网络模型参数分别为：输入层大小、隐藏层1大小、隐藏层2大小、输出层大小（10分类）
model = Net(28 * 28, 300, 100, 10)
# 将模型移动到GPU加速计算
model.to(device)

# 定义模型训练中用到的损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# parameters()将model中可优化的参数传入到SGD中
# 6对模型进行训练






# 开始训练
losses = []  # 记录训练集损失
acces = []  # 记录训练集准确率
eval_losses = []  # 记录测试集损失
eval_acces = []  # 记录测试集准确率

train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()  # 指明接下来model进行的是训练过程
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.9
    for img, label in train_loader:
        img = img.to(device)  # 将img移动到GPU计算
        label = label.to(device)
        img = img.view(img.size(0), -1)  # 把输入图像的维度由四维转化为2维，因为在torch中只能处理二维数据
        # img.size(0)为取size的第0个参数即此批样本的个数，-1为自适应参数
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()  # 先清空上一轮的梯度
        loss.backward()  # 根据前向传播得到损失，再由损失反向传播求得各个梯度
        optimizer.step()  # 根据反向传播得到的梯度优化模型中的参数

        train_loss += loss.item()  # 所有批次损失的和
        # 计算分类的准确率
        _, pred = out.max(1)  # 返回输出二维矩阵中每一行的最大值及其下标，1含义为以第1个维度（列）为参考
        # pred=torch.argmax(out,1)
        num_correct = (pred == label).sum().item()
        # num_correct = pred.eq(label).sum().item()
        acc = num_correct / img.shape[0]  # 每一批样本的准确率
        train_acc += acc

    losses.append(train_loss / len(train_loader))  # 所有样本平均损失
    acces.append(train_acc / len(train_loader))  # 所有样本的准确率

    # 7运用训练好的模型在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # 将模型改为预测模式
    model.eval()  # 指明接下来要进行模型测试（不需要反向传播）
    # with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    train_loss_all.append(train_loss / len(train_loader))
    test_loss_all.append(eval_loss / len(test_loader))
    train_acc_all.append(train_acc / len(train_loader))
    test_acc_all.append(eval_acc / len(test_loader))


    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                  eval_loss / len(test_loader), eval_acc / len(test_loader)))

plt.title("MultiLayerFullyConnected")
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, 'ro-', label='Train loss')
plt.plot(test_loss_all, 'bs-', label='Val loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(train_acc_all, 'ro-', label='Train acc')
plt.plot(test_acc_all, 'bs-', label='Val acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.show()
