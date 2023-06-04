# 导入模块
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm



# data_tensor = torch.tensor([])
torch.set_printoptions(precision=7)
datalist = []
# 遍历文件夹读为矩阵
def readFolderAsMatrix(foldername):
    filenames = os.listdir(foldername)
    martixList = []
    count = 0
    pd.set_option('display.float_format', lambda x: '%.7f' % x)
    for filename in filenames:
        filePath = foldername+'/'+filename
        folder_data_read = pd.read_csv(filePath, sep=' ', header=None,skiprows=1, names=['JD', 'Magnorm', 'Mage'])
#         # folder_data_read['JD']= [i * 10000000 for i in folder_data_read['JD']]
#         # print(folder_data_read['JD'])
#         folder_data_tensor = torch.tensor(folder_data_read['JD'])
#         # print(folder_data_tensor)
#         # print(folder_data_read)
        martixList.append(folder_data_read)
#         # folder_data_read = folder_data_read['JD']
#         # folder_data_read = [float(i) for i in folder_data_read]
#         # folder_data_read = [("%.4f" % i) for i in folder_data_read]
#         data_tensor= folder_data_tensor[: 500]
#         datalist.append(data_tensor)
        count += 1
#         break
#     # martixList_concat = pd.concat(martixList,axis=0,ignore_index=True)
#     print(count)
#     # print(martixList_concat)
#     # return martixList_concat
    return martixList

# df = pd.read_csv('./wind.csv', index_col=0)
#
df = pd.read_csv('sourceData/normalData/ref_023_15730595-G0013_356332_4877', sep=' ', header=None,skiprows=1, names=['JD', 'Magnorm', 'Mage'])
# df = readFolderAsMatrix('sourceData/normalData')
# df = pd.concat(df)
# df['JD'] = df['JD']
# 2.将数据进行标准化
scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['JD']).reshape(-1, 1))

# df = df.to_list()
# print(datalist)
# data = []
# x_data = datalist
# x_data = torch.stack(x_data).unsqueeze(1)
# x_data = x_data.view(-1)
# x_data = x_data.tolist()
# print(x_data)
# df = df[: 100]
# df = pd.read_csv('sourceData/normalData/ref_023_15730595-G0013_356332_4877', sep=' ', header=None,skiprows=1, names=['JD', 'Magnorm', 'Mage'])
print(df)
# # 2.将数据进行标准化
# scaler = MinMaxScaler()
# scaler_model = MinMaxScaler()
# data = scaler_model.fit_transform(np.array(df))
# scaler.fit_transform(np.array(data).reshape(-1, 1))
# print(len(data))


# #将矩阵读入x_train
# for i in range(len(normalList)):
#     data.append(normalList[i]['JD'])
#
# print(data)
# data_tensor = torch.Tensor(data)
# # print(data_tensor)
# print(data_tensor[1][1].item())
# 构建模型

# 定义超参数
feature_size = 1  # 输入数据的维度
hidden_size = 64  # 隐藏层的维度
output_size = 1  # 输出数据的维度
num_epochs = 1  # 训练的轮数
learning_rate = 0.0001  # 学习率
batch_size = 1
feature_size = 1
num_layers = 64  #gru层数
timestep = 2
best_loss = 0
model_name = 'gru'
save_path = './{}.pth'.format(model_name)  # 最优模型保存路径
# 形成训练数据，例如12345789 12-3、23-4、34-5


def split_data(data, timestep, input_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:,0])
        # print(dataX)
        dataY.append(data[index + timestep][0])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, input_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, input_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]

x_train, y_train, x_test, y_test = split_data(data,timestep,feature_size)
print('x_train:')
print(x_train)
print('y_train:')
print(y_train)
# print(x_train)

# # 定义GRU模型
# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, h):
#         out, h = self.gru(x, h)
#         out = self.fc(out)
#         return out, h
#



# # 创建模型实例
# model = GRU(input_size, hidden_size, output_size)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()  # 均方误差损失函数
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器



# # 冻结模型参数
# for param in model.parameters():
#     param.requires_grad = False




# 将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           False)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          False)


# 定义GRU网络
class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden

        # GRU运算
        output, h_0 = self.gru(x, h_0)

        # 获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape

        # 将output变成 batch_size * timestep, hidden_dim
        output = output.reshape(-1, hidden_size)

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 转换维度，用于输出
        output = output.reshape(timestep, batch_size, -1)

        # 我们只需要返回最后一个时间片的数据即可
        return output[-1]

model = GRU(feature_size, hidden_size,num_layers,output_size)  # 定义GRU网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

# 模型训练
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 num_epochs,
                                                                 loss)


    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
            test_loss += loss.item()
            test_bar.desc = "test loss:{:.3f}".format(loss)

    # 计算平均损失和精确性
    test_loss = test_loss / len(test_loader)
    accuracy = 1 - test_loss / torch.mean(torch.abs(y_test_tensor))
    print("Test loss: {:.3f}, accuracy: {:.3f}".format(test_loss, accuracy))


    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), save_path)
print('Finished Training')
torch.save(model.state_dict(), save_path)




# 9.绘制结果
plot_size = 200
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b")
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
print(x_train_tensor)
print(y_train_tensor)
plt.legend()
plt.show()

y_test_pred = model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size]), "b")
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
plt.legend()
plt.show()


    # # 模型验证
    # model.eval()
    # test_loss = 0
    # with torch.no_grad():
    #     test_bar = tqdm(test_loader)
    #     for data in test_bar:
    #         x_test, y_test = data
    #         y_test_pred = model(x_test)
    #         test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
    #
    # if test_loss < config.best_loss:
    #     config.best_loss = test_loss
    #     torch.save(model.state_dict(), save_path)

print('Finished Training')

#
# # 将输入数据转换为三维张量，形状为(时间步数，批次大小，输入维度)
# x_data = torch.stack(x_data).unsqueeze(1)
# print(x_data)
# # 定义y为下一个时间步的数据，形状与x_data相同
# y_data = x_data[1:]
#
# # 将x_data和y_data划分为训练集和测试集，300个时间步为训练集，200个时间步为测试集
# x_train = x_data[:300]
# y_train = y_data[:300]
# x_test = x_data[300:]
# y_test = y_data[300:]



# # 定义GRU模型
# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, h):
#         out, h = self.gru(x, h)
#         out = self.fc(out[-1])
#         return out, h
#
#
# # 定义超参数
# input_size = 26  # 输入数据的特征维度
# hidden_size = 64  # 隐藏层的大小
# output_size = 1  # 输出数据的维度
# batch_size = 1  # 批量大小
# learning_rate = 0.01  # 学习率
# num_epochs = 10  # 训练轮数
#
# # 创建模型和优化器
# model = GRUModel(input_size, hidden_size, output_size)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()  # 均方误差损失函数
#
# # 将输入数据转换为tensor，并划分训练集和测试集
# x_data = torch.stack(x_data)  # 将list转换为tensor，形状为(500, 26)
# x_train = x_data[:300]  # 取前300个时间步为训练集，形状为(300, 26)
# x_test = x_data[300:]  # 取后200个时间步为测试集，形状为(200, 26)
# print(x_train)
# # 训练模型
# for epoch in range(num_epochs):
#     # 初始化隐藏状态
#     h = torch.zeros(1, batch_size, hidden_size)
#     # 遍历训练集，每次取一个时间步的数据
#     for i in range(len(x_train)):
#         # 获取输入和目标数据，增加一个维度作为批量大小
#         x = x_train[i].unsqueeze(0).unsqueeze(0)  # 形状为(1, 1, 26)
#         y = x_train[i + 1].unsqueeze(0).unsqueeze(0)  # 形状为(1, 1, 26)，目标是下一个时间步的数据
#         # 前向传播
#         output, h = model(x, h)
#         # 计算损失
#         loss = criterion(output, y)
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # 打印每轮的损失值
#     print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#
# # 测试模型
# # 初始化隐藏状态和预测结果列表
# h = torch.zeros(1, batch_size, hidden_size)
# predictions = []
# # 遍历测试集，每次取一个时间步的数据
# for i in range(len(x_test)):
#     # 获取输入数据，增加一个维度作为批量大小
#     x = x_test[i].unsqueeze(0).unsqueeze(0)  # 形状为(1, 1, 26)
#     # 前向传播，得到预测值，并添加到列表中
#     output, h = model(x, h)
#     predictions.append(output.squeeze().tolist())
# # 计算预测结果和真实结果的均方误差和精确度（假设输出是二分类问题）
# predictions = torch.tensor(predictions)  # 将列表转换为tensor，形状为(200,)
# y_test = x_test[1:]  # 取测试集的后199个时间步作为真实结果，形状为(199,)
# mse = criterion(predictions[:-1], y_test)  # 计算均方误差，去掉最后一个预测值，因为没有对应的真实值
# accuracy = torch.sum(predictions[:-1].round() == y_test).float() / len(y_test)  # 计算精确度，将预测值四舍五入，然后和真实值比较
# # 打印测试结果
# print(f"Test MSE: {mse.item():.4f}")
# print(f"Test Accuracy: {accuracy.item():.4f}")





#
# # 定义超参数
# hidden_size = 128 # 隐藏层大小
# num_layers = 2 # gru层数
# output_size = 1 # 输出大小
# batch_size = 26 # 批量大小
# seq_len = 500 # 序列长度
# learning_rate = 0.01 # 学习率
# num_epochs = 10 # 训练轮数
#
# # 定义gru模型
# class GRUModel(nn.Module):
#     def __init__(self, hidden_size, num_layers, output_size):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # 初始化隐藏状态
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         # 通过gru层得到输出和最后一个隐藏状态
#         out, hn = self.gru(x, h0)
#         # 只取最后一个时间步的输出
#         out = out[:, -1, :]
#         # 通过全连接层得到最终输出
#         out = self.fc(out)
#         return out
#
# # 实例化模型
# model = GRUModel(hidden_size, num_layers, output_size)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss() # 均方误差损失函数，适用于回归问题
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器
#
# # 定义输入数据和验证集，假设x_train是一个list，包含26个tensor，每个tensor长度为500，代表500个时间步
# x_train = torch.stack(x_train) # 将list转换为tensor，形状为[batch_size, seq_len]
# x_train = x_train.unsqueeze(-1) # 增加一个维度，形状为[batch_size, seq_len, 1]
# print(x_train)
# y_train = torch.randn(batch_size, 1) # 随机生成一些目标值，形状为[batch_size, 1]
# x_val = torch.randn(batch_size, seq_len, 1) # 随机生成一些验证数据，形状为[batch_size, seq_len, 1]
# y_val = torch.randn(batch_size, 1) # 随机生成一些验证目标值，形状为[batch_size, 1]
#
#
# # 将数据转换为float类型，以避免类型不匹配的错误
# x_train = x_train.float()
# y_train = y_train.float()
# x_val = x_val.float()
# y_val = y_val.float()
#
# # 定义一个函数来计算精确度，假设精确度是指预测值和真实值之间的相对误差小于10%的比例
# def accuracy(y_pred, y_true):
#     relative_error = torch.abs(y_pred - y_true) / y_true # 计算相对误差
#     acc = torch.mean((relative_error < 0.1).float()) # 计算小于10%的比例，并取平均值
#     return acc
#
#
# # 开始训练
# for epoch in range(num_epochs):
#     # 前向传播
#     y_pred = model(x_train)
#     # 计算损失和精确度
#     loss = criterion(y_pred, y_train)
#     acc = accuracy(y_pred, y_train)
#     # 反向传播和更新参数
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # 打印训练损失和精确度
#     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
#     # 在验证集上评估模型
#     with torch.no_grad():
#         y_val_pred = model(x_val)
#         val_loss = criterion(y_val_pred, y_val)
#         val_acc = accuracy(y_val_pred, y_val)
#         print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc.item():.4f}")