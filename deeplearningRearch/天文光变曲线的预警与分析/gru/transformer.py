import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
import math


class Config():
    data_path = 'sourceData/normalData/ref_023_15730595-G0013_356332_4877'
    # data_path = 'wind.csv'
    timestep = 5  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    hidden_size = 256  # 隐层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    num_layers = 10  # lstm的层数
    transformer_num_layers = 10  # transformer层数
    epochs = 50  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.000000000001  # 学习率
    model_name = 'transformer'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()

# # 1.加载时间序列数据
# df = pd.read_csv('wind.csv', index_col=0)
# # 2.将数据进行标准化
# print(df)
# scaler = MinMaxScaler()
# scaler_model = MinMaxScaler()
# data = scaler_model.fit_transform(np.array(df))
# scaler.fit_transform(np.array(df['WIND']).reshape(-1, 1))
#
# print(data)


# 1.加载时间序列数据
df = pd.read_csv('sourceData/normalData/ref_023_15730595-G0013_356332_4877', sep=' ', header=None,skiprows=1, names=['JD', 'Magnorm', 'Mage'])
df = df[:200]
# 2.将数据进行标准化
print(df)
scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['JD']).reshape(-1, 1))

print(data)

# # 2.将数据进行标准化
# print(df)
# scaler = MinMaxScaler()
# scaler_model = MinMaxScaler()
# data = np.array(df)
# scaler.fit_transform(np.array(df['JD']).reshape(-1, 1))
#
# print(data)

# 形成训练数据，例如12345789 12-3456789
def split_data(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:, 0])
        dataY.append(data[index + timestep][0])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]


# 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)
print(x_train)
print(y_train)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           False)

test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, hidden_size, num_layers, feature_size, output_size, feedforward_dim=32, num_head=1,
                 transformer_num_layers=1, dropout=0.3, max_len=1):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        # 位置编码层
        self.positional_encoding = PositionalEncoding(hidden_size, dropout, max_len)

        # 编码层
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_head, feedforward_dim, dropout,
                                                        batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, transformer_num_layers)

        # 输出层
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

        # 维度为【序列长度，批次，嵌入向量维度】
        output = self.positional_encoding(output)
        # 维度为【序列长度，批次，嵌入向量维度】
        output = self.transformer(output)
        # 将每个词的输出向量取均值，也可以随意取一个标记输出结果，维度为【批次，嵌入向量维度】
        output = output.mean(axis=1)
        # 进行分类，维度为【批次，分类数】
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output


model = Transformer(config.hidden_size, config.num_layers, config.feature_size, config.output_size,
                    transformer_num_layers=config.transformer_num_layers)
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器

# 8.模型训练
for epoch in range(config.epochs):
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
                                                                 config.epochs,
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

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), config.save_path)
print('Finished Training')
torch.save(model.state_dict(), config.save_path)

print('Finished Training')

# 9.绘制结果
plot_size = 1000
plt.figure(figsize=(12, 8))
plt.plot((model(x_train_tensor).detach().numpy()[: plot_size].reshape(-1, 1)), "b")
plt.plot(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size], "r")
plt.legend()
plt.show()

y_test_pred = model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(y_test_pred.detach().numpy()[: plot_size], "b")
plt.plot(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size], "r")
plt.legend()
plt.show()

#
# # 9.绘制结果
# plot_size = 1000
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b")
# plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
# plt.legend()
# plt.show()
#
# y_test_pred = model(x_test_tensor)
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size]), "b")
# plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
# plt.legend()
# plt.show()