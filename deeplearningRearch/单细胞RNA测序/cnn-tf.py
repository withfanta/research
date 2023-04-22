import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import re
# 读取数据
data = pd.read_csv('data.csv', index_col=0)
# 去除列名中的特殊字符
data.columns = [col.replace('.', '') for col in data.columns]
data.columns = [re.sub(r'\d+', '', col) for col in data.columns]
# 划分数据集
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
# 将每个细胞的基因表达转换为矩阵
# 每个细胞的基因表达矩阵大小为 (gene_num, 1)
# 将每个细胞的基因表达矩阵按行拼接起来，形成数据矩阵，大小为 (cell_num, gene_num)
# 不足的部分补0
def get_data_matrix(data):
    gene_num = len(data.index)
    cell_num = len(data.columns)
    data_matrix = np.zeros((cell_num, gene_num))
    for i, cell in enumerate(data.columns):
        data_matrix[i, :] = np.array(data[cell])
    return data_matrix

train_data_matrix = get_data_matrix(train_data)
test_data_matrix = get_data_matrix(test_data)

# 将细胞类型字符串转换为数字
def get_label(data):
    label_dict = {'NP': 0, 'PEP': 1, 'NF': 2, 'TH': 3}
    label = [label_dict[cell_type] for cell_type in data.columns]
    return to_categorical(label)

train_label = get_label(train_data)
test_label = get_label(test_data)

# 构建卷积神经网络模型
model = Sequential()
# 输入大小为 (cell_num, gene_num)
# 第一层卷积层，卷积核大小为 3，个数为 16，步长为 1，输出大小为 (cell_num-2, 16)
model.add(Conv1D(16, 3, input_shape=train_data_matrix.shape, activation='relu'))
# 第一层池化层，池化大小为 2，步长为 2，输出大小为 (cell_num//2-1, 16)
model.add(MaxPooling1D(pool_size=2, strides=2))
# 第二层卷积层，卷积核大小为 3，个数为 32，步长为 1，输出大小为 (cell_num//2-3, 32)
model.add(Conv1D(32, 3, activation='relu'))
# 第二层池化层，池化大小为 2，步长为 2，输出大小为 (cell_num//4-1, 32)
model.add(MaxPooling1D(pool_size=2, strides=2))
# 将输出展平，输出大小为 (cell_num//4-1)*32
model.add(Flatten())
# 全连接层，输出大小为 32
model.add(Dense(32, activation='relu'))
# 输出层，输出大小为 4，对应四种细胞类型
model.add(Dense(4, activation='softmax'))

# 编译模型，使用交叉熵损失函数和Adam优化器
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_data_matrix, train_label, epochs=10, batch_size=32, validation_data=(test_data_matrix, test_label))

# 在测试集上评估模型
score = model.evaluate(test_data_matrix, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 输出混淆矩阵和准确率
y_pred = model.predict(test_data_matrix)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_label, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
print('Confusion matrix:\n', confusion_mtx)
print('Accuracy:', accuracy_score(y_true_classes, y_pred_classes))
