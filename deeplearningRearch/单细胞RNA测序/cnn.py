import tensorflow as tf
import pandas as pd
from keras.layers import Activation
from keras.optimizers import RMSprop
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score


learning_rate = 0.001 #学习率
batch_size = 128 #批大小
num_steps = 5000 #使用的样本数量
display_step = 50 #显示间隔

num_input = 784 # image shape:28*28
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 #用于随机丢弃，防止过拟

#把文件读为矩阵
def readAsMartix(filename):
    data_read = pd.read_csv(filename, header=None, index_col=0, low_memory=False)
    # data_read = dataframe_allowing_duplicate_headers(filename)
    # print(data_read.columns.values.tolist())
    # print(data_read[:10])
    return data_read
# 将每一列加入
def addCol(data_read):
    x = []
    y = []
    for col in data_read.columns[0:]:
        str_data_col = data_read[col].tolist()
        data_col = [float(x) for x in str_data_col[1:]]
        x.append(data_col)
        if str_data_col[:1] == ['TH']:
            y.append(0)
        if str_data_col[:1] == ['NF']:
            y.append(1)
        if str_data_col[:1] == ['NP']:
            y.append(2)
        if str_data_col[:1] == ['PEP']:
            y.append(3)
    # print(y)
    # print(x[0])
    return x,y

# 将矩阵变为160*160
def countMartix(martix):
    zero = tf.zeros((1, 25600))
    martixTensor = tf.convert_to_tensor(martix)
    # martixTensor = tf.reshape(martixTensor,[1,-1])
    #print(martixTensor.shape[1])
    sub = zero.shape[1] - martixTensor.shape[1]
    zeroadd = tf.zeros((martixTensor.shape[0], sub))
    result = tf.concat([martixTensor,zeroadd],1)
    # zero_vector = tf.reshape(x_train, [-1])
    # zero_padding = tf.zeros([160 * 160] - tf.shape(zero_vector), dtype=zero_vector.dtype)
    # zero_padded = tf.concat([zero_vector, zero_padding], 0)
    # result = tf.reshape(zero_padded,[160,160])
    result = tf.reshape(result,[-1,160,160,1])
    #print(sub)
    return result



data_read = readAsMartix('data.csv')
x,y = addCol(data_read)
y = np.reshape(y,(-1,1))
#y = tf.convert_to_tensor(y)
#print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

x_train = countMartix(x_train)
x_test = countMartix(x_test)
# print(x_train)
# print(y_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

# 定义模型
model = tf.keras.models.Sequential([
# 第一层卷积层，卷积核大小为 3，个数为 16，步长为 1，输出大小为158, 158, 16
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(160, 160, 1)),
# 第一层池化层，池化大小为 2，步长为 1，输出大小为 79, 79, 16
    tf.keras.layers.MaxPooling2D(2,2),
# 第二层卷积层，卷积核大小为 3，个数为 32，步长为 1，输出大小为77, 77, 32
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
# 第二层池化层，池化大小为 2，步长为 1，输出大小为38, 38, 32
    tf.keras.layers.MaxPooling2D(2,2),
# 第三层卷积层，卷积核大小为 3，步长为 1，输出大小为36, 36, 64
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# 第三层池化层，池化大小为 2，步长为 1，输出大小为18, 18, 64
    tf.keras.layers.MaxPooling2D(2,2),
# 将输出展平，输出大小为 20736
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型

model.fit(x_train, y_train, epochs=50, batch_size=20, validation_data=(x_test, y_test))

# 在测试集上评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# 输出混淆矩阵和准确率
y_pred = model.predict(x_train)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_train, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
print('训练集集混淆矩阵:\n', confusion_mtx)
print('训练集准确率:', accuracy_score(y_true_classes, y_pred_classes))
y_pred_test = model.predict(x_test)
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_true_classes_test = np.argmax(y_test, axis=1)
confusion_mtx_test = confusion_matrix(y_true_classes_test, y_pred_classes_test)
print('测试集混淆矩阵:\n', confusion_mtx_test)
print('测试集准确率:', accuracy_score(y_true_classes_test, y_pred_classes_test))
