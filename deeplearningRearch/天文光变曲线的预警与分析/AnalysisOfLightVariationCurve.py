import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def readAsMatrix(filename):
    data_read = pd.read_csv(filename, sep=' ', header=None,
                                   skiprows=1, names=['JD', 'Magnorm', 'Mage'])
    return data_read
target_dara_read =  readAsMatrix('ref_044_16280425-G0013_364820_9174')
normal_data_read1 = readAsMatrix('044_16280425-G0013/ref_044_16280425-G0013_331771_5921.txt')
# normal_data_read2 = readAsMatrix('044_16280425-G0013/ref_044_16280425-G0013_365364_11080.txt')
# print(np.array(normal_data_read1['JD'][0:50]))
# print(x.shape)
# print(x)
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

# 将获取一个二维列表，列表中的每一个元素都是50个x,y点对,并将50个点压缩至图片中间
x1,y1 = imgSpilt(target_dara_read,'JD','Magnorm',50)
x,y = imgSpilt(normal_data_read1,'JD','Magnorm',50)
# print(y)
for i in range(len(x)):
    # print(y[i][0])
    # plt.ylim(y[i][0]-1, y[i][0]+1)
    # plt.scatter(x[i], y[i], label="normalstar", color="blue")
    plt.ylim(y1[i][0]-1, y1[i][0]+1)
    plt.scatter(x1[i], y1[i], label="targetstar", color="red")
    plt.show()

# # 将获取两个个二维列表，列表中的每一个元素都是50个x,y点对，将两张图进行对比
# x1,y1 = imgSpilt(target_dara_read,'JD','Magnorm',50)
# x,y = imgSpilt(normal_data_read1,'JD','Magnorm',50)
# for i in range(len(x)):
#     plt.scatter(x[i], y[i], label="normalstar", color="blue")
#     plt.scatter(x1[i], y1[i], label="targetstar", color="red")
#     plt.show()



# targrt_data_read = readAsMatrix('ref_044_16280425-G0013_364820_9174')
# x1 = normal_data_read1['JD'][2000:2050]
# print(x1.shape[0])
# y1 = normal_data_read1['Magnorm'][2000:2050]
# x_target = targrt_data_read['JD'][2000:2050]
# y_target = targrt_data_read['Magnorm'][2000:2050]
# x2 = normal_data_read1['JD'][2000:2050]
# y2 = normal_data_read1['Magnorm'][2000:2050]
# def readnp(filename):
#     f = open(filename)
#     lines = f.readlines()
#     rows = len(lines)
#     for l in lines:
#         col = l.strip('\n').split(' ')
#         print(col)
#     return lines

# x = readnp('044_16280425-G0013/ref_044_16280425-G0013_331771_5921.txt')
# plt.scatter(x1,y1,label = "normalstar",color="blue")
# plt.scatter(x2,y2,label = "normalstar",color="green")
# plt.scatter(x_target,y_target,label = "targetstar",color="red")
#
# plt.legend(loc='upper right')
# plt.show()
